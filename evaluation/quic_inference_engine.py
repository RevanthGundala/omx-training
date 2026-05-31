"""QuicInferenceEngine — LeRobot ``InferenceEngine`` backed by remote QUIC.

A background thread polls a cached observation, ships it to the Modal
PI0.5 server over QUIC (``omx_quic.QuicClient``), receives an action
chunk back, and pushes the actions onto a thread-safe queue.  The main
control loop pops one action per tick via :meth:`get_action`.

The server already unnormalizes actions, so the chunks we enqueue are in
physical joint ``.pos`` space.  Pair this engine with identity
pre/postprocessors in the rollout context to keep LeRobot from
double-processing.

Closely modeled on ``lerobot/rollout/inference/rtc.py`` so it slots into
``RolloutStrategy`` (and ``DAggerStrategy`` in particular) unmodified.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from collections import deque
from threading import Event, Lock, Thread

import numpy as np
import torch

from lerobot.rollout.inference.base import InferenceEngine

from evaluation._pi0_quic.preflight import build_payload


logger = logging.getLogger(__name__)


_IDLE_SLEEP_S: float = 0.01
_ERROR_RETRY_DELAY_S: float = 0.5
_MAX_CONSECUTIVE_ERRORS: int = 10
_JOIN_TIMEOUT_S: float = 3.0
_REQUEST_TIMEOUT_S: float = 30.0


class QuicInferenceEngine(InferenceEngine):
    """Async remote inference over QUIC.

    Parameters
    ----------
    quic_client:
        A connected ``omx_quic.QuicClient`` instance (handshake already
        completed by the caller).
    hw_features:
        Hardware-side dataset features dict — used by ``build_payload``
        to extract ``observation.state`` from the raw obs.
    ordered_action_keys:
        Robot action key order (e.g. ``[..., "wrist_roll.pos", "gripper.pos"]``)
        used to map the server's flat action vector back into a tensor
        the strategy can unpack with ``ordered_action_keys``.
    refill_threshold:
        Trigger a new request as soon as the local queue dips to or
        below this many actions.  Matches ``RTC_QUEUE_REFILL_THRESHOLD``
        used by the legacy worker.
    """

    def __init__(
        self,
        *,
        quic_client,
        hw_features: dict,
        ordered_action_keys: list[str],
        refill_threshold: int = 30,
        shutdown_event: Event | None = None,
    ) -> None:
        self._client = quic_client
        self._hw_features = hw_features
        self._ordered_action_keys = list(ordered_action_keys)
        self._refill_threshold = refill_threshold
        self._global_shutdown_event = shutdown_event

        self._queue: deque[torch.Tensor] = deque()
        self._queue_lock = Lock()
        self._last_action: torch.Tensor | None = None
        # Number of actions popped from the queue since the last chunk was
        # received from the server.  We pass this back to the server as
        # ``prev_steps_consumed`` so its RTC inpainting anchors against the
        # correct slice of the previous chunk; otherwise the new chunk's
        # prefix collapses onto the start of the previous chunk and the
        # robot jerks back to the action that was already executed.
        self._consumed_since_last_chunk: int = 0

        self._obs_lock = Lock()
        self._obs: dict | None = None

        self._policy_active = Event()
        self._shutdown_event = Event()
        self._failed = Event()
        self._thread: Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return True

    @property
    def failed(self) -> bool:
        return self._failed.is_set()

    def start(self) -> None:
        self._shutdown_event.clear()
        self._failed.clear()
        self._thread = Thread(target=self._loop, daemon=True, name="QuicInference")
        self._thread.start()
        logger.info("QuicInferenceEngine started")

    def stop(self) -> None:
        logger.info("Stopping QuicInferenceEngine")
        self._shutdown_event.set()
        self._policy_active.clear()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                logger.warning("QUIC inference thread did not join within %.1fs", _JOIN_TIMEOUT_S)
            self._thread = None

    def pause(self) -> None:
        logger.info("Pausing QuicInferenceEngine")
        self._policy_active.clear()

    def resume(self) -> None:
        logger.info("Resuming QuicInferenceEngine")
        self._policy_active.set()

    def reset(self) -> None:
        """Clear local queue and request a fresh chunk from the server."""
        logger.info("Resetting QuicInferenceEngine (queue + server RTC state)")
        with self._queue_lock:
            self._queue.clear()
            self._last_action = None
            self._consumed_since_last_chunk = 0
        try:
            self._client.request(json.dumps({"op": "reset"}).encode("utf-8"), _REQUEST_TIMEOUT_S)
        except Exception as exc:
            logger.warning("Server reset request failed: %s", exc)

    # ------------------------------------------------------------------
    # Main-thread API
    # ------------------------------------------------------------------

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Pop the next action; hold last action if queue empty."""
        with self._queue_lock:
            if self._queue:
                action = self._queue.popleft()
                self._last_action = action
                self._consumed_since_last_chunk += 1
                return action
            return self._last_action.clone() if self._last_action is not None else None

    def notify_observation(self, obs: dict) -> None:
        """Cache the latest raw observation for the worker thread."""
        with self._obs_lock:
            self._obs = obs

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _qsize(self) -> int:
        with self._queue_lock:
            return len(self._queue)

    def _replace_chunk(self, actions: np.ndarray, *, chunk_consumed_offset: int) -> None:
        """Install a new chunk, skipping actions whose ticks already passed.

        ``chunk_consumed_offset`` is the number of actions at the *front*
        of the new chunk that we've effectively executed already via the
        previous chunk while waiting for the server response.  RTC
        inpainting guarantees ``new[:k] ≈ prev[prev_steps_consumed:prev_steps_consumed+k]``,
        so the first ``offset`` rows are already in the past.
        """
        with self._queue_lock:
            self._queue.clear()
            offset = max(0, min(chunk_consumed_offset, actions.shape[0]))
            for row in actions[offset:]:
                self._queue.append(torch.from_numpy(np.asarray(row, dtype=np.float32)))
            # The new chunk's t=0 is the request moment; we've already
            # consumed ``offset`` of its actions during the request RTT.
            self._consumed_since_last_chunk = offset

    def _loop(self) -> None:
        from lerobot.utils.feature_utils import build_dataset_frame
        from lerobot.utils.constants import OBS_STR

        consecutive_errors = 0
        first_call = True
        try:
            while not self._shutdown_event.is_set():
                if not self._policy_active.is_set():
                    time.sleep(_IDLE_SLEEP_S)
                    continue

                with self._obs_lock:
                    obs = self._obs
                if obs is None:
                    time.sleep(_IDLE_SLEEP_S)
                    continue

                if not first_call and self._qsize() > self._refill_threshold:
                    time.sleep(_IDLE_SLEEP_S)
                    continue

                try:
                    obs_frame = build_dataset_frame(self._hw_features, obs, prefix=OBS_STR)
                    with self._queue_lock:
                        prev_steps_consumed = self._consumed_since_last_chunk
                    payload = build_payload(
                        obs_frame,
                        obs,
                        inference_delay=prev_steps_consumed,
                        prev_steps_consumed=prev_steps_consumed,
                    )
                    t0 = time.perf_counter()
                    resp_bytes = self._client.request(payload, _REQUEST_TIMEOUT_S)
                    rtt_ms = (time.perf_counter() - t0) * 1000.0
                    data = json.loads(resp_bytes)
                    if "error" in data:
                        raise RuntimeError(f"server error: {data['error']}")
                    actions = np.asarray(data["actions"], dtype=np.float32)
                    if actions.ndim != 2 or actions.shape[1] != len(self._ordered_action_keys):
                        raise RuntimeError(
                            f"unexpected action shape {actions.shape}; "
                            f"expected (T, {len(self._ordered_action_keys)})"
                        )
                    # Actions executed while we were waiting for this
                    # response correspond to the first ``consumed_during_request``
                    # actions of the new chunk (the server's RTC inpainting
                    # has already aligned its front with our trajectory).
                    with self._queue_lock:
                        consumed_during_request = max(
                            0, self._consumed_since_last_chunk - prev_steps_consumed
                        )
                    self._replace_chunk(
                        actions,
                        chunk_consumed_offset=consumed_during_request,
                    )
                    logger.debug(
                        "QUIC chunk: T=%d, RTT=%.1fms, queue=%d, prev_consumed=%d, in_flight=%d",
                        actions.shape[0], rtt_ms, self._qsize(),
                        prev_steps_consumed, consumed_during_request,
                    )
                    consecutive_errors = 0
                    first_call = False
                except Exception as exc:
                    consecutive_errors += 1
                    logger.error(
                        "QUIC inference error (%d/%d): %s",
                        consecutive_errors, _MAX_CONSECUTIVE_ERRORS, exc,
                    )
                    logger.debug(traceback.format_exc())
                    if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                        raise
                    time.sleep(_ERROR_RETRY_DELAY_S)
        except Exception as exc:
            logger.error("Fatal QuicInferenceEngine error: %s", exc)
            logger.error(traceback.format_exc())
            self._failed.set()
            if self._global_shutdown_event is not None:
                self._global_shutdown_event.set()
