"""dagger_quic.py — LeRobot ``DAggerStrategy`` driven by remote QUIC PI0.5.

This is the porting replacement for ``evaluation/eval_pi0_quic.py``.

Setup:

  # Terminal 1: launch the Modal QUIC server.
  uv run modal run --detach deploy/serve_pi0_quic_modal.py \\
    --session-id dagger-run \\
    --checkpoint-repo-id RevanthGundala/pi05-pour-water-new35-from-base-globalstats \\
    --dataset-repo-id RevanthGundala/004-pour-water-new35-globalstats

  # Terminal 2: run the local DAgger driver.
  uv run python evaluation/dagger_quic.py --session-id dagger-run --save-corrections

Keyboard (LeRobot defaults — rebind via flags if desired):
  space  pause / resume policy
  tab    start / stop correction recording
  enter  upload dataset to hub
  ESC    stop session

Per-correction-window an episode is appended to the corrections dataset
(default ``RevanthGundala/005-pour-water-dagger-corrections``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.robots.omx_follower.config_omx_follower import OmxFollowerConfig
from lerobot.rollout.configs import DAggerKeyboardConfig, DAggerStrategyConfig
from lerobot.teleoperators.omx_leader.config_omx_leader import OmxLeaderConfig
from lerobot.utils.process import ProcessSignalHandler

from evaluation._pi0_quic.colors import CYAN, DIM, GREEN, RED, YELLOW, color
from evaluation._pi0_quic.quic import connect_quic
from evaluation.logged_dagger_strategy import LoggedDAggerStrategy
from evaluation.quic_rollout_context import build_quic_rollout_context
from utils.config import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CAMERAS,
    FOLLOWER_PORT,
    FPS,
    LEADER_PORT,
    TASK_NAME,
)


logger = logging.getLogger(__name__)


# Default HF repo for DAgger correction episodes.  Hoisted here from the
# (now-deleted) ``evaluation._pi0_quic.corrections`` module so this script
# has no legacy dependency.
DEFAULT_CORRECTION_DATASET_REPO_ID = "RevanthGundala/005-pour-water-dagger-corrections"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DAgger driver for PI0.5 over QUIC")
    p.add_argument("--session-id", required=True, help="QUIC rendezvous session id (must match server)")
    p.add_argument("--stun", default="stun.l.google.com:19302")
    p.add_argument(
        "--correction-dataset-repo-id",
        default=DEFAULT_CORRECTION_DATASET_REPO_ID,
        help="HF repo id for the corrections dataset.",
    )
    p.add_argument("--num-episodes", type=int, default=10, help="Stop after this many corrections.")
    p.add_argument("--task", default=TASK_NAME)
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume an existing corrections dataset.")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Directory for side-channel run artifacts (default: outputs/dagger_runs/<ts>).",
    )
    p.add_argument(
        "--pause-key", default="space",
        help="Key to toggle pause/resume (LeRobot default: 'space').",
    )
    p.add_argument(
        "--correction-key", default="tab",
        help="Key to toggle correction recording (LeRobot default: 'tab').",
    )
    p.add_argument(
        "--upload-key", default="enter",
        help="Key to trigger hub upload (LeRobot default: 'enter').",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------


def _camera_configs() -> dict[str, OpenCVCameraConfig]:
    return {
        name: OpenCVCameraConfig(
            index_or_path=index,
            fps=FPS,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            warmup_s=5,
        )
        for name, index in CAMERAS.items()
    }


def _build_cfg(args: argparse.Namespace):
    """Assemble a duck-typed RolloutConfig that DAggerStrategy can consume.

    We avoid ``lerobot.rollout.configs.RolloutConfig`` directly because its
    ``__post_init__`` requires a local ``--policy.path`` which we deliberately
    don't have (the policy lives on the Modal server).  ``DAggerStrategy``
    only reads attributes off ``ctx.runtime.cfg``, so a ``SimpleNamespace``
    that exposes the same surface area works identically.
    """
    robot_cfg = OmxFollowerConfig(
        port=FOLLOWER_PORT, id="omx_follower_arm", cameras=_camera_configs()
    )
    teleop_cfg = OmxLeaderConfig(port=LEADER_PORT, id="omx_leader_arm")

    dataset_cfg = DatasetRecordConfig(
        repo_id=args.correction_dataset_repo_id,
        single_task=args.task,
        fps=FPS,
        num_episodes=args.num_episodes,
        push_to_hub=args.push_to_hub,
        # Streaming encoding lets episode saves stay off the control thread.
        streaming_encoding=True,
    )

    strategy_cfg = DAggerStrategyConfig(
        num_episodes=args.num_episodes,
        record_autonomous=False,
        input_device="keyboard",
        keyboard=DAggerKeyboardConfig(
            pause_resume=args.pause_key,
            correction=args.correction_key,
            upload=args.upload_key,
        ),
    )

    return SimpleNamespace(
        robot=robot_cfg,
        teleop=teleop_cfg,
        policy=None,
        strategy=strategy_cfg,
        inference=None,
        dataset=dataset_cfg,
        fps=FPS,
        duration=0.0,
        interpolation_multiplier=1,
        device="cpu",
        task=args.task,
        display_data=False,
        display_ip=None,
        display_port=None,
        display_compressed_images=False,
        play_sounds=False,
        resume=args.resume,
        rename_map={},
        return_to_initial_position=True,
        use_torch_compile=False,
        torch_compile_backend="inductor",
        torch_compile_mode="default",
        compile_warmup_inferences=2,
        rtc_config=None,
    )


# ---------------------------------------------------------------------------
# Server health probe
# ---------------------------------------------------------------------------


def _health_probe(client) -> None:
    try:
        resp = client.request(json.dumps({"op": "health"}).encode("utf-8"), 10.0)
        data = json.loads(resp)
        print(color("[client] server ready", GREEN))
        if isinstance(data, dict):
            print(color(
                "  "
                f"checkpoint={data.get('checkpoint_source')} | "
                f"dataset={data.get('dataset_repo_id')} | "
                f"stats={data.get('stats_source')} | "
                f"relative={data.get('use_relative_actions')}",
                DIM,
            ))
    except Exception as exc:
        print(color(f"[client] health probe failed: {exc}", YELLOW))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    run_dir = args.run_dir or (
        Path("outputs/dagger_runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(color(f"Run artifacts → {run_dir}", DIM))

    print(color(f"Connecting QUIC for session={args.session_id!r} ...", CYAN))
    client = connect_quic(args.session_id, args.stun)
    _health_probe(client)

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    cfg = _build_cfg(args)
    print(color("Building rollout context (connecting robot + teleop + dataset) ...", CYAN))
    ctx = build_quic_rollout_context(cfg, shutdown_event, quic_client=client)

    strategy = LoggedDAggerStrategy(cfg.strategy, run_dir=run_dir)
    print(color(
        f"DAgger ready. Keys: pause/resume={args.pause_key}  "
        f"correction={args.correction_key}  upload={args.upload_key}  (ESC to stop)",
        GREEN,
    ))
    print(color(f"Target: {args.num_episodes} correction episodes → "
                f"{args.correction_dataset_repo_id}", GREEN))

    try:
        strategy.setup(ctx)
        strategy.run(ctx)
    except KeyboardInterrupt:
        print(color("Interrupted by user", YELLOW))
    except Exception as exc:
        print(color(f"Fatal error: {exc}", RED))
        logger.exception("Fatal error in DAgger run")
        raise
    finally:
        try:
            strategy.teardown(ctx)
        finally:
            try:
                client.close()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
