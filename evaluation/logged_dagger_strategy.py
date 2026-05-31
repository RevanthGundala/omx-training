"""LoggedDAggerStrategy — DAgger + our event/run artifacts.

Subclasses :class:`lerobot.rollout.strategies.dagger.DAggerStrategy` to
intercept phase transitions and emit them to our ``EventLogger``.  The
control loop itself is inherited verbatim.

The strategy keeps a single :class:`EventLogger` and (optionally) an
:class:`EvalRunLogger` so per-run JSONL artifacts continue to land in
``runs/<timestamp>/`` exactly as before.
"""

from __future__ import annotations

import logging
from pathlib import Path

from lerobot.rollout.strategies.dagger import DAggerStrategy

from evaluation._pi0_quic.artifacts import EvalRunLogger, EventLogger


logger = logging.getLogger(__name__)


class LoggedDAggerStrategy(DAggerStrategy):
    """DAgger with side-channel artifact logging.

    Parameters
    ----------
    config:
        Standard ``DAggerStrategyConfig``.
    run_dir:
        Directory to write ``event_log.jsonl`` and run metadata into.
        Created if missing.
    """

    def __init__(self, config, *, run_dir: Path):
        super().__init__(config)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.event_logger = EventLogger(run_dir)
        self.run_logger = EvalRunLogger(run_dir)
        logger.info("LoggedDAggerStrategy artifacts: %s", run_dir)

    # ------------------------------------------------------------------
    # Lifecycle (delegate then log)
    # ------------------------------------------------------------------

    def setup(self, ctx) -> None:
        super().setup(ctx)
        self.event_logger.log(
            "dagger_setup",
            num_episodes=self.config.num_episodes,
            input_device=self.config.input_device,
            record_autonomous=self.config.record_autonomous,
        )

    def teardown(self, ctx) -> None:
        try:
            super().teardown(ctx)
        finally:
            self.event_logger.log("dagger_teardown")
            self.event_logger.close()
            self.run_logger.close()

    # ------------------------------------------------------------------
    # Phase transition hook
    # ------------------------------------------------------------------
    # NOTE: ``DAggerStrategy._apply_transition`` is a ``@staticmethod`` but is
    # invoked from the ``_run_*`` loops as ``self._apply_transition(...)``.
    # Overriding it as a plain instance method here (no ``@staticmethod``
    # decorator) gives us a ``self`` bind so we can log, while still resolving
    # via the MRO.
    def _apply_transition(self, old_phase, new_phase, engine, interpolator, ctx, prev_action):
        self.event_logger.log(
            "dagger_phase_transition",
            old_phase=old_phase.value,
            new_phase=new_phase.value,
            has_prev_action=prev_action is not None,
        )
        DAggerStrategy._apply_transition(old_phase, new_phase, engine, interpolator, ctx, prev_action)
