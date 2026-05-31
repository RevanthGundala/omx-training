"""Private helpers retained by the new ``evaluation.dagger_quic`` driver.

Only the modules useful as building blocks for the DAgger-over-QUIC port
remain: ``colors``, ``quic`` (connection setup), ``preflight``
(payload builder), ``scene_check`` (one-shot YOLO coverage check), and
``artifacts`` (event/run JSONL writers).  Everything else was deleted
when control flow moved to LeRobot's ``DAggerStrategy``.
"""
