"""Build a :class:`RolloutContext` for QUIC-backed remote inference.

Mirrors ``lerobot.rollout.context.build_rollout_context`` but skips the
local-policy loading step entirely.  The Modal PI0.5 server is the real
policy; we only need the robot + teleop + dataset + a
:class:`QuicInferenceEngine` plugged into the context.

The returned context has the exact shape ``DAggerStrategy`` expects, so
it composes with no upstream patching.
"""

from __future__ import annotations

import logging
from threading import Event
from types import SimpleNamespace

from lerobot.datasets import LeRobotDataset, aggregate_pipeline_dataset_features, create_initial_features
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.rollout.configs import RolloutConfig
from lerobot.rollout.context import (
    DatasetContext,
    HardwareContext,
    PolicyContext,
    ProcessorContext,
    RolloutContext,
    RuntimeContext,
)
from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.feature_utils import combine_feature_dicts, hw_to_dataset_features

from evaluation.quic_inference_engine import QuicInferenceEngine


logger = logging.getLogger(__name__)


def build_quic_rollout_context(
    cfg: RolloutConfig,
    shutdown_event: Event,
    *,
    quic_client,
) -> RolloutContext:
    """Wire up hardware, dataset, and QUIC inference into a RolloutContext.

    Parameters
    ----------
    cfg:
        Standard ``RolloutConfig``.  ``cfg.policy`` may be ``None`` —
        the QUIC engine carries no local model.
    shutdown_event:
        Threading event signalled when the process should shut down.
    quic_client:
        A connected ``omx_quic.QuicClient`` instance.  The caller owns
        the handshake; this builder only attaches it to the engine.
    """
    if cfg.robot is None:
        raise ValueError("--robot.type is required")

    teleop_action_proc, robot_action_proc, robot_obs_proc = make_default_processors()

    # --- Hardware -----------------------------------------------------
    logger.info("Connecting robot (%s)...", cfg.robot.type)
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    initial_obs = robot.get_observation()
    initial_position = {k: v for k, v in initial_obs.items() if k.endswith(".pos")}
    robot_wrapper = ThreadSafeRobot(robot)

    teleop = None
    if cfg.teleop is not None:
        logger.info("Connecting teleoperator (%s)...", cfg.teleop.type)
        teleop = make_teleoperator_from_config(cfg.teleop)
        teleop.connect()

    # --- Features (same logic as upstream) ----------------------------
    all_obs = robot.observation_features
    observation_features_hw = {
        k: v
        for k, v in all_obs.items()
        if isinstance(v, tuple) or (v is float and k.endswith(".pos"))
    }
    action_features_hw = {k: v for k, v in robot.action_features.items() if k.endswith(".pos")}

    use_videos = cfg.dataset.video if cfg.dataset else True
    action_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=teleop_action_proc,
        initial_features=create_initial_features(action=action_features_hw),
        use_videos=use_videos,
    )
    observation_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=robot_obs_proc,
        initial_features=create_initial_features(observation=observation_features_hw),
        use_videos=use_videos,
    )
    dataset_features = combine_feature_dicts(action_dataset_features, observation_dataset_features)
    hw_features = hw_to_dataset_features(observation_features_hw, "observation")
    ordered_action_keys = list(action_features_hw.keys())

    # --- Dataset ------------------------------------------------------
    dataset = None
    if cfg.dataset is not None:
        # DAgger always wants the intervention column.
        dataset_features["intervention"] = {"dtype": "bool", "shape": (1,), "names": None}
        if cfg.resume:
            logger.info("Resuming dataset %s", cfg.dataset.repo_id)
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(getattr(robot, "cameras", []) or []),
            )
        else:
            logger.info("Creating dataset %s", cfg.dataset.repo_id)
            cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera
                * len(getattr(robot, "cameras", []) or []),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                video_files_size_in_mb=getattr(cfg.strategy, "target_video_file_size_mb", None),
            )
        logger.info("Dataset ready: %s (%d episodes)", dataset.repo_id, dataset.num_episodes)

    # --- Inference (QUIC) --------------------------------------------
    inference = QuicInferenceEngine(
        quic_client=quic_client,
        hw_features=hw_features,
        ordered_action_keys=ordered_action_keys,
        shutdown_event=shutdown_event,
    )

    # --- Policy context shim -----------------------------------------
    # DAggerStrategy and base RolloutStrategy never dereference
    # ctx.policy.{policy,preprocessor,postprocessor}; only ctx.policy.inference
    # is read.  We pass simple namespaces so attribute access never blows up.
    policy_shim = SimpleNamespace(config=SimpleNamespace(action_feature_names=ordered_action_keys))
    identity_processor = SimpleNamespace(steps=(), reset=lambda: None)

    return RolloutContext(
        runtime=RuntimeContext(cfg=cfg, shutdown_event=shutdown_event),
        hardware=HardwareContext(
            robot_wrapper=robot_wrapper,
            teleop=teleop,
            initial_position=initial_position,
        ),
        policy=PolicyContext(
            policy=policy_shim,
            preprocessor=identity_processor,
            postprocessor=identity_processor,
            inference=inference,
        ),
        processors=ProcessorContext(
            teleop_action_processor=teleop_action_proc,
            robot_action_processor=robot_action_proc,
            robot_observation_processor=robot_obs_proc,
        ),
        data=DatasetContext(
            dataset=dataset,
            dataset_features=dataset_features,
            hw_features=hw_features,
            ordered_action_keys=ordered_action_keys,
        ),
    )
