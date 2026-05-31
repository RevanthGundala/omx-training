"""
record.py — Record teleoperation episodes to a LeRobot dataset.

Uses the leader arm for teleoperation and the follower arm as the robot.
Records joint positions + camera images to a local dataset.

Controls:
  →  stop recording, enter review
  ←  discard episode immediately

During review (after → stops recording):
  ↑  replay the episode on the follower
  →  save episode
  ←  discard episode

Ctrl+C during reset → stop recording entirely
"""

import time
import threading
import shutil
from pathlib import Path

import cv2
import multiprocessing as mp
import numpy as np
import rerun as rr
from huggingface_hub import HfApi
from pynput import keyboard

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from data.scene_assist import (
    DEFAULT_MODEL as SCENE_ASSIST_DEFAULT_MODEL,
    analyze_start_scene,
    format_coverage_summary,
    format_preflight_report,
    save_scene_artifacts,
)
from utils.config import CAMERAS, FPS, JOINT_NAMES, RECORD_DATASET_REPO_ID as DATASET_REPO_ID, TASK_NAME
from utils.control_utils import maintain_fps
from utils.lerobot_compat import build_dataset_frame, hw_to_dataset_features
from utils.rerun_utils import init_rerun
from utils.robot_utils import create_follower, create_leader, safe_disconnect

USE_RERUN = False  # Set True to enable Rerun visualizer (adds latency)
SHOW_CAMERAS = True  # Live camera preview via OpenCV (minimal latency)
SAVE_START_SCENE_ARTIFACTS = True
SCENE_ASSIST_ENABLED = True
SCENE_ASSIST_TOP_CAMERA = "top"
SCENE_ASSIST_MODEL = SCENE_ASSIST_DEFAULT_MODEL
SCENE_ASSIST_MIN_CONFIDENCE = 0.25
SCENE_ASSIST_TARGET_COUNT = 4
SCENE_ASSIST_MIN_EPISODE_INDEX = 50
SCENE_DIVERSITY_DIR = Path("outputs/record_scene_diversity")

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RED = "\033[91m"


def color(text: str, ansi: str) -> str:
    return f"{ansi}{text}{RESET}"


def _camera_display_worker(frame_queue: mp.Queue, stop_event):
    """Subprocess that displays camera frames. Runs in its own process to avoid
    macOS segfaults from OpenCV GUI + pynput threading conflicts."""
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except Exception:
            continue
        if frame is None:
            break
        cv2.imshow("Camera Preview", frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()


class CameraPreview:
    """Manages a separate process for camera display."""
    def __init__(self):
        self._proc = None
        self._queue = None
        self._stop = None

    def start(self):
        if not SHOW_CAMERAS:
            return
        self._stop = mp.Event()
        self._queue = mp.Queue(maxsize=2)
        self._proc = mp.Process(target=_camera_display_worker,
                                args=(self._queue, self._stop), daemon=True)
        self._proc.start()

    def show(self, observation):
        if not SHOW_CAMERAS or self._proc is None or not self._proc.is_alive():
            return
        panels = []
        for cam_name in CAMERAS:
            img = observation.get(cam_name)
            if img is not None and isinstance(img, np.ndarray):
                if img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.putText(img, cam_name, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                panels.append(img)
        if panels:
            combined = np.vstack(panels) if len(panels) > 1 else panels[0]
            # Drop old frame if display is behind
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Exception:
                    pass
            try:
                self._queue.put_nowait(combined)
            except Exception:
                pass

    def stop(self):
        if self._proc is None:
            return
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        self._proc.join(timeout=2)
        if self._proc.is_alive():
            self._proc.terminate()
        self._proc = None


# Global camera preview instance
camera_preview = CameraPreview()


# ──────────────────────────────────────────────
# Object-centric start-scene assistant
# ──────────────────────────────────────────────

def _dataset_artifact_dir() -> Path:
    return SCENE_DIVERSITY_DIR / DATASET_REPO_ID.replace("/", "__")


def bootstrap_scene_diversity_artifacts(dataset_root: Path) -> None:
    if not SCENE_ASSIST_ENABLED or not USE_CAMERA:
        return

    artifact_root = _dataset_artifact_dir()
    print(color(f"  Scene assistant artifacts: {artifact_root}", CYAN))
    print(color(f"  Scene assistant coverage: episodes {SCENE_ASSIST_MIN_EPISODE_INDEX}+ only (ignoring earlier episodes).", DIM))

    try:
        import pandas as pd

        episode_files = sorted((dataset_root / "meta" / "episodes").glob("**/*.parquet"))
        if not episode_files:
            return
        episodes = (
            pd.concat([pd.read_parquet(path) for path in episode_files], ignore_index=True)
            .sort_values("episode_index")
        )
    except Exception as e:
        print(color(f"  ⚠️  Scene assistant bootstrap skipped: {e}", YELLOW))
        return

    created = 0
    for _, row in episodes.iterrows():
        episode_index = int(row["episode_index"])
        if episode_index < SCENE_ASSIST_MIN_EPISODE_INDEX:
            continue
        if (artifact_root / f"episode-{episode_index:04d}.scene.json").exists():
            continue

        observation = {}
        for cam_name in CAMERAS:
            prefix = f"videos/observation.images.{cam_name}"
            required = [f"{prefix}/chunk_index", f"{prefix}/file_index", f"{prefix}/from_timestamp"]
            if not all(key in row for key in required):
                continue
            video_path = (
                dataset_root
                / "videos"
                / f"observation.images.{cam_name}"
                / f"chunk-{int(row[f'{prefix}/chunk_index']):03d}"
                / f"file-{int(row[f'{prefix}/file_index']):03d}.mp4"
            )
            if not video_path.exists():
                continue
            cap = cv2.VideoCapture(str(video_path))
            try:
                frame_index = max(0, int(float(row[f"{prefix}/from_timestamp"]) * FPS))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, bgr = cap.read()
            finally:
                cap.release()
            if ok:
                observation[cam_name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        top_frame = observation.get(SCENE_ASSIST_TOP_CAMERA)
        if not isinstance(top_frame, np.ndarray):
            continue
        try:
            report = analyze_start_scene(
                top_frame,
                episode_index=episode_index,
                scene_root=artifact_root,
                model_path=SCENE_ASSIST_MODEL,
                min_confidence=SCENE_ASSIST_MIN_CONFIDENCE,
                target_count=SCENE_ASSIST_TARGET_COUNT,
                min_episode_index=SCENE_ASSIST_MIN_EPISODE_INDEX,
            )
            report["bootstrapped_from_dataset"] = True
            save_scene_artifacts(artifact_root, episode_index, observation, report, tuple(CAMERAS.keys()))
            created += 1
        except Exception as e:
            print(color(f"  ⚠️  Scene assistant skipped episode {episode_index}: {e}", YELLOW))

    if created:
        print(color(f"  Bootstrapped {created} start-scene checks from episodes {SCENE_ASSIST_MIN_EPISODE_INDEX}+.", CYAN))


def start_scene_preflight(robot, episode_index: int) -> dict | None:
    if not SAVE_START_SCENE_ARTIFACTS or not USE_CAMERA or not SCENE_ASSIST_ENABLED:
        return None

    root = _dataset_artifact_dir()
    try:
        observation = robot.get_observation()
    except ConnectionError as e:
        print(color(f"\n  ⚠️  USB glitch during start-scene capture: {e}", YELLOW))
        return None
    camera_preview.show(observation)

    top_frame = observation.get(SCENE_ASSIST_TOP_CAMERA)
    if not isinstance(top_frame, np.ndarray):
        print(color(f"\n  ⚠️  Scene assistant skipped: missing {SCENE_ASSIST_TOP_CAMERA!r} camera frame.", YELLOW))
        return {"observation": observation, "report": {"episode_index": episode_index, "error": "missing_top_frame"}}

    try:
        report = analyze_start_scene(
            top_frame,
            episode_index=episode_index,
            scene_root=root,
            model_path=SCENE_ASSIST_MODEL,
            min_confidence=SCENE_ASSIST_MIN_CONFIDENCE,
            target_count=SCENE_ASSIST_TARGET_COUNT,
            min_episode_index=SCENE_ASSIST_MIN_EPISODE_INDEX,
        )
    except Exception as e:
        report = {"episode_index": episode_index, "error": str(e)}

    report["operator_decision"] = "auto_record"
    print(format_preflight_report(report))
    return {"observation": observation, "report": report}

# ──────────────────────────────────────────────
# Replay from buffer helper
# ──────────────────────────────────────────────
MOVE_TO_START_DURATION_S = 4.0


def replay_from_buffer(action_buffer, state_buffer, action_names, state_names, follower, fps):
    """Replay buffered frames on the follower arm."""
    num_frames = len(action_buffer)
    if num_frames == 0:
        print("  No frames to replay.")
        return

    print(f"\n  ▶ Replaying {num_frames} frames...")

    # Move to start position
    start_state = dict(zip(state_names, state_buffer[0]))
    current = follower.get_observation()
    steps = max(1, int(MOVE_TO_START_DURATION_S * fps))
    print(f"  Moving to start pose over {MOVE_TO_START_DURATION_S}s...")
    for step in range(1, steps + 1):
        loop_start = time.perf_counter()
        alpha = step / steps
        target = {k: current[k] + alpha * (start_state[k] - current[k]) for k in start_state}
        follower.send_action(target)
        maintain_fps(loop_start, fps)

    # Play back actions
    print(f"  Playing... (Ctrl+C to skip)")
    try:
        for idx, action_vals in enumerate(action_buffer):
            loop_start = time.perf_counter()
            action = dict(zip(action_names, action_vals))
            follower.send_action(action)

            # Stream live camera to Rerun during replay
            observation = follower.get_observation()
            if USE_RERUN:
                rr.set_time("replay_step", sequence=idx)
                for cam_name in CAMERAS:
                    if cam_name in observation:
                        rr.log(f"camera/{cam_name}", rr.Image(observation[cam_name]))
                for name in JOINT_NAMES:
                    key = f"{name}.pos"
                    if key in observation:
                        rr.log(f"joints/{name}/replay", rr.Scalars(observation[key]))

            print(f"  Replay frame {idx+1:4d}/{num_frames}", end="\r")
            maintain_fps(loop_start, fps)
    except KeyboardInterrupt:
        print("\n  Replay interrupted.")
    print(f"\n  ▶ Replay complete.")

# ──────────────────────────────────────────────
# Recording-specific configuration
# ──────────────────────────────────────────────
USE_CAMERA = True
NUM_EPISODES = 50
EPISODE_DURATION_S = 90
RESET_DURATION_S = 10
USE_VIDEO = True
PUSH_TO_HUB = False
SOFT_START_DURATION_S = 3.0  # gradually ramp follower to leader position on connect


def soft_start(follower, leader, duration_s=SOFT_START_DURATION_S):
    """Gradually move follower to match leader position to prevent jerk/overload."""
    print(f"  Soft-starting: ramping follower to leader over {duration_s}s...")
    current = follower.get_observation()
    target = leader.get_action()
    steps = max(1, int(duration_s * FPS))

    for step in range(1, steps + 1):
        loop_start = time.perf_counter()
        alpha = step / steps
        blended = {
            key: current[key] + alpha * (target[key] - current[key])
            for key in target
        }
        follower.send_action(blended)
        # Re-read leader in case it moved during ramp
        target = leader.get_action()
        maintain_fps(loop_start, FPS)

    print("  Soft-start complete.")


def record_one_episode(robot, leader, dataset, episode_num, rerun_step=0):
    """Record a single episode. Returns (frame_count, action_buffer, state_buffer, rerun_step).
    frame_count = -1 means discard was pressed during recording."""
    print(color(f"\n{'='*60}", BLUE))
    print(color(f"  RECORDING Episode {episode_num}", BOLD + BLUE))
    print(color(f"  Task: {TASK_NAME}", BLUE))
    print(color(f"  Max duration: {EPISODE_DURATION_S}s — → stop & review, ← discard", BLUE))
    print(color(f"{'='*60}\n", BLUE))

    end_episode = threading.Event()
    discard_episode = threading.Event()

    def on_press(key):
        if key == keyboard.Key.right:
            end_episode.set()
        elif key == keyboard.Key.left:
            discard_episode.set()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    frame_count = 0
    action_buffer = []
    state_buffer = []
    start_time = time.perf_counter()

    try:
        while not end_episode.is_set() and not discard_episode.is_set():
            loop_start = time.perf_counter()
            elapsed = loop_start - start_time

            if elapsed >= EPISODE_DURATION_S:
                print(f"\n  Episode time limit reached ({EPISODE_DURATION_S}s).")
                break

            try:
                observation = robot.get_observation()
                action = leader.get_action()
                sent_action = robot.send_action(action)
            except ConnectionError as e:
                print(color(f"\n  ⚠️  USB glitch (retrying): {e}", YELLOW))
                time.sleep(0.05)
                continue

            camera_preview.show(observation)

            # Buffer for potential replay
            action_buffer.append([sent_action[k] for k in sorted(sent_action)])
            state_buffer.append([observation[k] for k in sorted(observation)])

            obs_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
            action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
            frame = {**obs_frame, **action_frame}
            frame["task"] = TASK_NAME
            dataset.add_frame(frame)
            frame_count += 1

            # ── Rerun: stream camera + joints live ──
            if USE_RERUN:
                rr.set_time("step", sequence=rerun_step)
                for cam_name in CAMERAS:
                    if cam_name in observation:
                        rr.log(f"camera/{cam_name}", rr.Image(observation[cam_name]))
                for i, name in enumerate(JOINT_NAMES):
                    obs_key = f"{name}.pos"
                    act_key = f"{name}.pos"
                    if obs_key in observation:
                        rr.log(f"joints/{name}/state", rr.Scalars(observation[obs_key]))
                    if act_key in sent_action:
                        rr.log(f"joints/{name}/action", rr.Scalars(sent_action[act_key]))
                rerun_step += 1

            print(f"  Frame {frame_count:4d} | Time: {elapsed:6.1f}s", end="\r")
            maintain_fps(loop_start, FPS)

    except KeyboardInterrupt:
        print(color(f"\n  Episode ended early by user.", YELLOW))
    finally:
        listener.stop()

    print(f"  Recorded {frame_count} frames ({frame_count/FPS:.1f}s)")

    action_names = sorted(sent_action.keys()) if frame_count > 0 else []
    state_names = sorted(observation.keys()) if frame_count > 0 else []

    if discard_episode.is_set():
        return -1, [], [], [], [], rerun_step
    return frame_count, action_buffer, state_buffer, action_names, state_names, rerun_step


def main():
    leader = create_leader()
    follower = create_follower(camera=USE_CAMERA)

    # Build dataset feature definitions from robot hardware
    action_features = hw_to_dataset_features(follower.action_features, "action", USE_VIDEO)
    obs_features = hw_to_dataset_features(follower.observation_features, "observation", USE_VIDEO)
    dataset_features = {**action_features, **obs_features}

    print("Dataset features:")
    for name, feat in dataset_features.items():
        print(f"  {name}: shape={feat['shape']}, dtype={feat['dtype']}")

    # Create or resume the dataset
    dataset_path = Path.home() / ".cache/huggingface/lerobot" / DATASET_REPO_ID
    info_path = dataset_path / "meta" / "info.json"

    if info_path.exists():
        import json
        with open(info_path) as f:
            info = json.load(f)
        if info.get("total_episodes", 0) > 0:
            # Has committed episodes — resume using the proper classmethod
            print(f"\nResuming existing dataset at {dataset_path}")
            dataset = LeRobotDataset.resume(
                repo_id=DATASET_REPO_ID,
                root=str(dataset_path),
                image_writer_processes=0,
                image_writer_threads=4,
            )
            print(f"  Existing episodes: {dataset.num_episodes}")
        else:
            # Created but 0 episodes — safe to recreate
            print(f"\n⚠️  Found empty dataset (0 episodes). Re-creating.")
            shutil.rmtree(dataset_path)
            dataset = LeRobotDataset.create(
                repo_id=DATASET_REPO_ID,
                fps=FPS,
                robot_type="omx_follower",
                features=dataset_features,
                use_videos=USE_VIDEO,
                image_writer_processes=0,
                image_writer_threads=4,
            )
    else:
        # No dataset at all — create fresh
        if dataset_path.exists():
            backup_path = dataset_path.with_name(dataset_path.name + f"_backup_{int(time.time())}")
            print(f"\n⚠️  Existing directory has no info.json. Backing up to: {backup_path}")
            shutil.move(str(dataset_path), str(backup_path))
        print(f"\nCreating new dataset: {DATASET_REPO_ID}")
        dataset = LeRobotDataset.create(
            repo_id=DATASET_REPO_ID,
            fps=FPS,
            robot_type="omx_follower",
            features=dataset_features,
            use_videos=USE_VIDEO,
            image_writer_processes=0,
            image_writer_threads=4,
        )

    bootstrap_scene_diversity_artifacts(dataset_path)

    # Connect hardware
    print("\nConnecting leader arm...")
    leader.connect()
    print("Connecting follower arm...")
    follower.connect()

    # Gradually ramp follower to leader position to prevent jerk/overload
    soft_start(follower, leader)

    # ── Rerun setup (camera POV + joint plots) ──
    if USE_RERUN:
        init_rerun("omx_record", has_camera=USE_CAMERA, camera_primary=True, save_rrd=False)

    # Start camera preview process
    camera_preview.start()

    # Record episodes in a loop
    episode = 0
    rerun_step = 0
    try:
        while episode < NUM_EPISODES:
            scene_report = start_scene_preflight(follower, dataset.num_episodes)
            if scene_report is not None and scene_report.get("report", {}).get("operator_decision") == "quit":
                print("\n  Scene assistant quit requested.")
                break
            try:
                result = record_one_episode(follower, leader, dataset, dataset.num_episodes, rerun_step)
            except Exception as e:
                print(color(f"\n  ⚠️  Episode error: {e}", RED))
                import traceback; traceback.print_exc()
                dataset.clear_episode_buffer()
                continue
            frame_count, action_buf, state_buf, action_names, state_names, rerun_step = result

            if frame_count == 0:
                print(color("  No frames recorded, skipping episode.", YELLOW))
                dataset.clear_episode_buffer()
                continue

            if frame_count < 0:
                print(color("  ← Episode DISCARDED.", RED))
                try:
                    dataset.clear_episode_buffer()
                except Exception as e:
                    print(color(f"  ⚠️  clear_episode_buffer error: {e}", YELLOW))
                continue

            # ── Review phase: replay/save/discard ──
            print(color(f"\n  REVIEW: ↑ replay, → save, ← discard", MAGENTA))
            save_ep = threading.Event()
            discard_ep = threading.Event()
            replay_ep = threading.Event()

            def on_review_press(key):
                if key == keyboard.Key.right:
                    save_ep.set()
                elif key == keyboard.Key.left:
                    discard_ep.set()
                elif key == keyboard.Key.up:
                    replay_ep.set()

            review_listener = keyboard.Listener(on_press=on_review_press)
            review_listener.start()

            try:
                while not save_ep.is_set() and not discard_ep.is_set():
                    # Keep teleop alive while waiting
                    loop_start = time.perf_counter()
                    try:
                        action = leader.get_action()
                        follower.send_action(action)
                        observation = follower.get_observation()
                    except ConnectionError as e:
                        print(color(f"\n  ⚠️  USB glitch during review (retrying): {e}", YELLOW))
                        time.sleep(0.05)
                        continue
                    camera_preview.show(observation)

                    if replay_ep.is_set():
                        replay_ep.clear()
                        replay_from_buffer(
                            action_buf, state_buf, action_names, state_names,
                            follower, FPS,
                        )
                        soft_start(follower, leader)
                        print(color(f"  REVIEW: ↑ replay again, → save, ← discard", MAGENTA))

                    maintain_fps(loop_start, FPS)
            except KeyboardInterrupt:
                review_listener.stop()
                print("\n\n  Stopping recording.")
                dataset.clear_episode_buffer()
                break
            finally:
                review_listener.stop()

            if discard_ep.is_set():
                print(color("  ← Episode DISCARDED.", RED))
                dataset.clear_episode_buffer()
                continue

            # Save the episode
            committed_episode_index = dataset.num_episodes
            dataset.save_episode(parallel_encoding=False)
            if scene_report is not None:
                save_scene_artifacts(
                    _dataset_artifact_dir(),
                    committed_episode_index,
                    scene_report["observation"],
                    scene_report["report"],
                    tuple(CAMERAS.keys()),
                )
            episode += 1
            print(color(f"  ✓ Episode saved! (Total episodes: {dataset.num_episodes})", GREEN))
            if scene_report is not None:
                print(color(f"  Scene diversity artifacts: {_dataset_artifact_dir()}", DIM))

            # Brief reset period
            print(color(f"\n  Resetting for {RESET_DURATION_S}s... (Ctrl+C to stop)", CYAN))
            try:
                start = time.perf_counter()
                while time.perf_counter() - start < RESET_DURATION_S:
                    remaining = RESET_DURATION_S - (time.perf_counter() - start)
                    print(f"  Next episode in {remaining:.0f}s...", end="\r")
                    try:
                        action = leader.get_action()
                        follower.send_action(action)
                        observation = follower.get_observation()
                    except ConnectionError as e:
                        print(color(f"\n  ⚠️  USB glitch during reset (retrying): {e}", YELLOW))
                        time.sleep(0.05)
                        continue
                    camera_preview.show(observation)
                    maintain_fps(time.perf_counter(), FPS)
            except KeyboardInterrupt:
                print("\n\n  Stopping recording.")
                break

    except KeyboardInterrupt:
        print("\n\nRecording interrupted.")

    finally:
        camera_preview.stop()
        safe_disconnect(follower)
        safe_disconnect(leader)
        # CRITICAL: finalize the dataset writer so the parquet footer is written.
        # Without this, the on-disk parquet has data but no footer and is unreadable
        # ("Parquet magic bytes not found in footer" on resume/load).
        try:
            dataset.finalize()
            print(color(f"Dataset finalized: {dataset.num_episodes} episodes committed.", GREEN))
        except Exception as e:
            print(color(f"WARNING: dataset.finalize() failed: {e}", YELLOW))

    # Push to hub (optional)
    if PUSH_TO_HUB:
        api = HfApi()
        api.create_repo(repo_id=DATASET_REPO_ID, repo_type="dataset", exist_ok=True)
        print("\nPushing dataset to HuggingFace Hub...")
        dataset.push_to_hub()
        print("Done!")

    print(f"\nDataset saved locally: {dataset.root}")
    print(f"Total episodes: {dataset.num_episodes}")


if __name__ == "__main__":
    main()
