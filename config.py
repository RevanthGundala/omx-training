"""Shared configuration constants for OMX training scripts."""

# ── Hardware Ports ──
LEADER_PORT = "/dev/tty.usbmodem11301"
FOLLOWER_PORT = "/dev/tty.usbmodem11401"

# ── Joint Configuration ──
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# ── Camera Defaults ──
# Map of camera name → OpenCV device index. Names appear in Rerun under
# camera/<name>, in datasets as observation.images.<name>, and the trained
# policy is fed frames matched by these names.
#
# IMPORTANT: this mapping must agree with the camera name used at training
# time. The 050000 checkpoint expects `observation.images.front`, even though
# physically that feed comes from the wrist-mounted camera. So `front` here
# must point at the physical wrist camera's OpenCV index.
#
# On macOS, index 0 is typically the built-in FaceTime camera; external USB
# cameras are usually 1, 2, ... Run:
#   uv run python -c "from lerobot.cameras.opencv.camera_opencv import OpenCVCamera; print(OpenCVCamera.find_cameras())"
# to enumerate.
CAMERAS = {
    "front": 1,  # physical wrist camera; matches `observation.images.front` in dataset/checkpoint
}
CAMERA_INDEX = CAMERAS["front"]  # back-compat for scripts that expect a single index
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ── Control Loop FPS ──
FPS = 30
TELEOP_FPS = 60

# ── Dataset & Models ──
RECORD_DATASET_REPO_ID = "RevanthGundala/001-fold-tissue"
TRAIN_DATASET_REPO_ID = "RevanthGundala/001-fold-tissue"
PI0_MODEL_REPO_ID = "lerobot/pi0"
PI05_MODEL_REPO_ID = "lerobot/pi05_base"

# ── Task ──
TASK_NAME = "Do a single fold of tissue while laying completely flat."
