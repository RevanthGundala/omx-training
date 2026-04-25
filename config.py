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
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ── Control Loop FPS ──
FPS = 30
TELEOP_FPS = 60

# ── Dataset & Models ──
RECORD_DATASET_REPO_ID = "RevanthGundala/pick_up_remote"
TRAIN_DATASET_REPO_ID = "RevanthGundala/pick_up_packet_test"
PI0_MODEL_REPO_ID = "lerobot/pi0"

# ── Task ──
TASK_NAME = "Pick up remote and place it onto the gray circle"
