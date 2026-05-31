"""Compatibility helpers for LeRobot hardware feature dictionaries.

LeRobot 0.5.2 removed ``build_dataset_frame`` and ``hw_to_dataset_features`` from
``lerobot.datasets.feature_utils``. These helpers preserve the small subset used by
our OMX recording/eval scripts.
"""

from __future__ import annotations

import numpy as np


def hw_to_dataset_features(hw_features: dict, prefix: str, use_video: bool = True) -> dict:
    scalar_names = [name for name, value in hw_features.items() if value is float]
    features = {}
    if scalar_names:
        key = "action" if prefix == "action" else f"{prefix}.state"
        features[key] = {
            "dtype": "float32",
            "shape": [len(scalar_names)],
            "names": scalar_names,
        }

    if prefix == "observation":
        image_dtype = "video" if use_video else "image"
        for name, shape in hw_features.items():
            if isinstance(shape, (tuple, list)) and len(shape) == 3:
                features[f"observation.images.{name}"] = {
                    "dtype": image_dtype,
                    "shape": list(shape),
                    "names": ["height", "width", "channels"],
                }
    return features


def build_dataset_frame(features: dict, values: dict, prefix: str) -> dict:
    frame = {}
    if prefix == "action":
        action_feature = features["action"]
        frame["action"] = np.asarray(
            [values[name] for name in action_feature["names"]],
            dtype=np.float32,
        )
        return frame

    state_key = f"{prefix}.state"
    if state_key in features:
        frame[state_key] = np.asarray(
            [values[name] for name in features[state_key]["names"]],
            dtype=np.float32,
        )

    image_prefix = f"{prefix}.images."
    for feature_key in features:
        if not feature_key.startswith(image_prefix):
            continue
        camera_name = feature_key.removeprefix(image_prefix)
        if camera_name in values:
            frame[feature_key] = values[camera_name]
    return frame
