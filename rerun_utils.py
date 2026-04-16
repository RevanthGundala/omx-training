"""Rerun visualization helpers for OMX scripts."""

import rerun as rr
import rerun.blueprint as rrb

from config import JOINT_NAMES


def build_joint_blueprint(
    joint_names: list[str] | None = None,
    has_camera: bool = True,
) -> rrb.Horizontal | rrb.Vertical:
    """Build the standard Rerun blueprint with camera + joint time series."""
    joint_names = joint_names or JOINT_NAMES
    joint_views = [
        rrb.TimeSeriesView(name=name, contents=[f"joints/{name}/**"])
        for name in joint_names
    ]
    if has_camera:
        return rrb.Horizontal(
            rrb.Spatial2DView(name="Camera", contents=["camera/**"]),
            rrb.Vertical(*joint_views),
            column_shares=[1, 2],
        )
    return rrb.Vertical(*joint_views)


def init_rerun(
    name: str,
    joint_names: list[str] | None = None,
    has_camera: bool = True,
) -> None:
    """Initialize the Rerun viewer with a standard blueprint."""
    blueprint = build_joint_blueprint(joint_names, has_camera)
    rr.init(name, spawn=True)
    rr.send_blueprint(blueprint)
