"""Rerun visualization helpers for OMX scripts."""

from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb

from config import JOINT_NAMES, CAMERAS

# Directory where .rrd log files are saved
RRD_DIR = Path("outputs/rerun")


def build_joint_blueprint(
    joint_names: list[str] | None = None,
    has_camera: bool = True,
    camera_primary: bool = False,
    camera_names: list[str] | None = None,
) -> rrb.Horizontal | rrb.Vertical | rrb.Tabs:
    """Build the standard Rerun blueprint with camera + joint time series.

    If camera_primary=True, camera fills the main view with joints in a second tab.
    Each entry in `camera_names` gets its own Spatial2DView so multiple cameras
    are shown side-by-side instead of overlapping in a single view.
    """
    joint_names = joint_names or JOINT_NAMES
    camera_names = camera_names or (list(CAMERAS.keys()) if has_camera else [])

    joint_views = [
        rrb.TimeSeriesView(name=name, contents=[f"joints/{name}/**"])
        for name in joint_names
    ]
    cam_views = [
        rrb.Spatial2DView(name=cam, contents=[f"camera/{cam}/**"])
        for cam in camera_names
    ]
    cam_panel = rrb.Vertical(*cam_views) if cam_views else None

    if has_camera and camera_primary and cam_panel is not None:
        if len(cam_views) > 1:
            cam_tab = rrb.Horizontal(*cam_views, name="Cameras")
        else:
            cam_tab = cam_views[0]
        joint_tab = rrb.Vertical(*joint_views, name="Joints")
        return rrb.Tabs(cam_tab, joint_tab)
    if has_camera and cam_panel is not None:
        return rrb.Horizontal(
            cam_panel,
            rrb.Vertical(*joint_views),
            column_shares=[1, 2],
        )
    return rrb.Vertical(*joint_views)


def init_rerun(
    name: str,
    joint_names: list[str] | None = None,
    has_camera: bool = True,
    camera_primary: bool = False,
    save_rrd: bool = True,
    camera_names: list[str] | None = None,
) -> Path | None:
    """Initialize the Rerun viewer with a standard blueprint.

    If camera_primary=True, camera is the main view with joints in a second tab.
    Also saves an .rrd file to outputs/rerun/<name>.rrd for offline analysis.
    Returns the path to the saved .rrd file, or None if saving is disabled.
    """
    blueprint = build_joint_blueprint(joint_names, has_camera, camera_primary, camera_names)
    rr.init(name)

    rrd_path = None
    if save_rrd:
        RRD_DIR.mkdir(parents=True, exist_ok=True)
        rrd_path = RRD_DIR / f"{name}.rrd"
        # Fan out to both the spawned viewer and an .rrd file. In rerun >=0.22
        # `rr.save` replaces the active sink, so we must use set_sinks to keep
        # the live viewer connected while also writing to disk.
        rr.set_sinks(
            rr.GrpcSink(),
            rr.FileSink(str(rrd_path)),
        )
        rr.spawn()
    else:
        rr.spawn()

    rr.send_blueprint(blueprint)
    return rrd_path
