"""Live Open3D point-cloud view of detection positions, colored by depth
("heat" -- near=hot/red, far=cool/blue). Debug/bench use only, activated
by --debug alongside the matplotlib 3D plot (src/debug_plot.py) and the
local nveglglessink video branch.

This does NOT reconstruct a real per-pixel scene depth map -- that needs
actual stereo camera calibration (checkerboard captures + cv2.stereoCalibrate
/stereoRectify), which hasn't been done yet (see CLAUDE.md/README roadmap).
It plots the SAME per-object monocular X/Y/Z already computed in
src/distance.py -- same data source as src/debug_plot.py's matplotlib
scatter, same axis convention (X=lateral, Z=forward depth on the horizontal
plane, Y=vertical=up) -- just rendered via Open3D and colored by distance
instead of by camera.

Same threading rules as debug_plot.py: add_point() is called from the
GStreamer streaming thread (cheap, thread-safe buffer append only);
update() must run on the main thread (Open3D's Visualizer, like most GUI
toolkits, is not thread-safe) via GLib.timeout_add from main.py.

Needs a display attached to the Jetson (same physical requirement as
nveglglessink/debug_plot) and a working Open3D installation. Open3D's
official PyPI wheels have had inconsistent aarch64/Jetson coverage across
versions -- this has NOT been verified on this exact device. If
`uv sync`/import fails here, that's a real possibility worth checking
first, not necessarily a bug in this module.
"""
import threading
from collections import deque

from . import config
from .distance import Detection

MAX_POINTS = 200


def _depth_to_heat_color(z_m: float) -> tuple[float, float, float]:
    """config.DEPTH_VIEW_MIN_M = hot red, DEPTH_VIEW_MAX_M = cool blue,
    through yellow/green/cyan in between."""
    span = max(config.DEPTH_VIEW_MAX_M - config.DEPTH_VIEW_MIN_M, 1e-6)
    t = max(0.0, min(1.0, (z_m - config.DEPTH_VIEW_MIN_M) / span))

    if t < 0.25:
        u = t / 0.25
        return (1.0, u, 0.0)         # red -> yellow
    elif t < 0.5:
        u = (t - 0.25) / 0.25
        return (1.0 - u, 1.0, 0.0)   # yellow -> green
    elif t < 0.75:
        u = (t - 0.5) / 0.25
        return (0.0, 1.0, u)         # green -> cyan
    else:
        u = (t - 0.75) / 0.25
        return (0.0, 1.0 - u, 1.0)   # cyan -> blue


class LiveDepthView:
    def __init__(self):
        self.enabled = False
        self._lock = threading.Lock()
        self._points: deque = deque(maxlen=MAX_POINTS)

        try:
            import open3d as o3d

            self._o3d = o3d
            self._vis = o3d.visualization.Visualizer()
            self._vis.create_window(window_name="Detection depth heatmap (debug)", width=800, height=600)

            self._pcd = o3d.geometry.PointCloud()
            self._vis.add_geometry(self._pcd)

            # Coordinate frame at the camera/drone origin -- gives the
            # otherwise-floating colored points a spatial reference.
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            self._vis.add_geometry(axes)

            render_option = self._vis.get_render_option()
            render_option.point_size = 12.0
            render_option.background_color = [0.05, 0.05, 0.05]

            self.enabled = True
        except Exception as exc:
            print(
                f"[debug_depth_view] disabled -- Open3D unavailable or failed "
                f"to open a window ({exc}). Needs a display attached to the "
                f"Jetson and a working Open3D install; continuing without it."
            )

    def add_point(self, detection: Detection) -> None:
        """Cheap, thread-safe. Call from any thread (e.g. the GStreamer
        streaming thread via register_detection_listener())."""
        if not self.enabled:
            return
        with self._lock:
            self._points.append(detection)

    def update(self) -> bool:
        """Redraw from the buffered points. Call only from the main thread
        (e.g. a GLib.timeout_add callback). Returns True so GLib keeps
        repeating the timeout."""
        if not self.enabled:
            return True

        with self._lock:
            points = list(self._points)

        if points:
            xyz = [[p.x_m, p.z_m, p.y_m] for p in points]
            colors = [_depth_to_heat_color(p.z_m) for p in points]
            self._pcd.points = self._o3d.utility.Vector3dVector(xyz)
            self._pcd.colors = self._o3d.utility.Vector3dVector(colors)
            self._vis.update_geometry(self._pcd)

        if not self._vis.poll_events():
            # User closed the window -- stop trying to render further
            # rather than erroring on every subsequent tick.
            self.enabled = False
            return True

        self._vis.update_renderer()
        return True
