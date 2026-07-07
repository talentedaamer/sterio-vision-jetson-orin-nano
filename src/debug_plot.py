"""Live 3D scatter plot of detection X/Y/Z coordinates -- debug/bench use
only, activated by --debug alongside the local nveglglessink video branch.

Two threading notes that matter for correctness:
  - GStreamer pad probes (which produce Detections, see src/probes.py) run
    on a streaming thread, not the main thread. add_point() only appends to
    a lock-protected buffer -- it must stay cheap and must never touch
    matplotlib objects directly.
  - matplotlib GUI calls are not thread-safe. All drawing happens in
    update(), which main.py drives via GLib.timeout_add() on the main
    thread (the same thread running the GLib mainloop), never from the
    streaming thread.

Same physical requirement as the nveglglessink debug branch: this needs a
display attached to the Jetson (HDMI/DP) with a desktop session, plus a
working interactive matplotlib backend (Tk/Qt/GTK) installed at the OS
level. If neither is available, this degrades to a harmless no-op with a
printed warning instead of crashing the detection pipeline.

On a `--system-site-packages` venv, the OS may also have an old
`python3-matplotlib` apt package installed. Its `mpl_toolkits` uses the
legacy `pkg_resources.declare_namespace(...)` style (a real `__init__.py`),
which always wins as a "regular package" over the venv's own newer,
implicit-namespace-style `mpl_toolkits` -- regardless of sys.path order.
The result is `mpl_toolkits.mplot3d` resolving to the ancient system
version, which then fails importing internals (e.g. `matplotlib.docstring`)
that don't exist in the venv's much newer matplotlib core. _import_plt()
below works around this by hiding the system site-packages path from
sys.path for the duration of the matplotlib import only, so the venv's own
compatible copy is the only one Python can find -- gi/pyds (which DO need
that system path) are untouched, since they're imported elsewhere.
"""
import sys
import threading
from collections import deque

from .distance import Detection

MAX_POINTS = 200
CAMERA_COLORS = {0: "tab:blue", 1: "tab:red"}
_SYSTEM_DIST_PACKAGES = "/usr/lib/python3/dist-packages"


def _import_plt():
    """Import matplotlib.pyplot (with a working Axes3D) while hiding the
    system dist-packages path, so mpl_toolkits resolves to the venv's own
    version-matched copy instead of an incompatible system one. See module
    docstring."""
    hidden = _SYSTEM_DIST_PACKAGES in sys.path
    if hidden:
        sys.path.remove(_SYSTEM_DIST_PACKAGES)
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- force-resolve now, while hidden
    finally:
        if hidden and _SYSTEM_DIST_PACKAGES not in sys.path:
            sys.path.append(_SYSTEM_DIST_PACKAGES)
    return plt


class LiveXYZPlot:
    def __init__(self):
        self.enabled = False
        self._lock = threading.Lock()
        self._points: deque[Detection] = deque(maxlen=MAX_POINTS)

        try:
            plt = _import_plt()
            plt.ion()
            self._plt = plt
            self._fig = plt.figure("Detection XYZ (debug)")
            self._ax = self._fig.add_subplot(111, projection="3d")
            self._label_axes()
            self._fig.show()
            self.enabled = True
        except Exception as exc:
            print(
                f"[debug_plot] disabled -- matplotlib GUI backend unavailable "
                f"({exc}). Needs a display attached to the Jetson and a "
                f"working Tk/Qt/GTK matplotlib backend; continuing without it."
            )

    def _label_axes(self) -> None:
        # Plotted so the view reads like a top-down/spatial map: lateral
        # offset and forward depth as the horizontal plane, height as up --
        # not a literal X/Y/Z-axis-letter match to the Detection fields.
        self._ax.set_xlabel("X: lateral (m)")
        self._ax.set_ylabel("Z: forward depth (m)")
        self._ax.set_zlabel("Y: vertical (m)")

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
        if not points:
            return True

        self._ax.cla()
        self._label_axes()
        for source_id, color in CAMERA_COLORS.items():
            cam_points = [p for p in points if p.source_id == source_id]
            if not cam_points:
                continue
            self._ax.scatter(
                [p.x_m for p in cam_points],
                [p.z_m for p in cam_points],
                [p.y_m for p in cam_points],
                c=color,
                label=f"cam{source_id}",
                s=20,
            )

        latest = points[-1]
        self._ax.set_title(
            f"{latest.label} conf={latest.confidence:.2f} "
            f"X={latest.x_m:.1f} Y={latest.y_m:.1f} Z={latest.z_m:.1f}m"
        )
        self._ax.legend(loc="upper left")

        try:
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
        except Exception:
            pass
        return True
