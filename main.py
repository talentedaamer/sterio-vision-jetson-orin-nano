#!/usr/bin/env python3
"""Entry point for the dual-camera YOLOv8n DeepStream detection pipeline.

Usage:
    uv run main.py            # headless + RTSP out (default)
    uv run main.py --debug    # headless + RTSP out + local bench display
"""
import argparse
import os
import signal
import sys

# Force the classic (property-configured) nvstreammux rather than DS 7.x's
# newer config-file-based one -- simpler to drive from Python properties for
# a fixed 2-source live-camera setup. Must be set before Gst.init().
os.environ.setdefault("USE_NEW_NVSTREAMMUX", "no")

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from src import config
from src.pipeline import build_pipeline, start_rtsp_server


def bus_call(_bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("[bus] end-of-stream")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"[bus] ERROR: {err}: {debug}", file=sys.stderr)
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        warn, debug = message.parse_warning()
        print(f"[bus] WARNING: {warn}: {debug}", file=sys.stderr)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug", action="store_true", default=config.DEBUG,
        help="Also render to a local display (nveglglessink) for bench testing.",
    )
    args = parser.parse_args()

    Gst.init(None)

    pipeline = build_pipeline(debug=args.debug)
    rtsp_server = start_rtsp_server()  # noqa: F841 -- keep alive for the process lifetime

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    if args.debug:
        # Lazy import -- matplotlib is only touched in --debug mode, zero
        # cost/impact on the normal headless + RTSP production path.
        from src.debug_plot import LiveXYZPlot
        from src.probes import register_detection_listener

        xyz_plot = LiveXYZPlot()
        register_detection_listener(xyz_plot.add_point)
        GLib.timeout_add(200, xyz_plot.update)  # redraw at ~5Hz, main thread only

    mavlink = None
    if config.MISSION_MODE != "NONE":
        # Lazy import -- pymavlink/serial connection only touched when a
        # mission mode is actually configured, zero cost/impact otherwise.
        from src.mavlink_link import MavlinkLink
        from src.mission import Mission
        from src.probes import register_detection_listener

        mavlink = MavlinkLink()
        mavlink.connect()
        mission = Mission(mavlink)
        register_detection_listener(mission.on_detection)
        GLib.timeout_add(int(config.FOLLOW_UPDATE_INTERVAL_S * 1000), mission.update)
        print(f"[main] mission mode: {config.MISSION_MODE} (dry_run={config.FOLLOW_DRY_RUN})")

    def shutdown(*_args):
        print("\n[main] shutting down")
        if mavlink is not None:
            mavlink.close()
        pipeline.send_event(Gst.Event.new_eos())
        loop.quit()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"[main] mode: {'DEBUG (bench display on)' if args.debug else 'headless + RTSP'}")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)

    return 0


if __name__ == "__main__":
    sys.exit(main())
