#!/usr/bin/env python3
"""Exports yolov8n.pt -> yolov8n.engine (TensorRT) on-device.

The export call itself is unchanged from the original:
    model.export(format='engine', device='0', half=True, workspace=4)

This wrapper only adds diagnostics around it, since TensorRT engines are
locked to the exact TensorRT version + GPU architecture they were built on
-- they must be exported directly on this Jetson (see CLAUDE.md). Run with:
    uv run python export_engine.py
"""
import shutil
import sys
import time
import traceback
from pathlib import Path

PT_PATH = Path("yolov8n.pt")
ENGINE_DEST = Path("models/yolov8n.engine")


def _section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def check_environment() -> None:
    _section("Environment")
    import torch

    print(f"torch: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print(
            "FATAL: CUDA is not available to torch. Export needs device='0' "
            "(GPU). See CLAUDE.md 'Hardware video encode'/torch setup notes "
            "-- common causes: wrong (non-Jetson) torch wheel, or a missing "
            "system library (e.g. libcudss.so.0, libcusparseLt.so.0)."
        )
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    total_mem_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    free_mem_mb, _ = torch.cuda.mem_get_info(0)
    free_mem_mb /= 1024**2
    print(f"device: {device_name}")
    print(f"GPU memory: {free_mem_mb:.0f} MiB free / {total_mem_mb:.0f} MiB total")

    import tensorrt as trt

    print(f"tensorrt: {trt.__version__}")

    import onnx

    print(f"onnx: {onnx.__version__}")

    import ultralytics

    print(f"ultralytics: {ultralytics.__version__}")


def check_inputs() -> None:
    _section("Input checks")
    if not PT_PATH.exists():
        print(f"FATAL: {PT_PATH} not found in {Path.cwd()}")
        sys.exit(1)
    print(f"{PT_PATH}: {PT_PATH.stat().st_size / (1024**2):.1f} MiB")

    ENGINE_DEST.parent.mkdir(parents=True, exist_ok=True)
    if ENGINE_DEST.exists():
        print(f"NOTE: {ENGINE_DEST} already exists and will be overwritten on success.")


def run_export() -> Path:
    _section("Export (format='engine', device='0', half=True, workspace=4)")
    from ultralytics import YOLO

    model = YOLO(str(PT_PATH))
    t0 = time.time()
    try:
        result_path = model.export(format="engine", device="0", half=True, workspace=4)
    except Exception:
        _section("Export FAILED")
        traceback.print_exc()
        print(
            "\nCommon causes on Jetson: cuBLAS/cuDSS/cuSPARSELt system "
            "libraries missing or shadowed by a pip nvidia-*-cu12 package "
            "(see pyproject.toml notes), or insufficient free GPU memory "
            "for the given `workspace` size."
        )
        sys.exit(1)
    elapsed = time.time() - t0
    print(f"\nExport finished in {elapsed:.1f}s -> {result_path}")
    return Path(result_path)


def install_engine(exported_path: Path) -> None:
    _section("Install")
    if not exported_path.exists():
        print(f"FATAL: expected exported engine at {exported_path}, not found.")
        sys.exit(1)
    shutil.copy2(exported_path, ENGINE_DEST)
    print(f"{exported_path} -> {ENGINE_DEST} ({ENGINE_DEST.stat().st_size / (1024**2):.1f} MiB)")


def main() -> int:
    check_environment()
    check_inputs()
    exported_path = run_export()
    install_engine(exported_path)
    _section("Done")
    print(f"Ready: {ENGINE_DEST}. Next: uv run main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
