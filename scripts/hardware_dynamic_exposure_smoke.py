from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from PySide6.QtCore import QCoreApplication, QTimer

TOOL_ROOT = Path(__file__).resolve().parents[1]
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from calibration_tool.camera import build_camera_provider, load_camera_config
from calibration_tool.gui.workers import PreviewThread
from calibration_tool.io_utils import dump_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="验证 GUI 取流线程能否在线修改真实相机曝光")
    parser.add_argument("--config", type=Path, default=Path("configs/camera.mvs.example.yaml"))
    parser.add_argument("--target-exposure-us", type=float, default=50000.0)
    parser.add_argument("--timeout-ms", type=int, default=15000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runtime = load_camera_config(args.config)
    provider = build_camera_provider(
        runtime["backend"],
        calibration_src=runtime["calibration_src"],
        backend_options=runtime["backend_options"],
    )
    app = QCoreApplication.instance() or QCoreApplication(sys.argv[:1])
    thread = PreviewThread(
        provider,
        runtime["serial_number"],
        runtime["camera"],
        "laser",
        runtime["quality_thresholds"],
        runtime["board_pattern"],
    )
    before: list[float] = []
    after: list[float] = []
    applied = []
    errors: list[str] = []
    requested = False
    timed_out = False

    def on_frame(frame, _quality) -> None:
        nonlocal requested
        mean = float(np.mean(frame.image))
        if not applied:
            before.append(mean)
            if len(before) >= 3 and not requested:
                requested = True
                thread.request_exposure_gain(args.target_exposure_us, runtime["camera"].gain_db)
        else:
            after.append(mean)
            if len(after) >= 3:
                thread.requestInterruption()

    def on_timeout() -> None:
        nonlocal timed_out
        timed_out = True
        thread.requestInterruption()

    thread.frame_ready.connect(on_frame)
    thread.settings_applied.connect(applied.append)
    thread.parameter_update_failed.connect(errors.append)
    thread.failed.connect(errors.append)
    thread.finished.connect(app.quit)
    QTimer.singleShot(args.timeout_ms, on_timeout)
    thread.start()
    app.exec()
    thread.wait(5000)

    result = {
        "passed": bool(
            not timed_out
            and not errors
            and applied
            and before
            and after
            and applied[-1].exposure_us == args.target_exposure_us
            and float(np.mean(after)) > float(np.mean(before)) * 2.0
        ),
        "timed_out": timed_out,
        "errors": errors,
        "initial_exposure_us": runtime["camera"].exposure_us,
        "applied_exposure_us": applied[-1].exposure_us if applied else None,
        "before_mean_dn": float(np.mean(before)) if before else None,
        "after_mean_dn": float(np.mean(after)) if after else None,
        "before_frames": len(before),
        "after_frames": len(after),
    }
    print(yaml.safe_dump(result, allow_unicode=True, sort_keys=False).rstrip())
    if args.output:
        dump_yaml(args.output, result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
