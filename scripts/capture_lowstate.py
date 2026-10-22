#!/usr/bin/env python3
"""Capture Unitree G1 LowState samples to CSV (read-only; publishes no commands)."""

import argparse
import csv
import threading
import time
from pathlib import Path

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0, help="capture duration in seconds")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain; simulator guide uses 0")
    parser.add_argument("--interface", default="lo", help="DDS network interface; sim uses lo")
    parser.add_argument("--motors", type=int, default=29, help="number of G1 motor slots to record")
    parser.add_argument("--output", required=True, help="output CSV path")
    args = parser.parse_args()
    if args.duration <= 0 or args.motors <= 0:
        parser.error("--duration and --motors must be positive")

    lock = threading.Lock()
    rows = []
    errors = []

    def on_state(msg):
        try:
            now = time.time()
            imu = msg.imu_state
            row = {
                "host_time_unix": now,
                "mode_machine": int(msg.mode_machine),
                "imu_quat_w": float(imu.quaternion[0]),
                "imu_quat_x": float(imu.quaternion[1]),
                "imu_quat_y": float(imu.quaternion[2]),
                "imu_quat_z": float(imu.quaternion[3]),
            }
            for i, value in enumerate(imu.gyroscope):
                row[f"gyro_{i}"] = float(value)
            for i, value in enumerate(imu.accelerometer):
                row[f"accel_{i}"] = float(value)
            motors = msg.motor_state
            for i in range(min(args.motors, len(motors))):
                motor = motors[i]
                row[f"m{i}_q"] = float(motor.q)
                row[f"m{i}_dq"] = float(motor.dq)
                row[f"m{i}_tau_est"] = float(motor.tau_est)
            with lock:
                rows.append(row)
        except Exception as exc:  # keep callback failures visible to the caller
            with lock:
                errors.append(repr(exc))

    ChannelFactoryInitialize(args.domain, args.interface)
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(on_state, 10)
    print(
        f"Read-only capture: rt/lowstate domain={args.domain} interface={args.interface}; "
        f"duration={args.duration}s; output={args.output}",
        flush=True,
    )
    time.sleep(args.duration)
    with lock:
        captured = list(rows)
        callback_errors = list(errors)
    if callback_errors:
        raise RuntimeError("LowState callback failed: " + callback_errors[0])
    if not captured:
        raise RuntimeError("No LowState received; check simulator/robot, DDS domain and interface")

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(captured[0].keys())
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(captured)
    print(f"Saved {len(captured)} samples to {output} (~{len(captured) / args.duration:.1f} samples/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
