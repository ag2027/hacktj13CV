#!/usr/bin/env python3
"""
collect_training_data.py

Collect rover training samples from manual driving sessions and append to CSV.

Each saved row has exactly:
- distance_cm
- turn_hint
- label   (0=left, 1=right)

Expected Arduino serial line format:
- DIST:<value>

Usage examples:
  python collect_training_data.py --port /dev/ttyACM0
  python collect_training_data.py --port COM3 --baud 9600 --output training_data.csv
  python collect_training_data.py --port /dev/ttyACM0 --min-interval-ms 100

How to log a sample in the prompt:
- Enter: turn_hint,label
  Example: 80,1  (right-biased right turn)
  Example: 25,0  (left-biased left turn)
- Type 'q' to quit.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import threading
import time
from pathlib import Path

import serial

DIST_RE = re.compile(r"^DIST:\s*([-+]?\d+(?:\.\d+)?)\s*$")


class SerialDistanceReader:
    def __init__(self, port: str, baud: int, timeout: float = 0.2):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser: serial.Serial | None = None
        self.latest_distance_cm: float | None = None
        self.last_update_ts: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.ser = serial.Serial(self.port, baudrate=self.baud, timeout=self.timeout)
        # Arduino Uno commonly resets on serial open
        time.sleep(2.0)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self.ser is not None
        while not self._stop.is_set():
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            except Exception:
                line = ""

            if not line:
                continue

            m = DIST_RE.match(line)
            if m:
                try:
                    dist = float(m.group(1))
                except ValueError:
                    continue
                self.latest_distance_cm = max(0.0, min(dist, 1000.0))
                self.last_update_ts = time.time()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass


def ensure_csv(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["distance_cm", "turn_hint", "label"])


def parse_user_entry(raw: str) -> tuple[int, int]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("Expected format: turn_hint,label")

    hint = int(parts[0])
    label = int(parts[1])

    if not (0 <= hint <= 100):
        raise ValueError("turn_hint must be in [0,100]")
    if label not in (0, 1):
        raise ValueError("label must be 0 (left) or 1 (right)")

    return hint, label


def append_row(path: Path, distance_cm: float, turn_hint: int, label: int) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([int(round(distance_cm)), turn_hint, label])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Log manual rover steering samples to training_data.csv"
    )
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyACM0 or COM3")
    parser.add_argument("--baud", type=int, default=9600, help="Serial baud rate")
    parser.add_argument(
        "--output",
        default="training_data.csv",
        help="Output CSV path (default: training_data.csv)",
    )
    parser.add_argument(
        "--min-interval-ms",
        type=int,
        default=0,
        help="Minimum time between saved samples (default: 0)",
    )
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()
    ensure_csv(output_path)

    reader = SerialDistanceReader(port=args.port, baud=args.baud)

    try:
        print(f"[INFO] Connecting to {args.port} @ {args.baud}...")
        reader.start()
        print(f"[INFO] Writing samples to: {output_path}")
        print("[INFO] Enter samples as: turn_hint,label   (example: 80,1)")
        print("[INFO] label: 0=left, 1=right")
        print("[INFO] Type 'q' to quit.\n")

        last_saved_ts = 0.0
        sample_count = 0

        while True:
            latest = reader.latest_distance_cm
            age = None
            if reader.last_update_ts is not None:
                age = time.time() - reader.last_update_ts

            if latest is None:
                prompt = "distance=NONE (waiting for DIST:...) > "
            else:
                age_text = "?" if age is None else f"{age:.2f}s"
                prompt = f"distance={latest:.1f}cm (age {age_text}) > "

            raw = input(prompt).strip()
            if raw.lower() in {"q", "quit", "exit"}:
                break
            if not raw:
                continue

            try:
                hint, label = parse_user_entry(raw)
            except ValueError as exc:
                print(f"[WARN] {exc}")
                continue

            if latest is None:
                print("[WARN] No distance reading yet; sample not saved.")
                continue

            now = time.time()
            if args.min_interval_ms > 0:
                min_dt = args.min_interval_ms / 1000.0
                if now - last_saved_ts < min_dt:
                    wait_left = min_dt - (now - last_saved_ts)
                    print(f"[WARN] Too soon. Wait {wait_left:.2f}s before next sample.")
                    continue

            append_row(output_path, latest, hint, label)
            last_saved_ts = now
            sample_count += 1
            print(f"[OK] saved #{sample_count}: distance_cm={int(round(latest))}, turn_hint={hint}, label={label}")

        print(f"\n[INFO] Done. Total samples saved this session: {sample_count}")
        print(f"[INFO] CSV path: {output_path}")
        return 0

    except serial.SerialException as exc:
        print(f"[ERROR] Serial connection failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        return 0
    finally:
        reader.stop()


if __name__ == "__main__":
    raise SystemExit(main())
