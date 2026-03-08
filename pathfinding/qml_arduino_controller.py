from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import serial

from dual_mode_qml_backend import run_mode

RIGHT_TURN = {"N": "E", "E": "S", "S": "W", "W": "N"}
LEFT_TURN = {"N": "W", "W": "S", "S": "E", "E": "N"}
VEC_TO_HEADING = {(0, -1): "N", (1, 0): "E", (0, 1): "S", (-1, 0): "W"}


@dataclass
class GridBounds:
    min_x: int
    max_x: int
    min_y: int
    max_y: int


def parse_bounds(args: argparse.Namespace) -> GridBounds:
    if args.bounds:
        parts = [int(x.strip()) for x in args.bounds.split(",")]
        if len(parts) != 4:
            raise ValueError("--bounds must be minX,maxX,minY,maxY")
        bounds = GridBounds(parts[0], parts[1], parts[2], parts[3])
    elif args.grid_width is not None and args.grid_height is not None:
        bounds = GridBounds(0, args.grid_width - 1, 0, args.grid_height - 1)
    else:
        result = run_mode(args.mode, seed=args.seed, case_index=args.case_index)
        bounds = GridBounds(0, int(result["width"]) - 1, 0, int(result["height"]) - 1)

    if bounds.min_x > bounds.max_x or bounds.min_y > bounds.max_y:
        raise ValueError("Invalid bounds: min values must be <= max values")
    return bounds


def path_to_motion(path: list[tuple[int, int]], initial_heading: str) -> list[str]:
    if len(path) < 2:
        return []

    heading = initial_heading
    out: list[str] = []

    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        step = (x2 - x1, y2 - y1)
        if step not in VEC_TO_HEADING:
            raise ValueError(f"Non-grid step: {path[i]} -> {path[i + 1]}")

        target = VEC_TO_HEADING[step]
        if target != heading:
            if RIGHT_TURN[heading] == target:
                out.append("RIGHT")
            elif LEFT_TURN[heading] == target:
                out.append("LEFT")
            else:
                out.extend(["RIGHT", "RIGHT"])
            heading = target
        out.append("FORWARD")

    return out


def path_to_spec(path: list[tuple[int, int]]) -> str:
    return "PATH:" + "|".join(f"{x},{y}" for x, y in path)


class ArduinoSession:
    def __init__(self, port: str, baud: int, timeout_s: float):
        self.port = port
        self.baud = baud
        self.timeout_s = timeout_s
        self.ser = serial.Serial(port, baudrate=baud, timeout=0.15)

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _readline(self) -> str:
        raw = self.ser.readline().decode("utf-8", errors="ignore").strip()
        return raw

    def drain(self) -> None:
        while self.ser.in_waiting:
            self._readline()

    def send(self, line: str) -> None:
        self.ser.write((line + "\n").encode("utf-8"))
        self.ser.flush()

    def send_and_wait(self, line: str, ack_prefix: str) -> str:
        self.send(line)
        end = time.time() + self.timeout_s
        last_line = ""

        while time.time() < end:
            got = self._readline()
            if not got:
                continue
            last_line = got

            if got.startswith("ERR:"):
                raise RuntimeError(f"Arduino error for '{line}': {got}")
            if got.startswith(ack_prefix):
                return got

        raise TimeoutError(f"Timeout waiting for {ack_prefix} after '{line}'. Last line: {last_line}")

    def fail_safe_stop(self) -> None:
        try:
            self.send("STOP")
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Run QML pathfinding and execute on Arduino rover")
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM3")
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--mode", choices=["mapping", "path"], default="path")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--heading", choices=["N", "E", "S", "W"], default="E")
    parser.add_argument("--forward-ms", type=int, default=450)
    parser.add_argument("--turn-ms", type=int, default=330)
    parser.add_argument("--timeout-s", type=float, default=4.0)
    parser.add_argument("--bounds", help="minX,maxX,minY,maxY")
    parser.add_argument("--grid-width", type=int)
    parser.add_argument("--grid-height", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run_mode(args.mode, seed=args.seed, case_index=args.case_index)
    path = [(int(p["x"]), int(p["y"])) for p in result["path"]]
    if len(path) < 2:
        raise RuntimeError("QML backend returned path with fewer than 2 points")

    bounds = parse_bounds(args)

    # Enforce bounds before sending anything to hardware.
    for x, y in path:
        if not (bounds.min_x <= x <= bounds.max_x and bounds.min_y <= y <= bounds.max_y):
            raise RuntimeError(
                f"Path point {(x, y)} is outside configured bounds "
                f"({bounds.min_x},{bounds.max_x},{bounds.min_y},{bounds.max_y})"
            )

    motion = path_to_motion(path, args.heading)
    path_payload = path_to_spec(path)

    print("[INFO] Scenario:", result["title"])
    print("[INFO] Mode:", result["mode"])
    print("[INFO] Grid:", result["width"], "x", result["height"])
    print("[INFO] Path points:", len(path))
    print("[INFO] Motion commands:", len(motion))
    print("[INFO] Bounds:", f"{bounds.min_x},{bounds.max_x},{bounds.min_y},{bounds.max_y}")

    if args.dry_run:
        print("[DRY RUN] SET:BOUNDS:", f"SET:BOUNDS:{bounds.min_x},{bounds.max_x},{bounds.min_y},{bounds.max_y}")
        print("[DRY RUN] SET:POS:", f"SET:POS:{path[0][0]},{path[0][1]}")
        print("[DRY RUN] PATH payload:", path_payload)
        return 0

    session = ArduinoSession(args.port, args.baud, args.timeout_s)
    try:
        time.sleep(2.0)
        session.drain()

        session.send_and_wait("PING", "PONG")
        session.send_and_wait("MODE:MANUAL", "MODE:MANUAL")
        session.send_and_wait(f"SET:FORWARD_MS:{args.forward_ms}", "ACK:SET:FORWARD_MS:")
        session.send_and_wait(f"SET:TURN_MS:{args.turn_ms}", "ACK:SET:TURN_MS:")
        session.send_and_wait(f"SET:HEADING:{args.heading}", "ACK:SET:HEADING:")
        session.send_and_wait(
            f"SET:BOUNDS:{bounds.min_x},{bounds.max_x},{bounds.min_y},{bounds.max_y}",
            "ACK:SET:BOUNDS:",
        )
        session.send_and_wait(f"SET:POS:{path[0][0]},{path[0][1]}", "ACK:SET:POS:")
        ack = session.send_and_wait(path_payload, "ACK:PATH:QSIZE:")

        queued = int(ack.split(":")[-1])
        done = 0
        deadline = time.time() + max(args.timeout_s, 10.0) + queued * (max(args.forward_ms, args.turn_ms) / 1000.0)
        while done < queued and time.time() < deadline:
            line = session._readline()
            if not line:
                continue
            if line.startswith("ERR:"):
                raise RuntimeError(f"Arduino execution error: {line}")
            if line == "DONE":
                done += 1
            if line.startswith("ACK:") or line == "DONE":
                print("[SERIAL]", line)

        if done < queued:
            raise TimeoutError(f"Timed out waiting for path completion ({done}/{queued} DONE)")

        session.send_and_wait("GET:STATE", "STATE:POS:")
        session.send("STOP")
        print("[INFO] Execution complete")
        return 0

    except Exception as exc:
        session.fail_safe_stop()
        print(f"[ERROR] {exc}")
        return 1
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
