"""
Execute QML patrol plans on Arduino motor controller via serial commands.

Uses the route planner output (full grid path), converts it into turn/forward
motor actions, and streams commands: FORWARD, LEFT, RIGHT, STOP.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import serial

from qml_route_planner import run_benchmark

HEADING_TO_VEC = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0),
}
VEC_TO_HEADING = {v: k for k, v in HEADING_TO_VEC.items()}
RIGHT_TURN = {"N": "E", "E": "S", "S": "W", "W": "N"}
LEFT_TURN = {"N": "W", "W": "S", "S": "E", "E": "N"}


def load_map_spec(path: str | None):
    if not path:
        return None
    return json.loads(Path(path).read_text())


def turn_sequence(current: str, target: str):
    if current == target:
        return []
    if RIGHT_TURN[current] == target:
        return ["RIGHT"]
    if LEFT_TURN[current] == target:
        return ["LEFT"]
    # 180 turn: two right turns
    return ["RIGHT", "RIGHT"]


def path_to_motion(path: list[list[int]] | list[tuple[int, int]], initial_heading: str):
    if len(path) < 2:
        return []

    cmds: list[str] = []
    heading = initial_heading

    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        step = (x2 - x1, y2 - y1)
        if step not in VEC_TO_HEADING:
            raise ValueError(f"Non-grid step detected: {path[i]} -> {path[i+1]}")

        target_heading = VEC_TO_HEADING[step]
        turns = turn_sequence(heading, target_heading)
        cmds.extend(turns)
        heading = target_heading

        cmds.append("FORWARD")

    return cmds


def send_command(ser: serial.Serial, command: str):
    ser.write((command + "\n").encode("utf-8"))
    ser.flush()


def execute_motion(
    port: str,
    baud: int,
    commands: list[str],
    forward_s: float,
    turn_s: float,
    settle_s: float,
):
    with serial.Serial(port, baudrate=baud, timeout=0.2) as ser:
        time.sleep(2.0)
        send_command(ser, "STOP")
        time.sleep(0.2)

        print(f"[INFO] Executing {len(commands)} motion commands on {port}")
        for idx, cmd in enumerate(commands, start=1):
            print(f"[STEP {idx:03d}] {cmd}")
            send_command(ser, cmd)
            time.sleep(turn_s if cmd in ("LEFT", "RIGHT") else forward_s)
            send_command(ser, "STOP")
            time.sleep(settle_s)

        send_command(ser, "STOP")
        print("[INFO] Execution complete. Rover stopped.")


def main():
    parser = argparse.ArgumentParser(description="Run QML patrol route on Arduino rover")
    parser.add_argument("--port", help="Serial port (e.g. /dev/ttyACM0 or COM3)")
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--random", action="store_true", help="Use randomized map generation")
    parser.add_argument("--checkpoints", type=int, default=4)
    parser.add_argument("--density", type=float, default=0.18)
    parser.add_argument("--map-json", help="Path to custom map JSON for arbitrary environment")
    parser.add_argument("--initial-heading", choices=["N", "E", "S", "W"], default="E")
    parser.add_argument("--forward-s", type=float, default=0.45)
    parser.add_argument("--turn-s", type=float, default=0.33)
    parser.add_argument("--settle-s", type=float, default=0.08)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without sending serial")
    args = parser.parse_args()

    map_spec = load_map_spec(args.map_json)

    result = run_benchmark(
        seed=args.seed,
        randomize=args.random and map_spec is None,
        n_checkpoints=args.checkpoints,
        obstacle_density=args.density,
        map_spec=map_spec,
    )

    full_path = result["full_path"]
    commands = path_to_motion(full_path, initial_heading=args.initial_heading)

    print("[INFO] Scenario:", result["scenario"])
    print("[INFO] Route:", [r["checkpoint_id"] for r in result["route"]])
    print("[INFO] Grid path length:", len(full_path))
    print("[INFO] Motor command count:", len(commands))

    if args.dry_run:
        print("[DRY RUN] Commands:", commands)
        return

    if not args.port:
        raise ValueError("--port is required unless --dry-run is used")

    execute_motion(
        port=args.port,
        baud=args.baud,
        commands=commands,
        forward_s=args.forward_s,
        turn_s=args.turn_s,
        settle_s=args.settle_s,
    )


if __name__ == "__main__":
    main()
