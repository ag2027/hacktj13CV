"""
Hybrid Rover Navigator (Python Side)
------------------------------------
Reads ultrasonic distance from Arduino and sends drive commands.
"""

import argparse
import time
from dataclasses import dataclass

import serial


@dataclass
class DecisionConfig:
    stop_distance_cm: float = 12.0
    turn_distance_cm: float = 30.0
    left_right_toggle_start_left: bool = True


class DecisionEngine:
    """
    Classical decision module.

    Future QML/Q-inspired upgrade point:
    Replace decide_command() internals with a probabilistic or quantum-inspired
    planner that maps sensor/state input to action distribution.
    Keep the same method signature to preserve integration.
    """

    def __init__(self, config: DecisionConfig):
        self.config = config
        self._turn_left_next = config.left_right_toggle_start_left

    def decide_command(self, distance_cm: float) -> str:
        if distance_cm <= self.config.stop_distance_cm:
            return "STOP"

        if distance_cm <= self.config.turn_distance_cm:
            cmd = "LEFT" if self._turn_left_next else "RIGHT"
            self._turn_left_next = not self._turn_left_next
            return cmd

        return "FORWARD"


def parse_distance_line(line: str):
    if not line.startswith("DIST:"):
        return None
    raw = line.split(":", 1)[1].strip()
    try:
        return float(raw)
    except ValueError:
        return None


def send_command(ser: serial.Serial, command: str):
    ser.write((command + "\n").encode("utf-8"))
    ser.flush()


def run_controller(
    port: str,
    baud: int,
    stop_distance_cm: float,
    turn_distance_cm: float,
    keepalive_s: float,
):
    config = DecisionConfig(
        stop_distance_cm=stop_distance_cm,
        turn_distance_cm=turn_distance_cm,
    )
    engine = DecisionEngine(config)

    ser = serial.Serial(port, baudrate=baud, timeout=0.2)
    time.sleep(2.0)  # Allow Arduino serial reset and boot

    print(f"[INFO] Connected to {port} @ {baud} baud")
    print(
        f"[INFO] Thresholds -> stop <= {stop_distance_cm} cm, "
        f"turn <= {turn_distance_cm} cm"
    )

    last_command = "STOP"
    last_sent_time = 0.0
    send_command(ser, last_command)

    try:
        while True:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            now = time.time()

            if raw:
                distance = parse_distance_line(raw)
                if distance is not None:
                    command = engine.decide_command(distance)
                    print(f"[DEBUG] Sensor DIST={distance:.1f} cm -> Command={command}")

                    if command != last_command:
                        send_command(ser, command)
                        last_command = command
                        last_sent_time = now

            # Keepalive prevents Arduino timeout even if decision stays unchanged.
            if now - last_sent_time >= keepalive_s:
                send_command(ser, last_command)
                last_sent_time = now

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received. Sending STOP and closing.")
        send_command(ser, "STOP")
    finally:
        ser.close()


def main():
    parser = argparse.ArgumentParser(description="Hybrid rover navigation controller")
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyACM0 or COM3")
    parser.add_argument("--baud", type=int, default=9600, help="Serial baud rate")
    parser.add_argument("--stop-cm", type=float, default=12.0, help="Stop threshold (cm)")
    parser.add_argument("--turn-cm", type=float, default=30.0, help="Turn threshold (cm)")
    parser.add_argument(
        "--keepalive-s",
        type=float,
        default=0.35,
        help="Seconds between repeated command sends",
    )
    args = parser.parse_args()

    if args.stop_cm >= args.turn_cm:
        raise ValueError("--stop-cm must be less than --turn-cm")

    run_controller(
        port=args.port,
        baud=args.baud,
        stop_distance_cm=args.stop_cm,
        turn_distance_cm=args.turn_cm,
        keepalive_s=args.keepalive_s,
    )


if __name__ == "__main__":
    main()
