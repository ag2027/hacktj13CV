# Pathfinding Demo

Run locally:

```bash
cd pathfinding
python3 dual_mode_qml_app.py
```

Open <http://127.0.0.1:8030>.

This folder contains the dual-mode rover pathfinding demo:
- `Full Mapping`: local sensing only
- `Point A to B`: goal-only knowledge plus local sensing

The backend uses PennyLane for QML scoring.

## Arduino Integration

Use the QML backend path and execute it on the Arduino rover over serial:

```bash
cd pathfinding
python3 qml_arduino_controller.py --port COM3 --mode path --case-index 0 --seed 11 --heading E
```

Manual bounds options:

```bash
python3 qml_arduino_controller.py --port COM3 --bounds 0,13,0,9
python3 qml_arduino_controller.py --port COM3 --grid-width 14 --grid-height 10
```

The controller sends:
- `MODE:MANUAL`
- `SET:FORWARD_MS`, `SET:TURN_MS`
- `SET:HEADING`
- `SET:BOUNDS`, `SET:POS`
- `PATH:x,y|x,y|...`

Safety behavior:
- Waits for `ACK:*` and `DONE` responses
- On timeout or `ERR:*`, sends `STOP`

## Test Plan

1. Simulation/dry run:
- `python3 qml_arduino_controller.py --port COM3 --dry-run`
- Verify printed bounds, start position, and PATH payload.

2. Hardware-in-loop basic:
- Upload `rover_hybrid/rover_hybrid.ino`.
- Run controller with `--case-index 0`.
- Confirm serial shows `ACK:PATH:QSIZE:*` and matching `DONE` count.

3. Boundary enforcement:
- Run with tighter bounds (for example `--bounds 0,5,0,5`).
- Confirm out-of-bounds paths are rejected before motion.
- In AUTO mode, verify rover turns when heading would cross edge.

4. Obstacle behavior:
- Place obstacle ahead of rover in AUTO mode.
- Verify turn behavior in stop/turn zone and forward movement when clear.
