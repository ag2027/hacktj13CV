# Hybrid Arduino + Python Rover Control System

This project uses a hybrid architecture:
- Arduino Uno handles motor control and HC-SR04 sensing in real time.
- Python handles navigation decisions and sends commands over serial.
- Decision logic can be upgraded from classical logic to quantum machine learning (QML).

## Files
- `rover_hybrid.ino`: Arduino firmware
- `rover_nav.py`: Python serial navigation controller (`DecisionEngine`)
- `qml_decision_engine.py`: QML steering/action model for rover decisions
- `qml_simulator_app.py`: Flask simulator for distance-to-action testing
- `qml_route_planner.py`: Grid patrol routing (classical + genuine PennyLane QML)
- `qml_grid_app.py`: Flask app for 2D pathfinding animation and benchmarks
- `templates/index.html`: QML decision simulator UI
- `templates/grid_qml.html`: 2D animated grid patrol UI
- `requirements.txt`: Python dependencies

## Wiring Assumptions

### HC-SR04 to Arduino Uno
- VCC -> 5V
- GND -> GND
- TRIG -> D9
- ECHO -> D10

### L298N to Arduino Uno
- ENA -> D5 (PWM)
- IN1 -> D2
- IN2 -> D3
- IN3 -> D7
- IN4 -> D8
- ENB -> D6 (PWM)
- L298N GND -> Arduino GND (common ground required)

### Motors and Power
- Left motor -> L298N OUT1/OUT2
- Right motor -> L298N OUT3/OUT4
- Motor battery to L298N motor power input
- Arduino can be USB-powered during testing

## Serial Protocol
- Arduino -> Python: `DIST:<value>`
- Python -> Arduino: `FORWARD`, `LEFT`, `RIGHT`, `STOP`

## Library Requirements
- Arduino IDE (standard core)
- Python 3.9+
- `pyserial`, `flask`, `pennylane`

Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup Instructions (Hardware + Classical Controller)
1. Open `rover_hybrid.ino` in Arduino IDE.
2. Select board: **Arduino Uno** and correct serial port.
3. Upload firmware.
4. Place rover on a stand so wheels can spin safely.
5. Run Python controller:

```bash
python3 rover_nav.py --port /dev/ttyACM0 --baud 9600
```

Windows example:
```bash
python rover_nav.py --port COM3 --baud 9600
```

## QML Decision Simulator (Browser)
Run:
```bash
python3 qml_simulator_app.py
```
Open:
- `http://127.0.0.1:8000`

## NEW: 2D QML Grid Pathfinding Animation
This module animates a robot moving from `A` to `B` through checkpoint coverage in a grid world with obstacles.

Included methods:
- Classical nearest-neighbor baseline
- Classical brute-force optimal baseline
- PennyLane variational QML route planner (`AngleEmbedding` + `StronglyEntanglingLayers`)

Run:
```bash
python3 qml_grid_app.py
```
Open:
- `http://127.0.0.1:8010`

The UI shows:
- animated robot path on grid
- obstacle nodes and checkpoints
- benchmark comparison (distance + runtime)
- QML circuit diagram text output

## Output Schema (QML Route API)
`GET /api/run` returns:
- route checkpoint ordering
- full grid path for animation
- total distance and estimated time
- improvement vs classical nearest-neighbor
- quantum metadata (`n_qubits_used`, `quantum_circuit_depth`)
- benchmark rows for classical and QML methods

## Future Upgrades
- Add multiple floor-plan presets and random scenario generation.
- Add 2-opt and genetic classical baselines.
- Replace variational scorer with QAOA/VQE encoding for richer combinatorial optimization.
- Export benchmark plots to PNG automatically for pitch deck use.
- Add ROS2 adapter for direct rover mission execution.
