# Hybrid Arduino + Python Rover Control System

This project uses a hybrid architecture:
- Arduino Uno handles motor control and HC-SR04 sensing in real time.
- Python handles navigation decisions and sends commands over serial.
- Decision logic can be upgraded from classical logic to quantum machine learning (QML).

## Files
- `rover_hybrid.ino`: Arduino firmware
- `rover_nav.py`: Python serial navigation controller (`DecisionEngine`)
- `QML/qml_decision_engine.py`: QML steering/action model for rover decisions
- `QML/qml_simulator_app.py`: Flask simulator for distance-to-action testing
- `QML/qml_route_planner.py`: Grid patrol routing (classical + PennyLane QML)
- `QML/qml_grid_app.py`: Flask app for 2D pathfinding animation and benchmarks
- `QML/patrol_to_arduino.py`: Executes planned QML route on Arduino motors over serial
- `QML/sample_map.json`: Example arbitrary map input
- `QML/templates/index.html`: QML decision simulator UI
- `QML/templates/grid_qml.html`: 2D animated grid patrol UI
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
- Python/Host -> Arduino: `FORWARD`, `LEFT`, `RIGHT`, `STOP`
- Timed protocol: `FORWARD:<ms>`, `LEFT:<ms>`, `RIGHT:<ms>`
- Mode control: `MODE:AUTO`, `MODE:MANUAL`

## Library Requirements
- Arduino IDE (standard core)
- Python 3.9+
- `pyserial`, `flask`, `pennylane`

Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup Instructions (Arduino Standalone, No Python Required)
1. Open `rover_hybrid.ino` in Arduino IDE.
2. Select board: **Arduino Uno** and correct serial port.
3. Upload firmware.
4. Place rover on a stand, then power the rover.
5. Rover starts in `AUTO` mode by default and drives using onboard ultrasonic logic.

Optional serial mode commands from Serial Monitor or host:
- `MODE:AUTO`
- `MODE:MANUAL`
- `FORWARD`, `LEFT`, `RIGHT`, `STOP`
- `FORWARD:<ms>`, `LEFT:<ms>`, `RIGHT:<ms>`
- `SET:STOP:<cm>`, `SET:TURN:<cm>`

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

## Arduino-Direct Motion Notes
- Upload `rover_hybrid.ino` first; this firmware now supports timed motor commands directly.
- `QML/patrol_to_arduino.py` sends timed commands so execution timing runs on Arduino.
- Firmware responses include `READY`, `ACK:<command>`, `DONE`, and `ERR:UNKNOWN:<cmd>`.

## QML Decision Simulator (Browser)
Run:
```bash
python3 QML/qml_simulator_app.py
```
Open:
- `http://127.0.0.1:8000`

## 2D QML Grid Pathfinding Animation
Run:
```bash
python3 QML/qml_grid_app.py
```
Open:
- `http://127.0.0.1:8010`

## Execute QML Route on Arduino
Dry run:
```bash
python3 QML/patrol_to_arduino.py --map-json QML/sample_map.json --dry-run
```

Hardware run:
```bash
python3 QML/patrol_to_arduino.py --port /dev/ttyACM0 --map-json QML/sample_map.json --initial-heading E
```
