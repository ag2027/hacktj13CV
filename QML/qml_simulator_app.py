from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from qml_decision_engine import QMLDecisionEngine

app = Flask(__name__)
engine = QMLDecisionEngine()
engine.train()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/simulate")
def simulate():
    payload = request.get_json(force=True)
    distance_cm = float(payload.get("distance_cm", 50.0))
    turn_hint = float(payload.get("turn_hint", 0.0))

    command, probs = engine.predict(distance_cm, turn_hint)
    return jsonify(
        {
            "distance_cm": distance_cm,
            "turn_hint": turn_hint,
            "command": command,
            "probabilities": {
                "STOP": probs[0],
                "LEFT": probs[1],
                "RIGHT": probs[2],
                "FORWARD": probs[3],
            },
        }
    )


@app.post("/api/sequence")
def sequence():
    payload = request.get_json(force=True)
    distances = payload.get("distances", [])
    turn_hint = float(payload.get("turn_hint", 0.0))

    out = []
    for d in distances:
        command, probs = engine.predict(float(d), turn_hint)
        out.append(
            {
                "distance_cm": float(d),
                "command": command,
                "probabilities": {
                    "STOP": probs[0],
                    "LEFT": probs[1],
                    "RIGHT": probs[2],
                    "FORWARD": probs[3],
                },
            }
        )
    return jsonify({"results": out})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False, use_reloader=False)
