from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from qml_route_planner import run_benchmark

app = Flask(__name__)
_CACHE: dict[tuple, dict] = {}


def _bool_arg(name: str, default: bool) -> bool:
    raw = request.args.get(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _scenario_key(randomize: bool, seed: int, n_checkpoints: int, obstacle_density: float, map_spec: dict | None):
    if map_spec is not None:
        return ("custom", str(map_spec))
    return ("generated", randomize, seed, n_checkpoints, round(obstacle_density, 3))


@app.get("/")
def index():
    return render_template("grid_qml.html")


@app.get("/api/run")
def api_run():
    randomize = _bool_arg("random", True)
    seed = int(request.args.get("seed", 4))
    n_checkpoints = int(request.args.get("checkpoints", 4))
    obstacle_density = float(request.args.get("density", 0.18))

    key = _scenario_key(randomize, seed, n_checkpoints, obstacle_density, None)
    if key not in _CACHE:
        _CACHE[key] = run_benchmark(
            seed=seed,
            randomize=randomize,
            n_checkpoints=n_checkpoints,
            obstacle_density=obstacle_density,
            map_spec=None,
        )
    return jsonify(_CACHE[key])


@app.get("/api/refresh")
def api_refresh():
    randomize = _bool_arg("random", True)
    seed = int(request.args.get("seed", 4))
    n_checkpoints = int(request.args.get("checkpoints", 4))
    obstacle_density = float(request.args.get("density", 0.18))

    key = _scenario_key(randomize, seed, n_checkpoints, obstacle_density, None)
    _CACHE[key] = run_benchmark(
        seed=seed,
        randomize=randomize,
        n_checkpoints=n_checkpoints,
        obstacle_density=obstacle_density,
        map_spec=None,
    )
    return jsonify(_CACHE[key])


@app.post("/api/solve")
def api_solve_custom_map():
    payload = request.get_json(force=True)
    map_spec = payload.get("map")
    if not isinstance(map_spec, dict):
        return jsonify({"error": "payload must include map object"}), 400

    key = _scenario_key(False, 0, 0, 0.0, map_spec)
    if key not in _CACHE:
        _CACHE[key] = run_benchmark(map_spec=map_spec, randomize=False)
    return jsonify(_CACHE[key])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8010, debug=False, use_reloader=False)
