from __future__ import annotations

import itertools
import math
import random
import time
from dataclasses import dataclass
from typing import Any

import pennylane as qml
from pennylane import numpy as np

Point = tuple[int, int]


@dataclass
class PlannerConfig:
    q_layers: int = 1
    q_lr: float = 0.22
    q_steps: int = 8
    q_seed: int = 7


class GridEnvironment:
    def __init__(
        self,
        width: int,
        height: int,
        obstacles: set[Point],
        checkpoints: dict[str, Point],
        start_id: str,
        end_id: str,
    ):
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.checkpoints = checkpoints
        self.start_id = start_id
        self.end_id = end_id
        self._dist_cache: dict[tuple[Point, Point], float] = {}
        self._validate()

    def _validate(self):
        if self.start_id not in self.checkpoints:
            raise ValueError(f"start_id '{self.start_id}' not found in checkpoints")
        if self.end_id not in self.checkpoints:
            raise ValueError(f"end_id '{self.end_id}' not found in checkpoints")

        for p in self.checkpoints.values():
            if not self.in_bounds(p):
                raise ValueError(f"checkpoint out of bounds: {p}")
            if p in self.obstacles:
                raise ValueError(f"checkpoint placed on obstacle: {p}")

    @classmethod
    def default(cls, width: int = 14, height: int = 10):
        obstacles = set()

        for x in range(1, 13):
            if x not in (3, 10):
                obstacles.add((x, 3))
            if x not in (5, 8, 12):
                obstacles.add((x, 6))

        for y in range(1, 9):
            if y not in (2, 7):
                obstacles.add((7, y))

        checkpoints = {
            "A": (0, 0),
            "C1": (2, 8),
            "C2": (5, 1),
            "C3": (10, 8),
            "C4": (12, 2),
            "B": (13, 9),
        }

        for p in checkpoints.values():
            obstacles.discard(p)

        return cls(width, height, obstacles, checkpoints, start_id="A", end_id="B")

    @classmethod
    def random(
        cls,
        width: int = 14,
        height: int = 10,
        seed: int = 4,
        n_checkpoints: int = 4,
        obstacle_density: float = 0.18,
    ):
        rng = random.Random(seed)
        n_checkpoints = max(2, min(n_checkpoints, 8))
        obstacle_density = max(0.05, min(obstacle_density, 0.40))

        start = (0, 0)
        end = (width - 1, height - 1)

        for _ in range(200):
            obstacles: set[Point] = set()
            for x in range(width):
                for y in range(height):
                    p = (x, y)
                    if p in (start, end):
                        continue
                    if rng.random() < obstacle_density:
                        obstacles.add(p)

            free = [
                (x, y)
                for x in range(width)
                for y in range(height)
                if (x, y) not in obstacles and (x, y) not in (start, end)
            ]
            if len(free) < n_checkpoints:
                continue

            picked = rng.sample(free, n_checkpoints)
            checkpoints = {"A": start}
            for i, p in enumerate(picked, start=1):
                checkpoints[f"C{i}"] = p
            checkpoints["B"] = end

            env = cls(width, height, obstacles, checkpoints, start_id="A", end_id="B")
            if env.all_checkpoints_connected():
                return env

        return cls.default(width, height)

    @classmethod
    def from_spec(cls, spec: dict[str, Any]):
        width = int(spec["width"])
        height = int(spec["height"])

        raw_obstacles = spec.get("obstacles", [])
        obstacles = {tuple(map(int, p)) for p in raw_obstacles}

        raw_checkpoints = spec["checkpoints"]
        checkpoints: dict[str, Point] = {}
        declared_start = None
        declared_end = None

        for cp in raw_checkpoints:
            cid = str(cp["id"])
            checkpoints[cid] = (int(cp["x"]), int(cp["y"]))
            role = str(cp.get("role", cp.get("type", ""))).lower()
            if role == "start":
                declared_start = cid
            elif role == "end":
                declared_end = cid

        start_id = spec.get("start_id") or declared_start or ("A" if "A" in checkpoints else sorted(checkpoints.keys())[0])
        end_id = spec.get("end_id") or declared_end or ("B" if "B" in checkpoints else sorted(checkpoints.keys())[-1])

        return cls(width, height, obstacles, checkpoints, str(start_id), str(end_id))

    def in_bounds(self, p: Point) -> bool:
        return 0 <= p[0] < self.width and 0 <= p[1] < self.height

    def passable(self, p: Point) -> bool:
        return p not in self.obstacles

    def neighbors(self, p: Point) -> list[Point]:
        x, y = p
        cands = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [q for q in cands if self.in_bounds(q) and self.passable(q)]

    def all_checkpoints_connected(self) -> bool:
        start = self.checkpoints[self.start_id]
        queue = [start]
        seen = {start}

        for cur in queue:
            for nxt in self.neighbors(cur):
                if nxt in seen:
                    continue
                seen.add(nxt)
                queue.append(nxt)

        return all(p in seen for p in self.checkpoints.values())

    def shortest_path(self, start: Point, goal: Point) -> list[Point]:
        if start == goal:
            return [start]

        queue = [start]
        parent = {start: None}

        for cur in queue:
            for nxt in self.neighbors(cur):
                if nxt in parent:
                    continue
                parent[nxt] = cur
                if nxt == goal:
                    queue = []
                    break
                queue.append(nxt)

        if goal not in parent:
            return []

        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        return list(reversed(path))

    def shortest_distance(self, start: Point, goal: Point) -> float:
        key = (start, goal) if start <= goal else (goal, start)
        if key in self._dist_cache:
            return self._dist_cache[key]

        path = self.shortest_path(start, goal)
        dist = float(len(path) - 1) if path else float("inf")
        self._dist_cache[key] = dist
        return dist


def route_cost(env: GridEnvironment, order: list[str]) -> float:
    total = 0.0
    for i in range(len(order) - 1):
        total += env.shortest_distance(env.checkpoints[order[i]], env.checkpoints[order[i + 1]])
    return total


def full_grid_path(env: GridEnvironment, order: list[str]) -> list[Point]:
    out: list[Point] = []
    for i in range(len(order) - 1):
        seg = env.shortest_path(env.checkpoints[order[i]], env.checkpoints[order[i + 1]])
        if not seg:
            continue
        if out and seg[0] == out[-1]:
            out.extend(seg[1:])
        else:
            out.extend(seg)
    return out


def _mid_nodes(env: GridEnvironment) -> list[str]:
    return [k for k in env.checkpoints.keys() if k not in (env.start_id, env.end_id)]


def nearest_neighbor_route(env: GridEnvironment) -> tuple[list[str], float]:
    remaining = _mid_nodes(env)
    route = [env.start_id]
    current = env.start_id

    while remaining:
        nxt = min(remaining, key=lambda n: env.shortest_distance(env.checkpoints[current], env.checkpoints[n]))
        route.append(nxt)
        remaining.remove(nxt)
        current = nxt

    route.append(env.end_id)
    return route, route_cost(env, route)


def brute_force_route(env: GridEnvironment) -> tuple[list[str], float]:
    mids = _mid_nodes(env)

    if len(mids) > 7:
        return nearest_neighbor_route(env)

    best_route = None
    best_cost = float("inf")
    for perm in itertools.permutations(mids):
        route = [env.start_id] + list(perm) + [env.end_id]
        c = route_cost(env, route)
        if c < best_cost:
            best_cost = c
            best_route = route

    return best_route or [env.start_id, env.end_id], best_cost


class QMLRoutePlanner:
    def __init__(self, config: PlannerConfig, end_id: str):
        self.cfg = config
        self.end_id = end_id
        self.dev = qml.device("default.qubit", wires=2)
        self.rng = np.random.default_rng(config.q_seed)
        self.weights = 0.01 * self.rng.normal(size=(self.cfg.q_layers, 2, 3), requires_grad=True)
        self.is_trained = False

        @qml.qnode(self.dev)
        def circuit(features, weights):
            qml.AngleEmbedding(features, wires=range(2), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    def _features(self, env: GridEnvironment, current: str, candidate: str):
        cur = env.checkpoints[current]
        can = env.checkpoints[candidate]
        goal = env.checkpoints[self.end_id]

        d_cur_can = env.shortest_distance(cur, can)
        d_can_goal = env.shortest_distance(can, goal)
        scale = float(env.width + env.height)

        return np.array([min(d_cur_can / scale, 1.0), min(d_can_goal / scale, 1.0)], requires_grad=False)

    def _target_score(self, env: GridEnvironment, current: str, candidate: str):
        cur = env.checkpoints[current]
        can = env.checkpoints[candidate]
        goal = env.checkpoints[self.end_id]
        raw = env.shortest_distance(cur, can) + 0.35 * env.shortest_distance(can, goal)
        return 1.0 / (1.0 + raw)

    def _sample_training(self, env: GridEnvironment, n_samples: int = 36):
        nodes = [k for k in env.checkpoints if k != self.end_id]
        candidates = [k for k in env.checkpoints if k != env.start_id]
        xs = []
        ys = []

        for _ in range(n_samples):
            current = random.choice(nodes)
            candidate = random.choice(candidates)
            if current == candidate:
                continue
            xs.append(self._features(env, current, candidate))
            ys.append(self._target_score(env, current, candidate))

        return np.array(xs, requires_grad=False), np.array(ys, requires_grad=False)

    @staticmethod
    def _expval_to_unit(v):
        return (v + 1.0) / 2.0

    def _loss(self, w, xs, ys):
        losses = []
        for x, y in zip(xs, ys):
            pred = self._expval_to_unit(self._circuit(x, w))
            losses.append((pred - float(y)) ** 2)
        return np.mean(np.array(losses))

    def train(self, env: GridEnvironment):
        xs, ys = self._sample_training(env)
        opt = qml.GradientDescentOptimizer(stepsize=self.cfg.q_lr)

        w = self.weights
        for _ in range(self.cfg.q_steps):
            w = opt.step(lambda ww: self._loss(ww, xs, ys), w)

        self.weights = w
        self.is_trained = True

    def score(self, env: GridEnvironment, current: str, candidate: str):
        if not self.is_trained:
            self.train(env)
        x = self._features(env, current, candidate)
        return float(self._expval_to_unit(self._circuit(x, self.weights)))

    def route(self, env: GridEnvironment) -> tuple[list[str], float]:
        remaining = _mid_nodes(env)
        route = [env.start_id]
        current = env.start_id

        while remaining:
            nxt = max(remaining, key=lambda c: self.score(env, current, c))
            route.append(nxt)
            remaining.remove(nxt)
            current = nxt

        route.append(env.end_id)
        return route, route_cost(env, route)

    def circuit_ascii(self):
        sample_x = np.array([0.2, 0.3], requires_grad=False)
        drawer = qml.draw(self._circuit)
        return drawer(sample_x, self.weights)


def _build_env(
    randomize: bool,
    seed: int,
    n_checkpoints: int,
    obstacle_density: float,
    map_spec: dict[str, Any] | None,
) -> GridEnvironment:
    if map_spec is not None:
        env = GridEnvironment.from_spec(map_spec)
    elif randomize:
        env = GridEnvironment.random(
            width=14,
            height=10,
            seed=seed,
            n_checkpoints=n_checkpoints,
            obstacle_density=obstacle_density,
        )
    else:
        env = GridEnvironment.default()

    if not env.all_checkpoints_connected():
        raise ValueError("Map is not fully connected across checkpoints")
    return env


def run_benchmark(
    seed: int = 1,
    randomize: bool = True,
    n_checkpoints: int = 4,
    obstacle_density: float = 0.18,
    map_spec: dict[str, Any] | None = None,
):
    random.seed(seed)
    env = _build_env(randomize, seed, n_checkpoints, obstacle_density, map_spec)

    cfg = PlannerConfig(q_seed=seed)
    qml_planner = QMLRoutePlanner(cfg, end_id=env.end_id)

    t0 = time.perf_counter()
    nn_route, nn_cost = nearest_neighbor_route(env)
    nn_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    bf_route, bf_cost = brute_force_route(env)
    bf_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    q_route, q_cost = qml_planner.route(env)
    q_ms = (time.perf_counter() - t0) * 1000.0

    improvement = 0.0
    if nn_cost > 0 and math.isfinite(nn_cost):
        improvement = ((nn_cost - q_cost) / nn_cost) * 100.0

    checkpoints_out = [{"id": k, "x": v[0], "y": v[1]} for k, v in env.checkpoints.items()]

    return {
        "scenario": {
            "source": "custom_map" if map_spec is not None else ("random" if randomize else "default"),
            "randomized": randomize,
            "seed": seed,
            "obstacle_density": obstacle_density,
            "obstacle_count": len(env.obstacles),
            "checkpoint_count": len(env.checkpoints),
            "start_id": env.start_id,
            "end_id": env.end_id,
        },
        "grid": {
            "width": env.width,
            "height": env.height,
            "obstacles": sorted(list(env.obstacles)),
            "checkpoints": checkpoints_out,
        },
        "route": [
            {
                "checkpoint_id": name,
                "x": env.checkpoints[name][0],
                "y": env.checkpoints[name][1],
                "order": i + 1,
            }
            for i, name in enumerate(q_route)
        ],
        "full_path": full_grid_path(env, q_route),
        "total_distance_m": round(q_cost, 2),
        "estimated_time_s": round(q_cost / 0.2, 1),
        "method": "qml_variational",
        "optimization_improvement_vs_classical": f"{improvement:.1f}%",
        "quantum_circuit_depth": cfg.q_layers * 2,
        "n_qubits_used": 2,
        "benchmarks": [
            {"method": "nearest_neighbor", "distance": round(nn_cost, 2), "time_ms": round(nn_ms, 2), "route": nn_route},
            {"method": "brute_force", "distance": round(bf_cost, 2), "time_ms": round(bf_ms, 2), "route": bf_route},
            {"method": "qml_variational", "distance": round(q_cost, 2), "time_ms": round(q_ms, 2), "route": q_route},
        ],
        "circuit_diagram": qml_planner.circuit_ascii(),
    }
