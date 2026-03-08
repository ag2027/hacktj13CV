from __future__ import annotations

import json
import math
import random
from functools import lru_cache
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pennylane as qml
from pennylane import numpy as np

Point = tuple[int, int]
TOTAL_CASES = 500
DATASET_PATH = Path(__file__).resolve().parent / 'gtmk_test_dataset.json'


@dataclass
class ScenarioConfig:
    width: int
    height: int
    start: Point
    goal: Point | None
    points: list[dict[str, Any]]
    title: str
    mode: str
    case_index: int


class QMLMovePolicy:
    """Real PennyLane move scorer used by the rover backend."""

    def __init__(self, seed: int = 7, layers: int = 1, steps: int = 8, lr: float = 0.18):
        self.seed = seed
        self.layers = layers
        self.steps = steps
        self.lr = lr
        self.dev = qml.device("default.qubit", wires=2)
        rng = np.random.default_rng(seed)
        self.weights = 0.01 * rng.normal(size=(layers, 2, 3), requires_grad=True)

        @qml.qnode(self.dev)
        def circuit(features, weights):
            qml.AngleEmbedding(features, wires=[0, 1], rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    def _features(self, progress_hint: float, safety_hint: float) -> np.ndarray:
        return np.array([
            max(0.0, min(progress_hint, 1.0)) * math.pi,
            max(0.0, min(safety_hint, 1.0)) * math.pi,
        ], requires_grad=False)

    def train(self) -> None:
        rng = np.random.default_rng(self.seed)
        xs: list[np.ndarray] = []
        ys: list[float] = []
        for _ in range(24):
            xs.append(self._features(float(rng.uniform(0.6, 1.0)), float(rng.uniform(0.6, 1.0))))
            ys.append(1.0)
        for _ in range(24):
            xs.append(self._features(float(rng.uniform(0.0, 0.5)), float(rng.uniform(0.0, 0.5))))
            ys.append(0.0)

        def loss_fn(weights):
            total = 0.0
            for x, y in zip(xs, ys):
                pred = float((self._circuit(x, weights) + 1.0) / 2.0)
                total += (pred - y) ** 2
            return total / max(1, len(xs))

        best = self.weights
        best_loss = loss_fn(best)
        for _ in range(self.steps * 6):
            candidate = best + 0.12 * rng.normal(size=best.shape)
            cand_loss = loss_fn(candidate)
            if cand_loss < best_loss:
                best = candidate
                best_loss = cand_loss
        self.weights = best

    def score(self, progress_hint: float, safety_hint: float) -> float:
        features = self._features(progress_hint, safety_hint)
        return float((self._circuit(features, self.weights) + 1.0) / 2.0)


class LocalExplorer:
    def __init__(self, scenario: ScenarioConfig, qml_policy: QMLMovePolicy):
        self.scenario = scenario
        self.qml = qml_policy
        self.width = scenario.width
        self.height = scenario.height
        self.start = scenario.start
        self.goal = scenario.goal
        self.point_by_coord = {(p["x"], p["y"]): p for p in scenario.points}
        self.known_safe: set[Point] = {self.start}
        self.known_hazards: set[Point] = set()
        self.known_blocked: set[Point] = set()
        self.covered: set[Point] = set()
        self.path: list[Point] = [self.start]
        self.timeline: list[str] = []
        self.debug: list[str] = []
        self.hazard_reports: list[dict[str, Any]] = []
        self.safe_reports: list[dict[str, Any]] = []
        self.blocked_reports: list[dict[str, Any]] = []
        self._hazard_ids: set[str] = set()
        self._safe_ids: set[str] = set()
        self._blocked_ids: set[str] = set()
        self.known_point_meta: dict[Point, dict[str, Any]] = {}
        self.last_move = (0, -1)
        self.sweep_direction = -1

    def in_bounds(self, p: Point) -> bool:
        return 0 <= p[0] < self.width and 0 <= p[1] < self.height

    def neighbors(self, p: Point) -> list[Point]:
        x, y = p
        return [q for q in [(x, y - 1), (x, y + 1), (x + 1, y), (x - 1, y)] if self.in_bounds(q)]

    def _point(self, p: Point) -> dict[str, Any] | None:
        return self.point_by_coord.get(p)

    def _sense_neighbors(self, current: Point) -> None:
        for n in self.neighbors(current):
            point = self._point(n)
            if point is None:
                self.known_safe.add(n)
                continue
            if point.get("flagged"):
                self.known_hazards.add(n)
                self.known_point_meta[n] = point
                if point["id"] not in self._hazard_ids:
                    self._hazard_ids.add(point["id"])
                    self.hazard_reports.append(point)
                    self.timeline.append(f"sensor hazard {point['id']} @ ({point['x']},{point['y']})")
                continue
            if point.get("blocked"):
                self.known_blocked.add(n)
                self.known_point_meta[n] = point
                if point["id"] not in self._blocked_ids:
                    self._blocked_ids.add(point["id"])
                    self.blocked_reports.append(point)
                    self.timeline.append(f"sensor blocked {point['id']} @ ({point['x']},{point['y']})")
                continue
            self.known_safe.add(n)
            self.known_point_meta[n] = point
            if point["id"] not in self._safe_ids:
                self.timeline.append(f"sensor safe-probe {point['id']} @ ({point['x']},{point['y']})")

    def _record_safe(self, p: Point) -> None:
        point = self.known_point_meta.get(p)
        if point and not point.get("flagged") and not point.get("blocked") and point["id"] not in self._safe_ids:
            self._safe_ids.add(point["id"])
            self.safe_reports.append(point)
            self.timeline.append(f"sensor safe {point['id']} @ ({point['x']},{point['y']})")

    def _shortest_known_path(self, start: Point, goal: Point) -> list[Point]:
        queue = deque([start])
        parent = {start: None}
        while queue:
            cur = queue.popleft()
            if cur == goal:
                break
            for nxt in self.neighbors(cur):
                if nxt in parent or nxt not in self.known_safe:
                    continue
                parent[nxt] = cur
                queue.append(nxt)
        if goal not in parent:
            return []
        out: list[Point] = []
        cur: Point | None = goal
        while cur is not None:
            out.append(cur)
            cur = parent[cur]
        return list(reversed(out))

    def _progress_hint(self, cell: Point) -> float:
        if self.goal is not None:
            d = abs(cell[0] - self.goal[0]) + abs(cell[1] - self.goal[1])
            scale = max(1, self.width + self.height)
            return 1.0 - min(d / scale, 1.0)
        column_bias = cell[0] / max(1, self.width - 1)
        row_bias = 1.0 - (cell[1] / max(1, self.height - 1))
        return 0.60 * column_bias + 0.40 * row_bias

    def _safety_hint(self, cell: Point) -> float:
        point = self.known_point_meta.get(cell)
        if point is None:
            return 0.7
        return max(0.0, 1.0 - float(point.get("risk", 0.0)))

    def _choose_local_move(self, current: Point) -> Point | None:
        candidates = [n for n in self.neighbors(current) if n in self.known_safe and n not in self.covered]
        if not candidates:
            return None

        def score(cell: Point) -> float:
            dx = cell[0] - current[0]
            dy = cell[1] - current[1]
            straight = 1.30 if (dx, dy) == self.last_move else 0.0
            column_sweep = 1.10 if dx == 0 and dy == self.sweep_direction else 0.0
            next_column = 0.82 if dx == 1 and dy == 0 else 0.0
            qml_score = self.qml.score(self._progress_hint(cell), self._safety_hint(cell))
            return qml_score + straight + column_sweep + next_column

        candidates.sort(key=score, reverse=True)
        return candidates[0]

    def _nearest_frontier(self, current: Point) -> Point | None:
        remain = [p for p in self.known_safe if p not in self.covered]
        if not remain:
            return None
        if self.goal is not None:
            remain.sort(key=lambda p: (abs(p[0] - self.goal[0]) + abs(p[1] - self.goal[1]), abs(p[0] - current[0]) + abs(p[1] - current[1])))
        else:
            remain.sort(key=lambda p: (abs(p[0] - current[0]) + abs(p[1] - current[1]), p[0], p[1]))
        return remain[0]

    def run(self) -> dict[str, Any]:
        current = self.start
        max_steps = self.width * self.height * 8
        steps = 0
        mode_goal = self.goal is not None

        while steps < max_steps:
            steps += 1
            self.covered.add(current)
            self._record_safe(current)
            self._sense_neighbors(current)

            if mode_goal and current == self.goal:
                self.timeline.append(f"goal reached @ ({current[0]},{current[1]})")
                break

            nxt = self._choose_local_move(current)
            if nxt is None:
                frontier = self._nearest_frontier(current)
                if frontier is None:
                    break
                path = self._shortest_known_path(current, frontier)
                if len(path) < 2:
                    break
                nxt = path[1]
                self.debug.append(f"frontier fallback ({current[0]},{current[1]}) -> ({frontier[0]},{frontier[1]})")

            self.path.append(nxt)
            self.timeline.append(f"move ({current[0]},{current[1]}) -> ({nxt[0]},{nxt[1]})")
            move = (nxt[0] - current[0], nxt[1] - current[1])
            self.last_move = move
            if move == (1, 0):
                self.sweep_direction *= -1
            current = nxt

        reachable_truth = self._reachable_truth()
        covered_count = len([p for p in self.covered if p in reachable_truth])
        reachable_count = len(reachable_truth)
        coverage_rate = covered_count / reachable_count if reachable_count else 0.0
        goal_reached = self.goal in self.path if self.goal is not None else True

        significant = []
        for point in self.scenario.points:
            coord = (point["x"], point["y"])
            if point.get("flagged"):
                label = "hazardous obstruction"
                detected = coord in self.known_hazards
            elif point.get("blocked"):
                label = "blocked obstruction"
                detected = coord in self.known_blocked
            else:
                label = "nonhazardous obstruction"
                detected = point["id"] in self._safe_ids
            significant.append({**point, "label": label, "detected": detected})

        return {
            "mode": self.scenario.mode,
            "title": self.scenario.title,
            "case_index": self.scenario.case_index,
            "width": self.width,
            "height": self.height,
            "start": {"x": self.start[0], "y": self.start[1]},
            "goal": None if self.goal is None else {"x": self.goal[0], "y": self.goal[1]},
            "points": self.scenario.points,
            "detected_points": [
                {**point, "label": ("hazardous obstruction" if point.get("flagged") else "blocked obstruction" if point.get("blocked") else "nonhazardous obstruction")}
                for point in self.scenario.points
                if (point["x"], point["y"]) in self.known_hazards or (point["x"], point["y"]) in self.known_blocked or point["id"] in self._safe_ids
            ],
            "path": [{"x": x, "y": y} for x, y in self.path],
            "covered": [{"x": x, "y": y} for x, y in self.covered],
            "hazards": self.hazard_reports,
            "blocked": self.blocked_reports,
            "safe_objects": self.safe_reports,
            "significant_points": significant,
            "timeline": self.timeline,
            "debug": self.debug,
            "stats": {
                "steps": len(self.path) - 1,
                "coverage_rate": round(coverage_rate, 4),
                "covered_cells": covered_count,
                "reachable_cells": reachable_count,
                "hazard_count": len(self.hazard_reports),
                "blocked_count": len(self.blocked_reports),
                "safe_count": len(self.safe_reports),
                "goal_reached": goal_reached,
            },
        }

    def _reachable_truth(self) -> set[Point]:
        blocked = {(p["x"], p["y"]) for p in self.scenario.points if p.get("flagged") or p.get("blocked")}
        queue = deque([self.start])
        seen = {self.start}
        while queue:
            cur = queue.popleft()
            for nxt in self.neighbors(cur):
                if nxt in seen or nxt in blocked:
                    continue
                seen.add(nxt)
                queue.append(nxt)
        return seen


@lru_cache(maxsize=1)
def load_dataset() -> dict[str, Any]:
    return json.loads(DATASET_PATH.read_text())


def base_case(case_index: int) -> dict[str, Any]:
    data = load_dataset()
    return data['cases'][case_index % len(data['cases'])]


def _perturb_points(base_points: list[dict[str, Any]], width: int, height: int, rng: random.Random, reserved: set[Point]) -> list[dict[str, Any]]:
    taken = set(reserved)
    out: list[dict[str, Any]] = []
    for point in base_points:
        placed = None
        offsets = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]
        rng.shuffle(offsets)
        for dx, dy in offsets:
            cand = (max(0, min(width - 1, point['x'] + dx)), max(0, min(height - 1, point['y'] + dy)))
            if cand in taken:
                continue
            placed = cand
            break
        if placed is None:
            placed = (point['x'], point['y'])
        taken.add(placed)
        clone = dict(point)
        clone['x'], clone['y'] = placed
        out.append(clone)
    return out


def build_mapping_scenario(case_index: int = 0) -> ScenarioConfig:
    case = base_case(case_index)
    width = int(case['map']['width'])
    height = int(case['map']['height'])
    rng = random.Random(1000 + case_index)
    start = (int(case['robot']['start']['x']), int(case['robot']['start']['y']))
    points = _perturb_points([dict(p) for p in case['points']], width, height, rng, {start})
    return ScenarioConfig(
        width=width,
        height=height,
        start=start,
        goal=None,
        points=points,
        title=f"{case['title']} | Mapping Case {case_index + 1}",
        mode='mapping',
        case_index=case_index,
    )


def _reachable_path_exists(width: int, height: int, start: Point, goal: Point, points: list[dict[str, Any]]) -> bool:
    blocked = {(p['x'], p['y']) for p in points if p.get('flagged') or p.get('blocked')}
    q = deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        if cur == goal:
            return True
        x, y = cur
        for nxt in [(x,y-1),(x,y+1),(x+1,y),(x-1,y)]:
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            if nxt in seen or nxt in blocked:
                continue
            seen.add(nxt)
            q.append(nxt)
    return False


def build_path_scenario(seed: int = 11, case_index: int = 0) -> ScenarioConfig:
    case = base_case(case_index)
    width = int(case['map']['width'])
    height = int(case['map']['height'])
    base_points = [dict(p) for p in case['points']]

    for attempt in range(200):
        rng = random.Random(seed * 1009 + case_index * 97 + attempt)
        start_candidates = [(x, y) for x in range(max(1, width // 2)) for y in range(max(1, height // 2))]
        goal_candidates = [(x, y) for x in range(width // 2, width) for y in range(height // 2, height)]
        start = rng.choice(start_candidates)
        goal = rng.choice(goal_candidates)
        while goal == start:
            goal = rng.choice(goal_candidates)

        points = _perturb_points(base_points, width, height, rng, {start, goal})
        occupied = {(p['x'], p['y']) for p in points}
        extra_count = max(3, 3 * len(base_points))
        free = [(x, y) for x in range(width) for y in range(height) if (x, y) not in occupied and (x, y) not in (start, goal)]
        rng.shuffle(free)
        extras: list[dict[str, Any]] = []
        for i, (x, y) in enumerate(free[:extra_count], start=1):
            flagged = (i % 3 == 0)
            blocked = not flagged
            extras.append({
                'id': f'extra_{case_index}_{i}',
                'x': x,
                'y': y,
                'kind': 'generated_obstruction' if blocked else 'generated_hazard',
                'flagged': flagged,
                'blocked': blocked,
                'risk': 0.92 if flagged else 0.74,
            })
        merged = points + extras
        if _reachable_path_exists(width, height, start, goal, merged):
            return ScenarioConfig(
                width=width,
                height=height,
                start=start,
                goal=goal,
                points=merged,
                title=f"{case['title']} | Point A to B Case {case_index + 1}",
                mode='path',
                case_index=case_index,
            )

    raise RuntimeError('could not generate reachable A->B scenario')


@lru_cache(maxsize=16)
def _trained_policy(seed: int) -> QMLMovePolicy:
    qml_policy = QMLMovePolicy(seed=seed)
    qml_policy.train()
    return qml_policy


@lru_cache(maxsize=256)
def _run_mode_cached(mode: str, seed: int, case_index: int) -> dict[str, Any]:
    qml_policy = _trained_policy(seed)
    scenario = build_mapping_scenario(case_index) if mode == 'mapping' else build_path_scenario(seed, case_index)
    result = LocalExplorer(scenario, qml_policy).run()
    result['qml'] = {
        'framework': 'PennyLane',
        'device': 'default.qubit',
        'circuit': 'AngleEmbedding + StronglyEntanglingLayers',
        'qubits': 2,
        'layers': qml_policy.layers,
        'trained': True,
        'note': 'Real QML circuit evaluated in Python backend for every scenario run.',
    }
    return result


def run_mode(mode: str, seed: int = 11, case_index: int = 0) -> dict[str, Any]:
    return json.loads(json.dumps(_run_mode_cached(mode, seed, case_index)))
