"""Fair-comparison baseline for ACO_MultiEntry_IGPU.

Same node generation, same `construct_path`, same `path_cost` as
`aco_multi_entry_igpu.ACO_MultiEntry_IGPU` — the *only* difference is
the pheromone update rule. This file uses the canonical evaporation
(rate 0.2) plus top-5 elitist deposit, matching the philosophy of
`aco_tsp.py`. History keys (P_new, kl, fisher_mean, gradient_norm,
mean_advantage) are populated so that `metrics.py` works on both runs.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils_math import kl_divergence, softmax_policy


class ACO_MultiEntry_Baseline:
    """Standard MAX-style ACO with evaporation + top-K elitist deposit."""

    def __init__(
        self,
        regions: List[Dict[str, Any]],
        num_entry: int = 8,
        n_ants: int = 30,
        n_iter: int = 150,
        alpha: float = 1.0,
        beta: float = 2.5,
        evaporation: float = 0.2,
        elite_k: int = 5,
        deposit_scale: float = 200.0,
        seed: Optional[int] = None,
    ) -> None:
        self.regions = regions
        self.num_entry = num_entry
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.elite_k = elite_k
        self.deposit_scale = deposit_scale

        self.rng = np.random.default_rng(seed)

        self.nodes: List[Tuple[float, float]] = []
        self.node_to_region: List[int] = []
        self.generate_nodes()
        self.n: int = len(self.nodes)

        self.tau: np.ndarray = np.ones((self.n, self.n), dtype=float)

        coords = np.asarray(self.nodes, dtype=float)
        diffs = coords[:, None, :] - coords[None, :, :]
        dist = np.linalg.norm(diffs, axis=2)
        self.dist: np.ndarray = dist
        self.eta: np.ndarray = 1.0 / (dist + 1e-5)

        self.history: List[Dict[str, Any]] = []

    def generate_nodes(self) -> None:
        for idx, region in enumerate(self.regions):
            cx, cy = region["center"]
            r = region["radius"]
            for i in range(self.num_entry):
                angle = 2 * math.pi * i / self.num_entry
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
                self.nodes.append((x, y))
                self.node_to_region.append(idx)

    def distance(self, i: int, j: int) -> float:
        return float(self.dist[i, j])

    def construct_path(self) -> List[int]:
        """Same per-step softmax + region-mask as the IGPU class."""
        node_to_region = np.asarray(self.node_to_region, dtype=int)
        n_regions = len(self.regions)

        start = int(self.rng.integers(0, self.n))
        path: List[int] = [start]
        visited_regions = {int(node_to_region[start])}

        while len(visited_regions) < n_regions:
            current = path[-1]
            allowed = np.array(
                [node_to_region[i] not in visited_regions for i in range(self.n)],
                dtype=bool,
            )

            row_mask = allowed[None, :]
            P_row = softmax_policy(
                self.tau[current:current + 1],
                self.eta[current:current + 1],
                self.alpha,
                self.beta,
                mask=row_mask,
            )[0]

            total = P_row.sum()
            if total <= 0.0:
                next_node = int(self.rng.choice(np.flatnonzero(allowed)))
            else:
                P_row = P_row / total
                next_node = int(self.rng.choice(self.n, p=P_row))

            path.append(next_node)
            visited_regions.add(int(node_to_region[next_node]))

        return path

    def path_cost(self, path: List[int]) -> float:
        cost = 0.0
        for i in range(len(path) - 1):
            cost += self.distance(path[i], path[i + 1])
        visited: set = set()
        for node in path:
            region = self.node_to_region[node]
            if region in visited:
                cost += 1000.0
            visited.add(region)
        return cost

    def _update_pheromones(
        self,
        tours: List[List[int]],
        costs: List[float],
    ) -> None:
        """Evaporate, then deposit `deposit_scale / cost` on the top-K tours' edges (symmetric)."""
        self.tau *= (1.0 - self.evaporation)

        order = np.argsort(costs)[: self.elite_k]
        for k in order:
            tour = tours[int(k)]
            cost = costs[int(k)]
            for i, j in zip(tour[:-1], tour[1:]):
                self.tau[i, j] += self.deposit_scale / cost
                self.tau[j, i] += self.deposit_scale / cost

    def run(self) -> Tuple[List[int], float, List[Dict[str, Any]]]:
        """Standard-ACO run. Returns (best_path, best_cost, history)."""
        best_cost: float = float("inf")
        best_path: Optional[List[int]] = None

        for it in range(self.n_iter):
            tours: List[List[int]] = []
            costs: List[float] = []

            for _ in range(self.n_ants):
                path = self.construct_path()
                cost = self.path_cost(path)
                tours.append(path)
                costs.append(cost)
                if cost < best_cost:
                    best_cost = cost
                    best_path = path

            P_old = softmax_policy(self.tau, self.eta, self.alpha, self.beta)
            self._update_pheromones(tours, costs)
            P_new = softmax_policy(self.tau, self.eta, self.alpha, self.beta)

            # Per-row Shannon entropy, averaged over active rows.
            row_sums = P_new.sum(axis=1)
            active = row_sums > 1e-12
            if active.any():
                P_act = P_new[active]
                row_ent = -np.sum(P_act * np.log(P_act + 1e-12), axis=1)
                mean_entropy = float(np.mean(row_ent))
            else:
                mean_entropy = 0.0

            diag: Dict[str, Any] = {
                "iteration": it + 1,
                "P_old": P_old,
                "P_new": P_new,
                "kl": kl_divergence(P_new, P_old),
                "fisher_mean": float("nan"),
                "gradient_norm": float("nan"),
                "natural_gradient_norm": float("nan"),
                "mean_advantage": float("nan"),
                "best_cost": best_cost,
                "mean_cost": float(np.mean(costs)),
                "mean_entropy": mean_entropy,
            }
            self.history.append(diag)

            print(
                f"Iter {it + 1:3d} | best={best_cost:8.2f} | mean={diag['mean_cost']:8.2f} "
                f"| KL={diag['kl']:.2e}"
            )

        assert best_path is not None
        return best_path, best_cost, self.history


if __name__ == "__main__":
    demo_regions = [
        {"center": (0.0, 0.0), "radius": 5.0},
        {"center": (30.0, 5.0), "radius": 4.0},
        {"center": (15.0, 25.0), "radius": 6.0},
        {"center": (40.0, 30.0), "radius": 3.5},
    ]
    aco = ACO_MultiEntry_Baseline(
        regions=demo_regions,
        num_entry=6,
        n_ants=10,
        n_iter=5,
        seed=0,
    )
    best_path, best_cost, history = aco.run()
    print(f"\nFinished. best_cost={best_cost:.2f}, path_len={len(best_path)}")
    print(f"history entries: {len(history)}")

    expected_keys = {
        "P_new", "kl", "fisher_mean", "gradient_norm", "mean_advantage",
    }
    assert expected_keys.issubset(history[-1].keys()), \
        f"history dict missing keys: {expected_keys - set(history[-1].keys())}"
    assert math.isnan(history[-1]["fisher_mean"])
    assert math.isnan(history[-1]["gradient_norm"])
    assert math.isnan(history[-1]["mean_advantage"])
    print("baseline self-tests passed.")
