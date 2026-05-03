"""ACO_MultiEntry with Information-Geometric Pheromone Update (IGPU).

Drop-in structural replacement for aco_multi_entry.ACO_MultiEntry. The
heuristic evaporation + elitist-deposit rule is replaced by a single
natural-gradient step (see igpu_update.IGPUEngine) — IGPU is the SOLE
pheromone update mechanism, by experimental design.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from igpu_update import IGPUEngine
from utils_math import softmax_policy


class ACO_MultiEntry_IGPU:
    """Multi-entry ACO whose pheromone update is replaced by IGPU."""

    def __init__(
        self,
        regions: List[Dict[str, Any]],
        num_entry: int = 8,
        n_ants: int = 30,
        n_iter: int = 150,
        alpha: float = 1.0,
        beta: float = 2.5,
        gamma: float = 0.05,
        lambda_damping: float = 1e-4,
        tau_min: float = 0.01,
        tau_max: float = 10.0,
        seed: Optional[int] = None,
    ) -> None:
        self.regions = regions
        self.num_entry = num_entry
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta

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

        # Zero self-loops so the policy never concentrates on staying put.
        np.fill_diagonal(self.tau, 0.0)
        np.fill_diagonal(self.eta, 0.0)
        self.mask: np.ndarray = ~np.eye(self.n, dtype=bool)

        self.igpu = IGPUEngine(
            n_nodes=self.n,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lambda_damping=lambda_damping,
            tau_min=tau_min,
            tau_max=tau_max,
        )

        self.history: List[Dict[str, Any]] = []

    def generate_nodes(self) -> None:
        """Place `num_entry` evenly-spaced entry points on each region's circumference."""
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
        """Greedy-stochastic path that visits every region exactly once.

        Per-step transition probabilities come from `softmax_policy` over
        the row of the current node, with a mask zeroing nodes whose
        region has already been visited — so the policy is identical to
        the one IGPU sees, modulo the per-step legal-action constraint.
        """
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
        """Euclidean tour length plus a 1000-unit penalty per region revisit."""
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

    def run(self) -> Tuple[List[int], float, List[Dict[str, Any]]]:
        """Run IGPU-driven ACO. Returns (best_path, best_cost, history)."""
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

            self.tau, diag = self.igpu.step(self.tau, self.eta, tours, costs, mask=self.mask)
            diag = dict(diag)
            diag["iteration"] = it + 1
            diag["best_cost"] = best_cost
            diag["mean_cost"] = float(np.mean(costs))

            # Per-row Shannon entropy, averaged over active rows.
            P_new = diag["P_new"]
            row_sums = P_new.sum(axis=1)
            active = row_sums > 1e-12
            if active.any():
                P_act = P_new[active]
                row_ent = -np.sum(P_act * np.log(P_act + 1e-12), axis=1)
                diag["mean_entropy"] = float(np.mean(row_ent))
            else:
                diag["mean_entropy"] = 0.0

            self.history.append(diag)

            print(
                f"Iter {it + 1:3d} | best={best_cost:8.2f} | mean={diag['mean_cost']:8.2f} "
                f"| KL={diag['kl']:.2e} | |g|={diag['gradient_norm']:.2e} "
                f"| |ng|={diag['natural_gradient_norm']:.2e}"
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
    aco = ACO_MultiEntry_IGPU(
        regions=demo_regions,
        num_entry=6,
        n_ants=10,
        n_iter=5,
        seed=0,
    )
    best_path, best_cost, history = aco.run()
    print(f"\nFinished. best_cost={best_cost:.2f}, path_len={len(best_path)}")
    print(f"history entries: {len(history)}")
