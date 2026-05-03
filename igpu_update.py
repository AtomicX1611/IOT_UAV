"""Information-Geometric Pheromone Update (IGPU).

Replaces the heuristic ACO pheromone update with a per-iteration
natural-gradient step in the space of edge-selection policies. The
preconditioner is the diagonal of the Fisher information matrix, so the
effective step size in distribution space is (approximately) bounded by
`gamma`, independent of the absolute scale of `tau`.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils_math import (
    fisher_diagonal,
    kl_divergence,
    softmax_policy,
    tour_log_prob_gradient,
)


class IGPUEngine:
    """Natural-gradient pheromone updater with diagonal-Fisher preconditioning."""

    def __init__(
        self,
        n_nodes: int,
        alpha: float = 1.0,
        beta: float = 2.5,
        gamma: float = 0.05,
        lambda_damping: float = 1e-4,
        tau_min: float = 0.01,
        tau_max: float = 10.0,
    ) -> None:
        self.n_nodes: int = n_nodes
        self.alpha: float = alpha
        self.beta: float = beta
        self.gamma: float = gamma
        self.lambda_damping: float = lambda_damping
        self.tau_min: float = tau_min
        self.tau_max: float = tau_max
        self.history: List[Dict[str, Any]] = []

    def reset_history(self) -> None:
        """Clear the per-step diagnostics buffer."""
        self.history = []

    def compute_advantage(self, costs: List[float]) -> np.ndarray:
        """REINFORCE-style advantage A_k = (mean(cost) - cost_k) / (std(cost) + 1e-9)."""
        c = np.asarray(costs, dtype=float)
        std_c = c.std() + 1e-9
        return (c.mean() - c) / std_c

    def step(
        self,
        tau: np.ndarray,
        eta: np.ndarray,
        tours: List[List[int]],
        costs: List[float],
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """One natural-gradient update of `tau` from a batch of (tour, cost) pairs."""
        P_old = softmax_policy(tau, eta, self.alpha, self.beta, mask)

        A = self.compute_advantage(costs)

        G_total = np.zeros_like(tau, dtype=float)
        for k, tour in enumerate(tours):
            G_k = tour_log_prob_gradient(tour, P_old, tau, self.alpha)
            G_total += A[k] * G_k
        G_total /= max(len(tours), 1)

        F_diag = fisher_diagonal(P_old, tau, self.alpha, self.lambda_damping)
        natural_gradient = G_total / F_diag

        tau_new = np.clip(tau + self.gamma * natural_gradient, self.tau_min, self.tau_max)

        P_new = softmax_policy(tau_new, eta, self.alpha, self.beta, mask)

        diagnostics: Dict[str, Any] = {
            "P_old": P_old,
            "P_new": P_new,
            "kl": kl_divergence(P_new, P_old),
            "fisher_mean": float(F_diag.mean()),
            "gradient_norm": float(np.linalg.norm(G_total)),
            "natural_gradient_norm": float(np.linalg.norm(natural_gradient)),
            "mean_advantage": float(A.mean()),
        }
        self.history.append(diagnostics)
        return tau_new, diagnostics


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 8
    tau = rng.uniform(0.5, 1.5, size=(n, n))
    eta = rng.uniform(0.5, 1.5, size=(n, n))
    mask = ~np.eye(n, dtype=bool)
    np.fill_diagonal(tau, 1e-12)
    np.fill_diagonal(eta, 1e-12)

    engine = IGPUEngine(n_nodes=n)

    tours: List[List[int]] = []
    for _ in range(5):
        interior = list(rng.permutation(n - 1) + 1)
        tours.append([0] + interior + [0])
    costs = [float(rng.uniform(10.0, 50.0)) for _ in tours]

    tau_new, diag = engine.step(tau, eta, tours, costs, mask=mask)

    assert tau_new.shape == tau.shape
    assert np.all(tau_new >= engine.tau_min - 1e-12)
    assert np.all(tau_new <= engine.tau_max + 1e-12)
    assert diag["kl"] >= 0.0
    assert len(engine.history) == 1

    engine.reset_history()
    assert engine.history == []

    print("igpu_update.py self-tests passed.")
    print(f"  KL(P_new || P_old)        = {diag['kl']:.3e}")
    print(f"  ||grad||_F                = {diag['gradient_norm']:.3e}")
    print(f"  ||nat_grad||_F            = {diag['natural_gradient_norm']:.3e}")
    print(f"  mean Fisher diagonal      = {diag['fisher_mean']:.3e}")
    print(f"  mean advantage            = {diag['mean_advantage']:.3e}")
