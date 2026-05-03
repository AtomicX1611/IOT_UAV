"""Math primitives for the Information-Geometric Pheromone Update (IGPU).

Pure numpy functions: row-stochastic policy, KL divergence, Fisher
information diagonal, and per-tour log-probability gradient.
"""

from typing import List, Optional

import numpy as np

_EPS: float = 1e-12


def softmax_policy(
    tau: np.ndarray,
    eta: np.ndarray,
    alpha: float,
    beta: float,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Row-stochastic ACO policy p_ij ∝ tau_ij^alpha * eta_ij^beta.

    `mask` (boolean, same shape as `tau`) zeros disallowed edges before
    normalization. A small epsilon is added to the row denominators.
    """
    weights = np.power(tau, alpha) * np.power(eta, beta)
    if mask is not None:
        weights = weights * mask.astype(weights.dtype)
    row_sums = weights.sum(axis=1, keepdims=True) + _EPS
    return weights / row_sums


def log_policy(P: np.ndarray) -> np.ndarray:
    """Elementwise log with epsilon for numerical safety."""
    return np.log(P + _EPS)


def kl_divergence(P_new: np.ndarray, P_old: np.ndarray) -> float:
    """Mean per-row KL divergence  E_i[ Σ_j P_new[i,j] * log(P_new/P_old) ].

    Rows whose P_new sums to ~0 (fully masked) are skipped.
    """
    row_sums = P_new.sum(axis=1)
    keep = row_sums > _EPS
    if not np.any(keep):
        return 0.0
    log_ratio = log_policy(P_new[keep]) - log_policy(P_old[keep])
    per_row_kl = np.sum(P_new[keep] * log_ratio, axis=1)
    return float(np.mean(per_row_kl))


def fisher_diagonal(
    P: np.ndarray,
    tau: np.ndarray,
    alpha: float,
    lambda_damping: float = 1e-4,
) -> np.ndarray:
    """Diagonal of the Fisher information matrix wrt log-pheromone parameters.

    F_ij = (alpha / tau_ij)^2 * P_ij * (1 - P_ij) + lambda_damping.
    The damping term guarantees strictly positive entries.
    """
    safe_tau = tau + _EPS
    return (alpha / safe_tau) ** 2 * P * (1.0 - P) + lambda_damping


def tour_log_prob_gradient(
    tour: List[int],
    P: np.ndarray,
    tau: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Gradient of log π(tour) wrt log-pheromones, sparse over visited edges.

    G[i,j] = (alpha / tau[i,j]) * (1 - P[i,j]) for each consecutive edge
    (i, j) in `tour`; all other entries are zero.
    """
    G = np.zeros_like(tau, dtype=float)
    safe_tau = tau + _EPS
    for i, j in zip(tour[:-1], tour[1:]):
        G[i, j] = (alpha / safe_tau[i, j]) * (1.0 - P[i, j])
    return G


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 8
    tau = rng.uniform(0.1, 1.0, size=(n, n))
    eta = rng.uniform(0.1, 1.0, size=(n, n))
    np.fill_diagonal(tau, _EPS)
    np.fill_diagonal(eta, _EPS)

    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)

    P = softmax_policy(tau, eta, alpha=1.0, beta=2.0, mask=mask)
    assert np.allclose(P.sum(axis=1), 1.0), "softmax_policy rows must sum to 1"

    assert abs(kl_divergence(P, P)) < 1e-9, "KL(P || P) must be ~0"

    F = fisher_diagonal(P, tau, alpha=1.0)
    assert np.all(F > 0.0), "fisher_diagonal entries must be strictly positive"

    tour = [0, 3, 5, 1, 7, 2, 4, 6, 0]
    G = tour_log_prob_gradient(tour, P, tau, alpha=1.0)
    assert G.shape == tau.shape
    assert np.count_nonzero(G) == len(tour) - 1, "gradient nonzeros = #edges in tour"

    print("utils_math.py self-tests passed.")
