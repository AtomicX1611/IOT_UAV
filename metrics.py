"""Diagnostics for IGPU-ACO experiments.

Operates on the per-iteration history produced by IGPUEngine.step (or
the run-loop in ACO_MultiEntry_IGPU). All inputs are plain lists/dicts
so this module has no dependency on the optimiser internals.
"""

from typing import Any, Dict, List

import numpy as np


def oscillation_rate(
    history: List[Dict[str, Any]],
    region_assignments: List[int],
    window: int = 20,
) -> float:
    """Fraction of consecutive iterations where the dominant entry-point flips.

    For each region r, the dominant entry-point at iteration t is
    argmax_j Σ_i P_new[i, j] over j ∈ r. The oscillation rate is the
    fraction of consecutive (t, t+1) pairs whose argmax differs,
    averaged over regions, computed on the trailing `window` iterations.
    """
    if len(history) < 2:
        return 0.0

    region_arr = np.asarray(region_assignments, dtype=int)
    regions = np.unique(region_arr)

    tail = history[-window:] if window < len(history) else history
    if len(tail) < 2:
        return 0.0

    rates: List[float] = []
    for r in regions:
        cols = np.flatnonzero(region_arr == r)
        if cols.size == 0:
            continue
        argmaxes: List[int] = []
        for entry in tail:
            P = np.asarray(entry["P_new"])
            incoming = P[:, cols].sum(axis=0)
            argmaxes.append(int(cols[np.argmax(incoming)]))
        flips = sum(1 for a, b in zip(argmaxes[:-1], argmaxes[1:]) if a != b)
        rates.append(flips / (len(argmaxes) - 1))

    return float(np.mean(rates)) if rates else 0.0


def iterations_to_converge(
    cost_history: List[float],
    tolerance: float = 0.05,
) -> int:
    """Smallest t where cost_history[t] ≤ (1 + tolerance) · min(cost_history)."""
    if not cost_history:
        return 0
    target = (1.0 + tolerance) * min(cost_history)
    for t, c in enumerate(cost_history):
        if c <= target:
            return t
    return len(cost_history) - 1


def kl_trajectory(history: List[Dict[str, Any]]) -> List[float]:
    """Per-iteration KL(P_new ‖ P_old) values."""
    return [float(entry["kl"]) for entry in history]


def summarize_run(
    history: List[Dict[str, Any]],
    cost_history: List[float],
    region_assignments: List[int],
) -> Dict[str, float]:
    """Aggregate run-level diagnostics into a single dict."""
    kls = kl_trajectory(history) if history else [0.0]
    fishers = [float(entry["fisher_mean"]) for entry in history] if history else [0.0]
    final_adv = float(history[-1]["mean_advantage"]) if history else 0.0

    return {
        "final_cost": float(cost_history[-1]) if cost_history else float("nan"),
        "iter_to_converge": int(iterations_to_converge(cost_history)),
        "mean_kl": float(np.mean(kls)),
        "max_kl": float(np.max(kls)),
        "final_oscillation": float(oscillation_rate(history, region_assignments)),
        "mean_fisher": float(np.mean(fishers)),
        "mean_advantage_final": final_adv,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_regions = 3
    entries_per_region = 4
    n = n_regions * entries_per_region
    region_assignments = [r for r in range(n_regions) for _ in range(entries_per_region)]

    history: List[Dict[str, Any]] = []
    cost_history: List[float] = []
    cost = 100.0
    for it in range(50):
        W = rng.uniform(0.1, 1.0, size=(n, n))
        np.fill_diagonal(W, 0.0)
        P_new = W / (W.sum(axis=1, keepdims=True) + 1e-12)
        P_old = P_new + rng.normal(scale=1e-3, size=P_new.shape)
        P_old = np.clip(P_old, 1e-9, None)
        P_old = P_old / P_old.sum(axis=1, keepdims=True)
        history.append({
            "P_old": P_old,
            "P_new": P_new,
            "kl": float(rng.uniform(1e-5, 1e-3)),
            "fisher_mean": float(rng.uniform(0.05, 0.2)),
            "gradient_norm": float(rng.uniform(1e-3, 1e-2)),
            "natural_gradient_norm": float(rng.uniform(1.0, 10.0)),
            "mean_advantage": float(rng.normal(scale=1e-3)),
        })
        cost *= rng.uniform(0.97, 1.005)
        cost_history.append(cost)

    osc = oscillation_rate(history, region_assignments, window=20)
    assert 0.0 <= osc <= 1.0

    conv = iterations_to_converge(cost_history)
    assert 0 <= conv < len(cost_history)

    kls = kl_trajectory(history)
    assert len(kls) == len(history)

    summary = summarize_run(history, cost_history, region_assignments)
    expected_keys = {
        "final_cost",
        "iter_to_converge",
        "mean_kl",
        "max_kl",
        "final_oscillation",
        "mean_fisher",
        "mean_advantage_final",
    }
    assert set(summary.keys()) == expected_keys

    print("metrics.py self-tests passed.")
    for k, v in summary.items():
        print(f"  {k:24s} = {v:.6g}")
