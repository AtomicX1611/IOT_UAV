import csv
import os
import random
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from aco_tsp import ACO_TSP


# ---------------------------------------------------------------------------
# Global figure defaults for paper-quality output
# ---------------------------------------------------------------------------

_FONT_FAMILY = "sans-serif"
_LABEL_SIZE = 11
_DPI = 200

COLOR_BASE = "tab:blue"
COLOR_IGPU = "tab:red"


def _apply_paper_rc() -> None:
    """Set matplotlib RC params for paper-quality figures (sans-serif, 11pt)."""
    matplotlib.rcParams.update({
        "font.family": _FONT_FAMILY,
        "axes.labelsize": _LABEL_SIZE,
        "axes.titlesize": _LABEL_SIZE + 1,
        "xtick.labelsize": _LABEL_SIZE - 1,
        "ytick.labelsize": _LABEL_SIZE - 1,
        "legend.fontsize": _LABEL_SIZE - 1,
    })


def _save_fig(fig: plt.Figure, save_path: str) -> None:
    """Save figure as both PNG and PDF at paper DPI."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    base, _ = os.path.splitext(save_path)
    fig.savefig(base + ".png", dpi=_DPI, bbox_inches="tight")
    fig.savefig(base + ".pdf", dpi=_DPI, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Existing functions (unchanged)
# ---------------------------------------------------------------------------


def generate_points(n=20):
    return [(random.randint(0, 1000), random.randint(0, 1000)) for _ in range(n)]


def parameter_analysis():
    ants_list = [10, 20, 30, 50]
    results = []

    points = generate_points(20)

    for ants in ants_list:
        aco = ACO_TSP(points, n_ants=ants, n_iter=100)
        _, best_cost, _ = aco.run()
        results.append(best_cost)

    plt.figure()
    plt.plot(ants_list, results, marker='o')
    plt.title("Effect of Number of Ants on Cost")
    plt.xlabel("Number of Ants")
    plt.ylabel("Final Cost")
    plt.grid()
    plt.savefig("parameter_analysis.png")
    plt.close()


def stability_analysis():
    runs = 5
    costs = []

    points = generate_points(20)

    for _ in range(runs):
        aco = ACO_TSP(points, n_ants=30, n_iter=100)
        _, best_cost, _ = aco.run()
        costs.append(best_cost)

    avg_cost = sum(costs) / len(costs)

    print("\nStability Analysis:")
    print(f"Costs: {costs}")
    print(f"Average Cost: {avg_cost:.2f}")

    plt.figure()
    plt.plot(costs, marker='o')
    plt.title("Stability Across Runs")
    plt.xlabel("Run")
    plt.ylabel("Best Cost")
    plt.grid()
    plt.savefig("stability_analysis.png")
    plt.close()


# ---------------------------------------------------------------------------
# New paper-quality analysis functions
# ---------------------------------------------------------------------------


def compare_kl_trajectories(
    igpu_history: List[Dict[str, Any]],
    baseline_history: List[Dict[str, Any]],
    save_path: str = "kl_trajectories.png",
) -> None:
    """Two-line plot of per-iteration KL divergence for IGPU vs baseline.

    Automatically switches to log-y when the KL range spans 3+ orders of
    magnitude.  Tight axis limits, paper-quality styling.
    """
    _apply_paper_rc()

    kl_igpu = [float(h["kl"]) for h in igpu_history]
    kl_base = [float(h["kl"]) for h in baseline_history]

    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    iters_igpu = list(range(1, len(kl_igpu) + 1))
    iters_base = list(range(1, len(kl_base) + 1))

    ax.plot(iters_base, kl_base, color=COLOR_BASE, linewidth=1.4,
            label="Baseline", alpha=0.85)
    ax.plot(iters_igpu, kl_igpu, color=COLOR_IGPU, linewidth=1.4,
            label="IGPU", alpha=0.85)

    # Auto log-y when KL spans ≥3 orders of magnitude
    all_kl = [v for v in kl_igpu + kl_base if v > 0]
    if all_kl:
        kl_min, kl_max = min(all_kl), max(all_kl)
        if kl_max / max(kl_min, 1e-30) >= 1e3:
            ax.set_yscale("log")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\mathrm{KL}(P_{\mathrm{new}} \| P_{\mathrm{old}})$")
    ax.set_title("Per-Iteration KL Divergence")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.set_xlim(left=1)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    _save_fig(fig, save_path)
    plt.close(fig)


def oscillation_heatmap(
    igpu_history: List[Dict[str, Any]],
    baseline_history: List[Dict[str, Any]],
    region_assignments: List[int],
    save_path: str = "oscillation_heatmap.png",
    window: int = 10,
) -> None:
    """Heatmap of per-region oscillation rate over a sliding window.

    One subplot per algorithm.  Rows = regions, columns = iterations,
    color = oscillation rate in [0, 1].
    """
    _apply_paper_rc()

    region_arr = np.asarray(region_assignments, dtype=int)
    regions = np.unique(region_arr)
    n_regions = len(regions)

    def _compute_heatmap(history: List[Dict[str, Any]]) -> np.ndarray:
        """Return (n_regions, n_iters) array of sliding-window oscillation."""
        n_iters = len(history)
        hm = np.zeros((n_regions, n_iters), dtype=float)

        for r_idx, r in enumerate(regions):
            cols = np.flatnonzero(region_arr == r)
            if cols.size == 0:
                continue

            # Pre-compute per-iteration argmax for this region
            argmaxes: List[int] = []
            for entry in history:
                P = np.asarray(entry["P_new"])
                incoming = P[:, cols].sum(axis=0)
                argmaxes.append(int(cols[np.argmax(incoming)]))

            for t in range(n_iters):
                start = max(0, t - window + 1)
                seg = argmaxes[start:t + 1]
                if len(seg) < 2:
                    hm[r_idx, t] = 0.0
                else:
                    flips = sum(
                        1 for a, b in zip(seg[:-1], seg[1:]) if a != b
                    )
                    hm[r_idx, t] = flips / (len(seg) - 1)

        return hm

    hm_igpu = _compute_heatmap(igpu_history)
    hm_base = _compute_heatmap(baseline_history)

    # Shared color limits
    vmin = 0.0
    vmax = max(hm_igpu.max(), hm_base.max(), 0.01)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.0, 3.2), sharey=True,
    )

    im1 = ax1.imshow(
        hm_base, aspect="auto", origin="lower", cmap="YlOrRd",
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax1.set_title("Baseline")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Region")
    ax1.set_yticks(range(n_regions))
    ax1.set_yticklabels([f"R{r}" for r in regions])

    im2 = ax2.imshow(
        hm_igpu, aspect="auto", origin="lower", cmap="YlOrRd",
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax2.set_title("IGPU")
    ax2.set_xlabel("Iteration")

    fig.colorbar(im2, ax=[ax1, ax2], label="Oscillation rate", shrink=0.85)

    fig.tight_layout()
    _save_fig(fig, save_path)
    plt.close(fig)


def density_sweep_plot(
    csv_path: str = "results/e1_density.csv",
    save_path: str = "density_sweep.png",
) -> None:
    """Plot final_cost mean ± std for both algos over num_entry (log-x).

    Reads the CSV produced by experiments.py (E1).  The density column
    maps to num_entry (entry points per region).
    """
    _apply_paper_rc()

    # Read CSV
    rows: List[Dict[str, Any]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "num_entry": int(row["density"]),
                "algo": row["algo"],
                "final_cost": float(row["final_cost"]),
            })

    densities = sorted({r["num_entry"] for r in rows})

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for algo_name, color, marker in [
        ("baseline", COLOR_BASE, "s"),
        ("igpu", COLOR_IGPU, "o"),
    ]:
        means: List[float] = []
        stds: List[float] = []
        for d in densities:
            costs = [
                r["final_cost"] for r in rows
                if r["num_entry"] == d and r["algo"] == algo_name
            ]
            means.append(float(np.mean(costs)))
            stds.append(float(np.std(costs)))

        m = np.asarray(means)
        s = np.asarray(stds)
        ax.plot(
            densities, m, marker=marker, color=color, linewidth=1.4,
            label=algo_name.upper() if algo_name == "igpu" else algo_name.capitalize(),
            markersize=5,
        )
        ax.fill_between(densities, m - s, m + s, alpha=0.18, color=color)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Entry points per region")
    ax.set_ylabel("Final tour cost")
    ax.set_title("Density Sweep (mean ± std)")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    _save_fig(fig, save_path)
    plt.close(fig)


def paper_figure_panel(
    results_dir: str = "results",
    save_path: str = "paper_panel.png",
) -> None:
    """2×2 hero figure for the paper.

    (a) density sweep — mean ± std of final cost over num_entry
    (b) KL trajectory — per-iteration KL for both algos
    (c) oscillation box — box plot of oscillation rate
    (d) final-cost box — box plot of final tour cost

    Data is loaded from pickle files in `results_dir` produced by
    experiments.py (E1, E3, E4, E5/E6).
    """
    import pickle

    _apply_paper_rc()

    def _load(name: str) -> Any:
        path = os.path.join(results_dir, name)
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Load data ──────────────────────────────────────────────────
    # (a) density sweep — from CSV
    csv_path = os.path.join(results_dir, "e1_density.csv")
    csv_rows: List[Dict[str, Any]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_rows.append({
                "num_entry": int(row["density"]),
                "algo": row["algo"],
                "final_cost": float(row["final_cost"]),
            })
    densities = sorted({r["num_entry"] for r in csv_rows})

    # (b) KL trajectory
    kl_data = _load("e3_kl.pkl")

    # (c) oscillation
    osc_data = _load("e4_oscillation.pkl")

    # (d) final cost stability
    e56_data = _load("e56_data.pkl")

    # ── Build 2×2 panel ───────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # --- (a) Density sweep ---
    for algo_name, color, marker in [
        ("baseline", COLOR_BASE, "s"),
        ("igpu", COLOR_IGPU, "o"),
    ]:
        means: List[float] = []
        stds: List[float] = []
        for d in densities:
            costs = [
                r["final_cost"] for r in csv_rows
                if r["num_entry"] == d and r["algo"] == algo_name
            ]
            means.append(float(np.mean(costs)))
            stds.append(float(np.std(costs)))
        m = np.asarray(means)
        s = np.asarray(stds)
        ax_a.plot(
            densities, m, marker=marker, color=color, linewidth=1.3,
            label=algo_name.upper() if algo_name == "igpu" else algo_name.capitalize(),
            markersize=4,
        )
        ax_a.fill_between(densities, m - s, m + s, alpha=0.18, color=color)

    ax_a.set_xscale("log", base=2)
    ax_a.set_xlabel("Entry points per region")
    ax_a.set_ylabel("Final tour cost")
    ax_a.set_title("(a) Density Sweep")
    ax_a.legend(frameon=True, fancybox=False, edgecolor="0.7", loc="best")
    ax_a.grid(True, linewidth=0.3, alpha=0.5)

    # --- (b) KL trajectory ---
    kl_base = kl_data["baseline_kl"]
    kl_igpu = kl_data["igpu_kl"]

    ax_b.plot(
        range(1, len(kl_base) + 1), kl_base, color=COLOR_BASE,
        linewidth=1.3, label="Baseline", alpha=0.85,
    )
    ax_b.plot(
        range(1, len(kl_igpu) + 1), kl_igpu, color=COLOR_IGPU,
        linewidth=1.3, label="IGPU", alpha=0.85,
    )
    all_kl = [v for v in kl_base + kl_igpu if v > 0]
    if all_kl:
        kl_min, kl_max = min(all_kl), max(all_kl)
        if kl_max / max(kl_min, 1e-30) >= 1e3:
            ax_b.set_yscale("log")
    ax_b.set_xlabel("Iteration")
    ax_b.set_ylabel(r"$\mathrm{KL}$")
    ax_b.set_title("(b) KL Trajectory")
    ax_b.legend(frameon=True, fancybox=False, edgecolor="0.7", loc="best")
    ax_b.set_xlim(left=1)
    ax_b.grid(True, linewidth=0.3, alpha=0.5)

    # --- (c) Oscillation box ---
    bp_c = ax_c.boxplot(
        [osc_data["baseline"], osc_data["igpu"]],
        tick_labels=["Baseline", "IGPU"],
        patch_artist=True,
        widths=0.45,
    )
    for patch, color in zip(bp_c["boxes"], [COLOR_BASE, COLOR_IGPU]):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    ax_c.set_ylabel("Argmax flip rate")
    ax_c.set_title("(c) Oscillation Rate")
    ax_c.grid(True, axis="y", linewidth=0.3, alpha=0.5)

    # --- (d) Final-cost box ---
    bp_d = ax_d.boxplot(
        [e56_data["baseline"]["final_cost"], e56_data["igpu"]["final_cost"]],
        tick_labels=["Baseline", "IGPU"],
        patch_artist=True,
        widths=0.45,
    )
    for patch, color in zip(bp_d["boxes"], [COLOR_BASE, COLOR_IGPU]):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    ax_d.set_ylabel("Final tour cost")
    ax_d.set_title("(d) Cost Stability (20 seeds)")
    ax_d.grid(True, axis="y", linewidth=0.3, alpha=0.5)

    fig.tight_layout(h_pad=2.5, w_pad=2.0)
    _save_fig(fig, save_path)
    plt.close(fig)