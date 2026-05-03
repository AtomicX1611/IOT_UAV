"""Run the seven IGPU paper experiments and produce plots + CSV/pickle outputs.

Usage:
    python experiments.py --experiment e1 --output_dir results/
    python experiments.py --experiment all
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import pickle
import time
from typing import Any, Callable, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aco_multi_entry_baseline import ACO_MultiEntry_Baseline
from aco_multi_entry_igpu import ACO_MultiEntry_IGPU
from energy_model import UAVEnergyModel
from metrics import (
    iterations_to_converge,
    kl_trajectory,
    oscillation_rate,
    summarize_run,
)


FIGSIZE: Tuple[float, float] = (6.0, 4.0)
COLOR_BASE: str = "tab:blue"
COLOR_IGPU: str = "tab:red"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_random_regions(
    n_regions: int,
    num_entry: int,
    seed: int,
    area: float = 1000.0,
    radius: float = 70.0,
) -> List[Dict[str, Any]]:
    """Place `n_regions` non-overlapping circles of radius `radius` in [0, area]^2.

    `num_entry` is unused here (regions are independent of entry-point density);
    it remains in the signature as a logical caller-side grouping.
    """
    del num_entry
    rng = np.random.default_rng(seed)
    centers: List[Tuple[float, float]] = []
    min_sep = 2.0 * radius + 5.0
    max_attempts = 20000
    attempts = 0
    while len(centers) < n_regions:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not place {n_regions} regions of r={radius} in [0,{area}]^2"
            )
        cx = float(rng.uniform(radius, area - radius))
        cy = float(rng.uniform(radius, area - radius))
        if all((cx - x) ** 2 + (cy - y) ** 2 >= min_sep ** 2 for x, y in centers):
            centers.append((cx, cy))
    return [{"center": c, "radius": radius} for c in centers]


def run_single(
    algo_class: Callable[..., Any],
    regions: List[Dict[str, Any]],
    num_entry: int,
    n_ants: int = 30,
    n_iter: int = 150,
    seed: int = 0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run one algo instance silently and return a result dict."""
    algo = algo_class(
        regions=regions,
        num_entry=num_entry,
        n_ants=n_ants,
        n_iter=n_iter,
        seed=seed,
        **kwargs,
    )
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        best_path, best_cost, history = algo.run()
    cost_history = [float(h["best_cost"]) for h in history]
    return {
        "best_cost": float(best_cost),
        "best_path": list(best_path),
        "cost_history": cost_history,
        "history": history,
        "region_assignments": list(algo.node_to_region),
        "nodes": list(algo.nodes),
    }


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _pkl_path(output_dir: str, name: str) -> str:
    return os.path.join(output_dir, name)


def _save_pickle(path: str, data: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# E1 + E2: density sweep (shared data collection)
# ---------------------------------------------------------------------------


def collect_e1_rows(output_dir: str) -> List[Dict[str, Any]]:
    pkl = _pkl_path(output_dir, "e1_density.pkl")
    if os.path.exists(pkl):
        return _load_pickle(pkl)

    densities = [8, 16, 32, 64, 128]
    seeds = [0, 1, 2, 3, 4]
    n_regions = 8
    n_iter = 150
    n_ants = 30

    rows: List[Dict[str, Any]] = []
    for density in densities:
        for seed in seeds:
            regions = build_random_regions(n_regions, density, seed=seed)
            for algo_name, algo_class in (
                ("baseline", ACO_MultiEntry_Baseline),
                ("igpu", ACO_MultiEntry_IGPU),
            ):
                t0 = time.time()
                res = run_single(
                    algo_class, regions, density, n_ants, n_iter, seed,
                )
                summary = summarize_run(
                    res["history"], res["cost_history"], res["region_assignments"],
                )
                row: Dict[str, Any] = {
                    "density": density,
                    "algo": algo_name,
                    "seed": seed,
                    "final_cost": summary["final_cost"],
                    "iter_to_converge": summary["iter_to_converge"],
                    "mean_kl": summary["mean_kl"],
                    "final_oscillation": summary["final_oscillation"],
                    "cost_history": res["cost_history"],
                    "kl_trajectory": kl_trajectory(res["history"]),
                }
                rows.append(row)
                print(
                    f"E1 density={density:3d} seed={seed} algo={algo_name:8s} "
                    f"final={row['final_cost']:8.2f} "
                    f"({time.time() - t0:5.1f}s)"
                )

    _save_pickle(pkl, rows)

    csv_path = os.path.join(output_dir, "e1_density.csv")
    fields = [
        "density", "algo", "seed", "final_cost",
        "iter_to_converge", "mean_kl", "final_oscillation",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fields})

    return rows


def plot_e1(output_dir: str, rows: List[Dict[str, Any]]) -> None:
    densities = sorted({r["density"] for r in rows})
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for algo_name, color in (("baseline", COLOR_BASE), ("igpu", COLOR_IGPU)):
        means: List[float] = []
        stds: List[float] = []
        for d in densities:
            costs = [r["final_cost"] for r in rows
                     if r["density"] == d and r["algo"] == algo_name]
            means.append(float(np.mean(costs)))
            stds.append(float(np.std(costs)))
        m = np.asarray(means)
        s = np.asarray(stds)
        ax.plot(densities, m, marker="o", color=color, label=algo_name)
        ax.fill_between(densities, m - s, m + s, alpha=0.2, color=color)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Entry points per region")
    ax.set_ylabel("Final tour cost")
    ax.set_title("E1: density sweep (mean ± std over 5 seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "e1_density.png"), dpi=120)
    plt.close(fig)


def plot_e2(output_dir: str, rows: List[Dict[str, Any]]) -> None:
    densities = sorted({r["density"] for r in rows})
    base_means: List[float] = []
    igpu_means: List[float] = []
    for d in densities:
        b = [r["iter_to_converge"] for r in rows
             if r["density"] == d and r["algo"] == "baseline"]
        i = [r["iter_to_converge"] for r in rows
             if r["density"] == d and r["algo"] == "igpu"]
        base_means.append(float(np.mean(b)))
        igpu_means.append(float(np.mean(i)))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(densities))
    width = 0.4
    ax.bar(x - width / 2, base_means, width, color=COLOR_BASE, label="baseline")
    ax.bar(x + width / 2, igpu_means, width, color=COLOR_IGPU, label="IGPU")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in densities])
    ax.set_xlabel("Entry points per region")
    ax.set_ylabel("Iterations to converge (5% band)")
    ax.set_title("E2: convergence speed across density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "e2_convergence.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# E3: KL trajectory
# ---------------------------------------------------------------------------


def experiment_e3(output_dir: str) -> None:
    n_regions = 8
    num_entry = 64
    seed = 0
    regions = build_random_regions(n_regions, num_entry, seed=seed)

    res_b = run_single(ACO_MultiEntry_Baseline, regions, num_entry, seed=seed)
    res_i = run_single(ACO_MultiEntry_IGPU, regions, num_entry, seed=seed)
    kl_b = kl_trajectory(res_b["history"])
    kl_i = kl_trajectory(res_i["history"])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(kl_b, color=COLOR_BASE, label="baseline")
    ax.plot(kl_i, color=COLOR_IGPU, label="IGPU")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\mathrm{KL}(P_{new}\,\|\,P_{old})$")
    ax.set_title("E3: per-iteration policy KL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "e3_kl.png"), dpi=120)
    plt.close(fig)

    _save_pickle(_pkl_path(output_dir, "e3_kl.pkl"), {
        "baseline_kl": kl_b,
        "igpu_kl": kl_i,
        "baseline_cost_history": res_b["cost_history"],
        "igpu_cost_history": res_i["cost_history"],
    })


# ---------------------------------------------------------------------------
# E4: oscillation rate box plot
# ---------------------------------------------------------------------------


def experiment_e4(output_dir: str) -> None:
    n_regions = 8
    num_entry = 64
    seeds = list(range(10))
    osc_b: List[float] = []
    osc_i: List[float] = []
    for seed in seeds:
        regions = build_random_regions(n_regions, num_entry, seed=seed)
        for algo_class, lst in (
            (ACO_MultiEntry_Baseline, osc_b),
            (ACO_MultiEntry_IGPU, osc_i),
        ):
            t0 = time.time()
            res = run_single(algo_class, regions, num_entry, seed=seed)
            osc = oscillation_rate(res["history"], res["region_assignments"])
            lst.append(float(osc))
            print(f"E4 seed={seed} {algo_class.__name__:25s} osc={osc:.3f} ({time.time() - t0:5.1f}s)")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.boxplot([osc_b, osc_i], tick_labels=["baseline", "IGPU"])
    ax.set_ylabel("Argmax flip rate (last 20 iters)")
    ax.set_title("E4: oscillation rate (10 seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "e4_oscillation.png"), dpi=120)
    plt.close(fig)

    _save_pickle(_pkl_path(output_dir, "e4_oscillation.pkl"), {
        "baseline": osc_b,
        "igpu": osc_i,
        "seeds": seeds,
    })


# ---------------------------------------------------------------------------
# E5 + E6: shared 20-seed runs (cost stability + UAV energy)
# ---------------------------------------------------------------------------


def collect_e56_data(output_dir: str) -> Dict[str, Any]:
    pkl = _pkl_path(output_dir, "e56_data.pkl")
    if os.path.exists(pkl):
        return _load_pickle(pkl)

    n_regions = 8
    num_entry = 32
    seeds = list(range(20))

    energy_model = UAVEnergyModel()
    base: Dict[str, List[Any]] = {"final_cost": [], "energy": []}
    igpu: Dict[str, List[Any]] = {"final_cost": [], "energy": []}

    for seed in seeds:
        regions = build_random_regions(n_regions, num_entry, seed=seed)
        for algo_class, bucket in (
            (ACO_MultiEntry_Baseline, base),
            (ACO_MultiEntry_IGPU, igpu),
        ):
            t0 = time.time()
            res = run_single(algo_class, regions, num_entry, seed=seed)
            energy = energy_model.total_path_energy(res["nodes"], res["best_path"])
            bucket["final_cost"].append(float(res["best_cost"]))
            bucket["energy"].append(float(energy))
            print(
                f"E5/E6 seed={seed:2d} {algo_class.__name__:25s} "
                f"cost={res['best_cost']:8.2f} energy={energy:10.1f} J "
                f"({time.time() - t0:5.1f}s)"
            )

    data = {"seeds": seeds, "baseline": base, "igpu": igpu}
    _save_pickle(pkl, data)
    return data


def plot_e5(output_dir: str, data: Dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.boxplot(
        [data["baseline"]["final_cost"], data["igpu"]["final_cost"]],
        tick_labels=["baseline", "IGPU"],
    )
    ax.set_ylabel("Final tour cost")
    ax.set_title("E5: stability across 20 seeds")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "e5_stability.png"), dpi=120)
    plt.close(fig)


def plot_e6(output_dir: str, data: Dict[str, Any]) -> None:
    base_e = data["baseline"]["energy"]
    igpu_e = data["igpu"]["energy"]
    means = [float(np.mean(base_e)), float(np.mean(igpu_e))]
    stds = [float(np.std(base_e)), float(np.std(igpu_e))]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(2)
    ax.bar(x, means, yerr=stds, capsize=5,
           color=[COLOR_BASE, COLOR_IGPU])
    ax.set_xticks(x)
    ax.set_xticklabels(["baseline", "IGPU"])
    ax.set_ylabel("UAV total energy [J]")
    ax.set_title("E6: total path energy (mean ± std, 20 seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "e6_energy.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# E7: lambda sweep
# ---------------------------------------------------------------------------


def experiment_e7(output_dir: str) -> None:
    """Lambda-damping sensitivity study.

    Uses a smaller problem (num_entry=16 → 128 nodes) and higher gamma so
    that the natural-gradient signal is large enough for lambda to matter.
    Results are averaged over 5 seeds for statistical robustness.
    """
    n_regions = 8
    num_entry = 16
    seeds = [0, 1, 2, 3, 4]
    gamma_e7 = 0.15  # amplified for sensitivity study

    regions_per_seed = {s: build_random_regions(n_regions, num_entry, seed=s) for s in seeds}

    lambdas: List[float] = [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0]
    igpu_means: List[float] = []
    igpu_stds: List[float] = []

    for lam in lambdas:
        costs_lam: List[float] = []
        for s in seeds:
            t0 = time.time()
            res = run_single(
                ACO_MultiEntry_IGPU, regions_per_seed[s], num_entry,
                n_ants=30, n_iter=150, seed=s,
                lambda_damping=lam, gamma=gamma_e7,
            )
            costs_lam.append(float(res["best_cost"]))
            print(
                f"E7 lambda={lam:8.0e} seed={s} "
                f"cost={res['best_cost']:8.2f} ({time.time() - t0:5.1f}s)"
            )
        igpu_means.append(float(np.mean(costs_lam)))
        igpu_stds.append(float(np.std(costs_lam)))

    # Baseline reference averaged over same seeds
    baseline_costs: List[float] = []
    for s in seeds:
        res_b = run_single(
            ACO_MultiEntry_Baseline, regions_per_seed[s], num_entry, seed=s,
        )
        baseline_costs.append(float(res_b["best_cost"]))
    baseline_mean = float(np.mean(baseline_costs))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    m = np.asarray(igpu_means)
    st = np.asarray(igpu_stds)
    ax.plot(lambdas, m, marker="o", color=COLOR_IGPU, label="IGPU")
    ax.fill_between(lambdas, m - st, m + st, alpha=0.2, color=COLOR_IGPU)
    ax.axhline(baseline_mean, color=COLOR_BASE, linestyle="--", label="baseline ref")
    ax.set_xscale("log")
    ax.set_xlabel(r"Fisher damping $\lambda$")
    ax.set_ylabel("Final tour cost")
    ax.set_title(r"E7: $\lambda$ sweep (mean ± std, 5 seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "e7_lambda.png"), dpi=120)
    plt.close(fig)

    _save_pickle(_pkl_path(output_dir, "e7_lambda.pkl"), {
        "lambdas": lambdas,
        "igpu_means": igpu_means,
        "igpu_stds": igpu_stds,
        "baseline_mean": baseline_mean,
    })


# ---------------------------------------------------------------------------
# E8: extended density sweep
# ---------------------------------------------------------------------------


def experiment_e8(output_dir: str) -> None:
    """Extended density sweep: finer grid, more seeds, longer runs."""
    densities = [32, 64, 128, 256, 512]
    seeds = list(range(8))
    n_regions = 8
    n_iter = 200
    n_ants = 30

    rows: List[Dict[str, Any]] = []
    for density in densities:
        for seed in seeds:
            regions = build_random_regions(n_regions, density, seed=seed)
            for algo_name, algo_class in (
                ("baseline", ACO_MultiEntry_Baseline),
                ("igpu", ACO_MultiEntry_IGPU),
            ):
                t0 = time.time()
                res = run_single(
                    algo_class, regions, density, n_ants, n_iter, seed,
                )
                cost_hist = res["cost_history"]
                osc = oscillation_rate(
                    res["history"], res["region_assignments"],
                )
                itc = iterations_to_converge(cost_hist)
                ent_final = float(res["history"][-1].get("mean_entropy", 0.0))

                row: Dict[str, Any] = {
                    "density": density,
                    "algo": algo_name,
                    "seed": seed,
                    "final_cost": float(res["best_cost"]),
                    "iter_to_converge": int(itc),
                    "final_oscillation": float(osc),
                    "mean_entropy_final": ent_final,
                }
                rows.append(row)
                elapsed = time.time() - t0
                print(
                    f"E8 d={density:3d} s={seed} {algo_name:8s} "
                    f"cost={row['final_cost']:8.2f} itc={itc:3d} "
                    f"osc={osc:.3f} ent={ent_final:.4f} ({elapsed:5.1f}s)"
                )

    # Save CSV
    csv_path = os.path.join(output_dir, "e8_extended.csv")
    fields = [
        "density", "algo", "seed", "final_cost",
        "iter_to_converge", "final_oscillation", "mean_entropy_final",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fields})

    # Plot: two side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    for algo_name, color in (("baseline", COLOR_BASE), ("igpu", COLOR_IGPU)):
        # -- Left: iter_to_converge
        means_itc: List[float] = []
        stds_itc: List[float] = []
        # -- Right: final_oscillation
        means_osc: List[float] = []
        stds_osc: List[float] = []
        for d in densities:
            vals_itc = [r["iter_to_converge"] for r in rows
                        if r["density"] == d and r["algo"] == algo_name]
            vals_osc = [r["final_oscillation"] for r in rows
                        if r["density"] == d and r["algo"] == algo_name]
            means_itc.append(float(np.mean(vals_itc)))
            stds_itc.append(float(np.std(vals_itc)))
            means_osc.append(float(np.mean(vals_osc)))
            stds_osc.append(float(np.std(vals_osc)))

        ax1.errorbar(
            densities, means_itc, yerr=stds_itc,
            marker="o", color=color, label=algo_name, capsize=3,
        )
        ax2.errorbar(
            densities, means_osc, yerr=stds_osc,
            marker="o", color=color, label=algo_name, capsize=3,
        )

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Entry points per region")
    ax1.set_ylabel("Iterations to converge (5% band)")
    ax1.set_title("E8a: convergence speed")
    ax1.legend()

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Entry points per region")
    ax2.set_ylabel("Oscillation rate (last 20 iters)")
    ax2.set_title("E8b: policy oscillation")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "e8_extended.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "all"],
        required=True,
    )
    parser.add_argument("--output_dir", default="results/")
    args = parser.parse_args()

    _ensure_dir(args.output_dir)

    selected = (
        ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"]
        if args.experiment == "all"
        else [args.experiment]
    )

    if "e1" in selected or "e2" in selected:
        rows = collect_e1_rows(args.output_dir)
        if "e1" in selected:
            plot_e1(args.output_dir, rows)
        if "e2" in selected:
            plot_e2(args.output_dir, rows)

    if "e3" in selected:
        experiment_e3(args.output_dir)

    if "e4" in selected:
        experiment_e4(args.output_dir)

    if "e5" in selected or "e6" in selected:
        data = collect_e56_data(args.output_dir)
        if "e5" in selected:
            plot_e5(args.output_dir, data)
        if "e6" in selected:
            plot_e6(args.output_dir, data)

    if "e7" in selected:
        experiment_e7(args.output_dir)

    if "e8" in selected:
        experiment_e8(args.output_dir)

    print(f"\nDone. Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
