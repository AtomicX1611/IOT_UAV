"""Unified CLI entry point for the IOT_UAV project.

Usage examples:
    python main.py --mode demo
    python main.py --mode igpu --num_regions 6 --n_iter 100
    python main.py --mode compare --seed 7
    python main.py --mode experiments --experiment e3
    python main.py --mode smoke
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from plotly_aco_3d import SimulationConfig, run_demo
from aco_multi_entry_igpu import ACO_MultiEntry_IGPU
from aco_multi_entry_baseline import ACO_MultiEntry_Baseline
from metrics import summarize_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_regions(
    n_regions: int, seed: int, area: float = 1000.0, radius: float = 70.0,
) -> List[Dict[str, Any]]:
    """Place non-overlapping circles for multi-entry demos."""
    rng = np.random.default_rng(seed)
    centers: List[Tuple[float, float]] = []
    min_sep = 2.0 * radius + 5.0
    attempts = 0
    while len(centers) < n_regions:
        attempts += 1
        if attempts > 20_000:
            raise RuntimeError(
                f"Could not place {n_regions} regions of r={radius} in [0,{area}]^2"
            )
        cx = float(rng.uniform(radius, area - radius))
        cy = float(rng.uniform(radius, area - radius))
        if all((cx - x) ** 2 + (cy - y) ** 2 >= min_sep ** 2 for x, y in centers):
            centers.append((cx, cy))
    return [{"center": c, "radius": radius} for c in centers]


def _run_multi_entry(
    algo_class: type,
    regions: List[Dict[str, Any]],
    num_entry: int,
    n_ants: int,
    n_iter: int,
    seed: int,
) -> Dict[str, Any]:
    """Run a multi-entry algo, return unified result dict."""
    algo = algo_class(
        regions=regions,
        num_entry=num_entry,
        n_ants=n_ants,
        n_iter=n_iter,
        seed=seed,
    )
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


def _print_summary(label: str, summary: Dict[str, float]) -> None:
    """Pretty-print a summarize_run dict."""
    print(f"\n{'-' * 50}")
    print(f"  {label}")
    print(f"{'-' * 50}")
    for key, val in summary.items():
        if isinstance(val, float) and (math.isnan(val) or abs(val) > 1e6):
            print(f"  {key:28s}  {val:.4e}")
        elif isinstance(val, float):
            print(f"  {key:28s}  {val:.4f}")
        else:
            print(f"  {key:28s}  {val}")
    print(f"{'-' * 50}")


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------


def mode_demo(args: argparse.Namespace) -> None:
    """Existing 3D Plotly TSP demo — preserved exactly."""
    config = SimulationConfig(
        n_cities=24,
        n_ants=36,
        n_iter=45,
        max_ants_visualized=8,
        ant_interp_steps=4,
        top_pheromone_edges=45,
        save_html_path="aco_3d_simulation.html",
    )
    run_demo(config)


def mode_baseline(args: argparse.Namespace) -> None:
    """Run multi-entry baseline, print summary, save best path."""
    regions = _build_regions(args.num_regions, seed=args.seed)
    res = _run_multi_entry(
        ACO_MultiEntry_Baseline, regions,
        args.num_entry, args.n_ants, args.n_iter, args.seed,
    )
    summary = summarize_run(
        res["history"], res["cost_history"], res["region_assignments"],
    )
    _print_summary("Baseline ACO (Multi-Entry)", summary)
    print(f"  Best path ({len(res['best_path'])} nodes): {res['best_path']}")


def mode_igpu(args: argparse.Namespace) -> None:
    """Run multi-entry IGPU, print summary, save best path."""
    regions = _build_regions(args.num_regions, seed=args.seed)
    res = _run_multi_entry(
        ACO_MultiEntry_IGPU, regions,
        args.num_entry, args.n_ants, args.n_iter, args.seed,
    )
    summary = summarize_run(
        res["history"], res["cost_history"], res["region_assignments"],
    )
    _print_summary("IGPU ACO (Multi-Entry)", summary)
    print(f"  Best path ({len(res['best_path'])} nodes): {res['best_path']}")


def mode_compare(args: argparse.Namespace) -> None:
    """Run both algos with the same seed, print side-by-side summary table."""
    regions = _build_regions(args.num_regions, seed=args.seed)

    print("Running Baseline …")
    t0 = time.time()
    res_b = _run_multi_entry(
        ACO_MultiEntry_Baseline, regions,
        args.num_entry, args.n_ants, args.n_iter, args.seed,
    )
    time_b = time.time() - t0
    sum_b = summarize_run(
        res_b["history"], res_b["cost_history"], res_b["region_assignments"],
    )

    print("\nRunning IGPU …")
    t0 = time.time()
    res_i = _run_multi_entry(
        ACO_MultiEntry_IGPU, regions,
        args.num_entry, args.n_ants, args.n_iter, args.seed,
    )
    time_i = time.time() - t0
    sum_i = summarize_run(
        res_i["history"], res_i["cost_history"], res_i["region_assignments"],
    )

    # Side-by-side table
    print(f"\n{'=' * 62}")
    print(f"  {'Metric':28s} {'Baseline':>14s} {'IGPU':>14s}")
    print(f"{'=' * 62}")
    all_keys = list(sum_b.keys())
    for key in all_keys:
        vb = sum_b[key]
        vi = sum_i[key]
        if isinstance(vb, float):
            print(f"  {key:28s} {vb:14.4f} {vi:14.4f}")
        else:
            print(f"  {key:28s} {vb!s:>14s} {vi!s:>14s}")
    print(f"  {'wall_time_s':28s} {time_b:14.2f} {time_i:14.2f}")
    print(f"{'=' * 62}")


def mode_experiments(args: argparse.Namespace) -> None:
    """Delegate to experiments.py dispatcher."""
    saved_argv = sys.argv[:]
    try:
        sys.argv = [
            "experiments.py",
            "--experiment", args.experiment,
            "--output_dir", "results/",
        ]
        import experiments
        experiments.main()
    finally:
        sys.argv = saved_argv


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def smoke_test() -> None:
    """Run a tiny config for each mode and assert no exceptions.

    Uses 4 regions, 8 entries, 5 ants, 10 iters — finishes in seconds.
    """
    print("=" * 60)
    print("  SMOKE TEST — tiny configs, asserting no exceptions")
    print("=" * 60)

    errors: List[str] = []

    # --- demo (TSP, no browser window — just build figure) ---
    print("\n[1/4] smoke: demo (TSP) …")
    try:
        config = SimulationConfig(
            n_cities=6,
            n_ants=5,
            n_iter=3,
            max_ants_visualized=2,
            ant_interp_steps=1,
            top_pheromone_edges=5,
            save_html_path=None,  # don't write file
        )
        # Suppress plotly .show() — capture silently
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            fig = run_demo(config)
        assert fig is not None, "run_demo returned None"
        print("  [OK] demo passed")
    except Exception as exc:
        errors.append(f"demo: {exc}")
        print(f"  [FAIL] demo FAILED: {exc}")

    # --- baseline ---
    print("\n[2/4] smoke: baseline …")
    try:
        regions = _build_regions(4, seed=0, radius=30.0)
        res = _run_multi_entry(
            ACO_MultiEntry_Baseline, regions,
            num_entry=8, n_ants=5, n_iter=10, seed=0,
        )
        summary = summarize_run(
            res["history"], res["cost_history"], res["region_assignments"],
        )
        assert res["best_cost"] < float("inf"), "best_cost is inf"
        assert len(summary) > 0, "empty summary"
        print(f"  [OK] baseline passed (cost={res['best_cost']:.2f})")
    except Exception as exc:
        errors.append(f"baseline: {exc}")
        print(f"  [FAIL] baseline FAILED: {exc}")

    # --- igpu ---
    print("\n[3/4] smoke: igpu …")
    try:
        regions = _build_regions(4, seed=0, radius=30.0)
        res = _run_multi_entry(
            ACO_MultiEntry_IGPU, regions,
            num_entry=8, n_ants=5, n_iter=10, seed=0,
        )
        summary = summarize_run(
            res["history"], res["cost_history"], res["region_assignments"],
        )
        assert res["best_cost"] < float("inf"), "best_cost is inf"
        assert len(summary) > 0, "empty summary"
        print(f"  [OK] igpu passed (cost={res['best_cost']:.2f})")
    except Exception as exc:
        errors.append(f"igpu: {exc}")
        print(f"  [FAIL] igpu FAILED: {exc}")

    # --- compare ---
    print("\n[4/4] smoke: compare …")
    try:
        regions = _build_regions(4, seed=0, radius=30.0)
        res_b = _run_multi_entry(
            ACO_MultiEntry_Baseline, regions,
            num_entry=8, n_ants=5, n_iter=10, seed=0,
        )
        res_i = _run_multi_entry(
            ACO_MultiEntry_IGPU, regions,
            num_entry=8, n_ants=5, n_iter=10, seed=0,
        )
        sum_b = summarize_run(
            res_b["history"], res_b["cost_history"], res_b["region_assignments"],
        )
        sum_i = summarize_run(
            res_i["history"], res_i["cost_history"], res_i["region_assignments"],
        )
        assert set(sum_b.keys()) == set(sum_i.keys()), "summary keys mismatch"
        print(f"  [OK] compare passed (baseline={res_b['best_cost']:.2f}, "
              f"igpu={res_i['best_cost']:.2f})")
    except Exception as exc:
        errors.append(f"compare: {exc}")
        print(f"  [FAIL] compare FAILED: {exc}")

    # --- verdict ---
    print(f"\n{'=' * 60}")
    if errors:
        print(f"  SMOKE TEST FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"    • {e}")
        print(f"{'=' * 60}")
        sys.exit(1)
    else:
        print("  SMOKE TEST PASSED — all 4 modes OK")
        print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IOT_UAV: ACO multi-entry path planning with IGPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "baseline", "igpu", "compare", "experiments", "smoke"],
        default="demo",
        help="Execution mode (default: demo)",
    )
    parser.add_argument("--num_regions", type=int, default=8)
    parser.add_argument("--num_entry", type=int, default=32)
    parser.add_argument("--n_ants", type=int, default=30)
    parser.add_argument("--n_iter", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiment", default="all",
        help="Experiment ID for --mode=experiments (default: all)",
    )

    args = parser.parse_args()

    dispatch = {
        "demo": mode_demo,
        "baseline": mode_baseline,
        "igpu": mode_igpu,
        "compare": mode_compare,
        "experiments": mode_experiments,
        "smoke": lambda _: smoke_test(),
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()