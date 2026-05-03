import random
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from aco_tsp import ACO_TSP


Point3D = Tuple[float, float, float]


@dataclass
class SimulationConfig:
    n_cities: int = 24
    n_ants: int = 36
    n_iter: int = 45
    alpha: float = 1.0
    beta: float = 2.5
    evaporation: float = 0.2
    random_seed: Optional[int] = 42
    max_ants_visualized: int = 8
    ant_interp_steps: int = 4
    top_pheromone_edges: int = 45
    save_html_path: Optional[str] = "aco_3d_simulation.html"


def _normalize_axis(values: Sequence[float], target_span: float = 100.0) -> List[float]:
    min_v = min(values)
    max_v = max(values)
    span = max(max_v - min_v, 1e-9)
    return [((v - min_v) / span) * target_span for v in values]


def generate_3d_points(
    points_2d: Optional[Sequence[Tuple[float, float]]] = None,
    n_cities: int = 24,
    z_mode: str = "random",
    normalize: bool = True,
    seed: Optional[int] = None,
) -> List[Point3D]:
    """Create 3D points while preserving compatibility with existing 2D ACO inputs."""
    rng = random.Random(seed)

    if points_2d is None:
        points_2d = [
            (rng.uniform(0.0, 1000.0), rng.uniform(0.0, 1000.0))
            for _ in range(n_cities)
        ]

    points_3d: List[Point3D] = []
    for idx, (x, y) in enumerate(points_2d):
        if z_mode == "plane":
            z = 0.0
        elif z_mode == "layered":
            z = float((idx % 5) * 20)
        else:
            z = rng.uniform(0.0, 1000.0)
        points_3d.append((float(x), float(y), float(z)))

    if not normalize:
        return points_3d

    nx = _normalize_axis([p[0] for p in points_3d])
    ny = _normalize_axis([p[1] for p in points_3d])
    nz = _normalize_axis([p[2] for p in points_3d])
    return [(x, y, z) for x, y, z in zip(nx, ny, nz)]


def run_aco_with_history(
    points_3d: Sequence[Point3D],
    n_ants: int = 30,
    n_iter: int = 100,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation: float = 0.2,
) -> Dict[str, object]:
    """Run ACO while preserving per-iteration ant and pheromone history for animation."""
    points_2d = [(x, y) for x, y, _ in points_3d]
    aco = ACO_TSP(
        points_2d,
        n_ants=n_ants,
        n_iter=n_iter,
        alpha=alpha,
        beta=beta,
        evaporation=evaporation,
    )

    best_path: Optional[List[int]] = None
    best_cost = float("inf")
    cost_history: List[float] = []
    iteration_history: List[Dict[str, object]] = []

    for iteration_idx in range(aco.n_iter):
        all_paths: List[Tuple[List[int], float]] = []

        for _ in range(aco.n_ants):
            path = aco.construct_path()
            cost = aco.path_cost(path)
            all_paths.append((path, cost))

            if cost < best_cost:
                best_cost = cost
                best_path = path[:]

        aco.update_pheromones(all_paths)
        cost_history.append(best_cost)

        top_ants = sorted(all_paths, key=lambda item: item[1])
        iteration_history.append(
            {
                "iteration": iteration_idx + 1,
                "best_path": best_path[:] if best_path is not None else [],
                "best_cost": best_cost,
                "ant_paths": [path[:] for path, _ in top_ants],
                "ant_costs": [cost for _, cost in top_ants],
                "pheromone": [row[:] for row in aco.pheromone],
            }
        )

        iteration_best = top_ants[0][1]
        print(
            f"Iteration {iteration_idx + 1}: "
            f"Iter Best = {iteration_best:.2f}, Global Best = {best_cost:.2f}"
        )

    return {
        "best_path": best_path if best_path is not None else [],
        "best_cost": best_cost,
        "cost_history": cost_history,
        "iteration_history": iteration_history,
    }


def _path_xyz(points_3d: Sequence[Point3D], path: Sequence[int], close_loop: bool = True):
    if not path:
        return [], [], []

    ordered = list(path)
    if close_loop:
        ordered = ordered + [ordered[0]]

    xs = [points_3d[idx][0] for idx in ordered]
    ys = [points_3d[idx][1] for idx in ordered]
    zs = [points_3d[idx][2] for idx in ordered]
    return xs, ys, zs


def _extract_top_pheromone_edges(
    pheromone_matrix: Sequence[Sequence[float]],
    top_k: int,
) -> List[Tuple[int, int, float]]:
    weighted_edges: List[Tuple[int, int, float]] = []
    n = len(pheromone_matrix)

    for i in range(n):
        for j in range(i + 1, n):
            w = float(pheromone_matrix[i][j])
            weighted_edges.append((i, j, w))

    weighted_edges.sort(key=lambda item: item[2], reverse=True)
    if not weighted_edges:
        return []

    return weighted_edges[:top_k]


def _empty_line_trace(name: str = "") -> go.Scatter3d:
    return go.Scatter3d(
        x=[],
        y=[],
        z=[],
        mode="lines",
        line={"width": 1, "color": "rgba(0,0,0,0)"},
        name=name,
        showlegend=False,
        hoverinfo="skip",
    )


def _build_pheromone_edge_traces(
    points_3d: Sequence[Point3D],
    pheromone_matrix: Sequence[Sequence[float]],
    top_k: int,
) -> List[go.Scatter3d]:
    edges = _extract_top_pheromone_edges(pheromone_matrix, top_k=top_k)
    if not edges:
        return [_empty_line_trace("Pheromone") for _ in range(top_k)]

    min_w = min(w for _, _, w in edges)
    max_w = max(w for _, _, w in edges)
    denom = max(max_w - min_w, 1e-9)

    traces: List[go.Scatter3d] = []
    for idx in range(top_k):
        if idx >= len(edges):
            traces.append(_empty_line_trace("Pheromone"))
            continue

        i, j, w = edges[idx]
        intensity = (w - min_w) / denom
        color = sample_colorscale("YlOrRd", [intensity])[0]

        x0, y0, z0 = points_3d[i]
        x1, y1, z1 = points_3d[j]
        traces.append(
            go.Scatter3d(
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                mode="lines",
                line={"width": 1.5 + 4.5 * intensity, "color": color},
                opacity=0.65,
                name="Pheromone" if idx == 0 else "",
                showlegend=idx == 0,
                hovertemplate=(
                    f"Edge: {i}-{j}<br>Pheromone: {w:.3f}<extra></extra>"
                ),
            )
        )

    return traces


def _ant_positions_for_progress(
    points_3d: Sequence[Point3D],
    ant_paths: Sequence[Sequence[int]],
    progress: float,
) -> Tuple[List[float], List[float], List[float], List[str]]:
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    labels: List[str] = []

    for ant_idx, path in enumerate(ant_paths):
        if not path:
            continue

        route = list(path) + [path[0]]
        n_segments = len(route) - 1
        segment_float = min(progress, 0.999999) * n_segments
        segment_idx = int(segment_float)
        local_t = segment_float - segment_idx

        start_city = route[segment_idx]
        end_city = route[segment_idx + 1]

        x0, y0, z0 = points_3d[start_city]
        x1, y1, z1 = points_3d[end_city]

        xs.append(x0 + (x1 - x0) * local_t)
        ys.append(y0 + (y1 - y0) * local_t)
        zs.append(z0 + (z1 - z0) * local_t)
        labels.append(f"Ant {ant_idx}")

    return xs, ys, zs, labels


def create_frames(
    points_3d: Sequence[Point3D],
    iteration_history: Sequence[Dict[str, object]],
    max_ants_visualized: int = 8,
    ant_interp_steps: int = 4,
    top_pheromone_edges: int = 45,
) -> List[go.Frame]:
    """Build high-detail animation frames with smooth ant interpolation per iteration."""
    frames: List[go.Frame] = []
    total_iterations = max(len(iteration_history), 1)

    for iter_idx, state in enumerate(iteration_history):
        best_path = state["best_path"]
        best_cost = state["best_cost"]
        ant_paths_full = state["ant_paths"]
        pheromone = state["pheromone"]

        ant_paths = ant_paths_full[:max_ants_visualized]
        local_steps = max(1, ant_interp_steps * len(points_3d))

        for step_idx in range(local_steps):
            progress = (step_idx + 1) / local_steps
            ant_x, ant_y, ant_z, ant_labels = _ant_positions_for_progress(
                points_3d,
                ant_paths,
                progress,
            )

            path_x, path_y, path_z = _path_xyz(points_3d, best_path, close_loop=True)
            pheromone_traces = _build_pheromone_edge_traces(
                points_3d,
                pheromone,
                top_k=top_pheromone_edges,
            )

            angle = 2.0 * math.pi * ((iter_idx + progress) / total_iterations)
            camera_eye = {
                "x": 1.7 * math.cos(angle),
                "y": 1.7 * math.sin(angle),
                "z": 0.8,
            }

            frame_traces: List[go.Scatter3d] = [
                go.Scatter3d(
                    x=path_x,
                    y=path_y,
                    z=path_z,
                    mode="lines",
                    line={"color": "#ff3b30", "width": 9},
                    name="Global Best Path",
                    showlegend=True,
                    hovertemplate=(
                        f"Global Best Cost: {best_cost:.2f}<extra></extra>"
                    ),
                ),
                go.Scatter3d(
                    x=ant_x,
                    y=ant_y,
                    z=ant_z,
                    mode="markers",
                    marker={
                        "size": 5,
                        "color": "#00d4ff",
                        "line": {"width": 1, "color": "#003a4d"},
                    },
                    text=ant_labels,
                    name="Ants",
                    showlegend=True,
                    hovertemplate="%{text}<extra></extra>",
                ),
            ]
            frame_traces.extend(pheromone_traces)

            frames.append(
                go.Frame(
                    data=frame_traces,
                    name=f"it{iter_idx + 1:03d}_s{step_idx + 1:03d}",
                    layout={
                        "scene": {
                            "camera": {
                                "eye": camera_eye,
                            }
                        },
                        "title": (
                            "ACO 3D Simulation | "
                            f"Iteration {iter_idx + 1}/{len(iteration_history)} "
                            f"| Step {step_idx + 1}/{local_steps} "
                            f"| Best Cost: {best_cost:.2f}"
                        ),
                    },
                )
            )

    return frames


def plot_simulation(
    points_3d: Sequence[Point3D],
    simulation_result: Dict[str, object],
    max_ants_visualized: int = 8,
    ant_interp_steps: int = 4,
    top_pheromone_edges: int = 45,
    save_html_path: Optional[str] = "aco_3d_simulation.html",
):
    """Render and optionally persist the interactive Plotly ACO 3D simulation."""
    iteration_history = simulation_result["iteration_history"]
    if not iteration_history:
        raise ValueError("iteration_history is empty. Run ACO before plotting.")

    city_x = [p[0] for p in points_3d]
    city_y = [p[1] for p in points_3d]
    city_z = [p[2] for p in points_3d]
    city_labels = [f"City {idx}" for idx in range(len(points_3d))]

    frames = create_frames(
        points_3d,
        iteration_history,
        max_ants_visualized=max_ants_visualized,
        ant_interp_steps=ant_interp_steps,
        top_pheromone_edges=top_pheromone_edges,
    )

    initial_frame = frames[0]
    static_city_trace = go.Scatter3d(
        x=city_x,
        y=city_y,
        z=city_z,
        mode="markers+text",
        marker={
            "size": 7,
            "color": "#f5f7fa",
            "line": {"width": 1.5, "color": "#1b1f3b"},
            "opacity": 0.95,
        },
        text=[str(i) for i in range(len(points_3d))],
        textposition="top center",
        name="Cities",
        hovertext=city_labels,
        hovertemplate="%{hovertext}<extra></extra>",
    )

    data = [static_city_trace]
    data.extend(initial_frame.data)

    # One slider stop per iteration boundary for readable control.
    frame_names = [frame.name for frame in frames]
    step_size = max(1, ant_interp_steps * len(points_3d))
    slider_steps = []
    for iter_idx in range(len(iteration_history)):
        frame_index = min(len(frame_names) - 1, (iter_idx + 1) * step_size - 1)
        slider_steps.append(
            {
                "label": f"Iter {iter_idx + 1}",
                "method": "animate",
                "args": [
                    [frame_names[frame_index]],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
            }
        )

    fig = go.Figure(
        data=data,
        frames=frames,
        layout=go.Layout(
            title="ACO 3D Simulation",
            template="plotly_dark",
            paper_bgcolor="#111426",
            plot_bgcolor="#111426",
            scene={
                "xaxis": {"title": "X", "backgroundcolor": "#0c1020"},
                "yaxis": {"title": "Y", "backgroundcolor": "#0c1020"},
                "zaxis": {"title": "Z", "backgroundcolor": "#0c1020"},
                "aspectmode": "cube",
            },
            legend={"orientation": "h", "y": 1.02, "x": 0.0},
            updatemenus=[
                {
                    "type": "buttons",
                    "direction": "left",
                    "x": 0.0,
                    "y": 1.12,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 55, "redraw": True},
                                    "transition": {"duration": 30},
                                    "fromcurrent": True,
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "mode": "immediate",
                                    "frame": {"duration": 0, "redraw": False},
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                        {
                            "label": "Slow",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 95, "redraw": True},
                                    "transition": {"duration": 40},
                                    "fromcurrent": True,
                                },
                            ],
                        },
                        {
                            "label": "Fast",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 30, "redraw": True},
                                    "transition": {"duration": 15},
                                    "fromcurrent": True,
                                },
                            ],
                        },
                    ],
                }
            ],
            sliders=[
                {
                    "x": 0.0,
                    "y": -0.04,
                    "len": 1.0,
                    "active": 0,
                    "pad": {"t": 30},
                    "steps": slider_steps,
                }
            ],
            margin={"l": 0, "r": 0, "b": 0, "t": 50},
        ),
    )

    if save_html_path:
        fig.write_html(save_html_path)

    fig.show()
    return fig


def run_demo(config: Optional[SimulationConfig] = None):
    cfg = config or SimulationConfig()

    rng = random.Random(cfg.random_seed)
    points_2d = [
        (rng.uniform(0.0, 1000.0), rng.uniform(0.0, 1000.0))
        for _ in range(cfg.n_cities)
    ]
    points_3d = generate_3d_points(
        points_2d=points_2d,
        z_mode="random",
        normalize=True,
        seed=cfg.random_seed,
    )

    result = run_aco_with_history(
        points_3d,
        n_ants=cfg.n_ants,
        n_iter=cfg.n_iter,
        alpha=cfg.alpha,
        beta=cfg.beta,
        evaporation=cfg.evaporation,
    )

    print("\nFinal Best Path:", result["best_path"])
    print("Final Best Cost:", f"{result['best_cost']:.2f}")

    return plot_simulation(
        points_3d,
        result,
        max_ants_visualized=cfg.max_ants_visualized,
        ant_interp_steps=cfg.ant_interp_steps,
        top_pheromone_edges=cfg.top_pheromone_edges,
        save_html_path=cfg.save_html_path,
    )
