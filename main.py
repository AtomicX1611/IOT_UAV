from plotly_aco_3d import SimulationConfig, run_demo


def main():
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


if __name__ == "__main__":
    main()