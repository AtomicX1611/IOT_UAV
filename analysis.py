import random
import matplotlib.pyplot as plt
from aco_tsp import ACO_TSP


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