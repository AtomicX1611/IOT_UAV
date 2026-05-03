import matplotlib.pyplot as plt

def plot_path(points, path):
    x = [points[i][0] for i in path] + [points[path[0]][0]]
    y = [points[i][1] for i in path] + [points[path[0]][1]]

    plt.figure()
    plt.scatter([p[0] for p in points], [p[1] for p in points])
    plt.plot(x, y)
    plt.title("Best Path Found by ACO")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.show()


def plot_convergence(cost_history):
    plt.figure()
    plt.plot(cost_history)
    plt.title("ACO Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.grid()
    plt.show()