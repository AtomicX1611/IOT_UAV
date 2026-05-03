import random
import math

class ACO_TSP:
    def __init__(self, points, n_ants=20, n_iter=100, alpha=1, beta=2, evaporation=0.2):
        self.points = points
        self.n = len(points)
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation

        self.pheromone = [[1 for _ in range(self.n)] for _ in range(self.n)]

    def distance(self, i, j):
        x1, y1 = self.points[i]
        x2, y2 = self.points[j]
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

    def run(self):
        best_path = None
        best_cost = float('inf')

        cost_history = []

        for iteration in range(self.n_iter):
            all_paths = []

            for _ in range(self.n_ants):
                path = self.construct_path()
                cost = self.path_cost(path)
                all_paths.append((path, cost))

                if cost < best_cost:
                    best_cost = cost
                    best_path = path

            self.update_pheromones(all_paths)

          
            cost_history.append(best_cost)

            iteration_best = min(all_paths, key=lambda x: x[1])[1]
            print(f"Iteration {iteration+1}: Iter Best = {iteration_best:.2f}, Global Best = {best_cost:.2f}")

       
        return best_path, best_cost, cost_history

    def construct_path(self):
        start = 0
        path = [start]
        visited = set(path)

        while len(path) < self.n:
            current = path[-1]

            if random.random() < 0.1:
                next_city = random.choice([i for i in range(self.n) if i not in visited])
            else:
                probs = []

                for next_city in range(self.n):
                    if next_city not in visited:
                        pher = self.pheromone[current][next_city] ** self.alpha
                        dist = self.distance(current, next_city)
                        heuristic = (1 / (dist + 1e-5)) ** self.beta
                        probs.append((next_city, pher * heuristic))

                total = sum(p for _, p in probs)
                probs = [(city, p / total) for city, p in probs]

                r = random.random()
                cumulative = 0

                for city, prob in probs:
                    cumulative += prob
                    if cumulative >= r:
                        next_city = city
                        break

            path.append(next_city)
            visited.add(next_city)

        return path

    def path_cost(self, path):
        cost = 0
        for i in range(len(path) - 1):
            cost += self.distance(path[i], path[i+1])

        cost += self.distance(path[-1], path[0])
        return cost

    def update_pheromones(self, all_paths):
        for i in range(self.n):
            for j in range(self.n):
                self.pheromone[i][j] *= (1 - self.evaporation)

        all_paths.sort(key=lambda x: x[1])
        best_paths = all_paths[:5]

        for path, cost in best_paths:
            for i in range(len(path) - 1):
                a, b = path[i], path[i+1]
                self.pheromone[a][b] += 200 / cost
                self.pheromone[b][a] += 200 / cost