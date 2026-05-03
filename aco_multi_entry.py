import random
import math

class ACO_MultiEntry:
    def __init__(self, regions, num_entry=8, n_ants=30, n_iter=150):
        self.regions = regions
        self.num_entry = num_entry
        self.n_ants = n_ants
        self.n_iter = n_iter

        self.nodes = []
        self.node_to_region = []

        self.generate_nodes()

        self.n = len(self.nodes)
        self.pheromone = [[1 for _ in range(self.n)] for _ in range(self.n)]

    def generate_nodes(self):
        for idx, region in enumerate(self.regions):
            cx, cy = region["center"]
            r = region["radius"]

            for i in range(self.num_entry):
                angle = 2 * math.pi * i / self.num_entry
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)

                self.nodes.append((x, y))
                self.node_to_region.append(idx)

    def distance(self, i, j):
        x1, y1 = self.nodes[i]
        x2, y2 = self.nodes[j]
        return math.dist((x1,y1), (x2,y2))

    def run(self):
        best_cost = float('inf')
        best_path = None

        for it in range(self.n_iter):
            all_paths = []

            for _ in range(self.n_ants):
                path = self.construct_path()
                cost = self.path_cost(path)

                all_paths.append((path, cost))

                if cost < best_cost:
                    best_cost = cost
                    best_path = path

            self.update_pheromones(best_path, best_cost)

            print(f"Iteration {it+1}: Best = {best_cost:.2f}")

        return best_path, best_cost

    def construct_path(self):
        start = random.randint(0, self.n - 1)
        path = [start]
        visited_regions = {self.node_to_region[start]}

        while len(visited_regions) < len(self.regions):
            current = path[-1]

            candidates = []
            for i in range(self.n):
                if self.node_to_region[i] not in visited_regions:
                    pher = self.pheromone[current][i]
                    dist = self.distance(current, i)
                    score = pher * (1/(dist+1e-5))
                    candidates.append((i, score))

            total = sum(score for _, score in candidates)
            r = random.random() * total

            cum = 0
            for node, score in candidates:
                cum += score
                if cum >= r:
                    next_node = node
                    break

            path.append(next_node)
            visited_regions.add(self.node_to_region[next_node])

        return path

    def path_cost(self, path):
        cost = 0
        visited = set()

        for i in range(len(path)-1):
            cost += self.distance(path[i], path[i+1])

        for node in path:
            region = self.node_to_region[node]
            if region in visited:
                cost += 1000  
            visited.add(region)

        return cost

    def update_pheromones(self, best_path, best_cost):
        for i in range(self.n):
            for j in range(self.n):
                self.pheromone[i][j] *= 0.8

        for i in range(len(best_path)-1):
            a, b = best_path[i], best_path[i+1]
            self.pheromone[a][b] += 50 / best_cost
            self.pheromone[b][a] += 50 / best_cost