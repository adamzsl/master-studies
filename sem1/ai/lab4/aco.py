import time
import numpy as np

from lab1.route_search import tsp_bfs, tsp_dfs
from lab2.greedy import tsp_nn, tsp_greedy
from tsp_utils import *
from lab3.astar import *

def aco(map, start_city, n_ants, n_iterations, alpha, beta, evaporation, Q):
    best_path = []
    total_cost = np.inf
    n = len(map)
    pheromone = np.ones((n, n)) # pheromone trails between every city pair

    for _ in range(n_iterations):
        all_paths = []
        all_lengths = []

        # each ant builds a path
        for _ in range(n_ants):
            visited = [False] * n # tracks visited cities
            current = start_city
            visited[current] = True
            path = [current]
            length = 0

            # visit all other cities
            while not all(visited):
                unv = [j for j in range(n) if not visited[j] and map[current][j] > 0] # indices of unvisited cities
                if not unv:
                    length = np.inf # no unvisited cities left, break the loop
                    break
                probs = np.array([
                    pheromone[current, j]**alpha / (map[current][j]**beta) # τ (ij) ^ α / η (ij) ^ β
                    for j in unv
                ])
                probs /= probs.sum() # normalize to sum to 1
                nxt = np.random.choice(unv, p=probs) # select next city based on probabilities

                path.append(nxt)
                length += map[current][nxt]
                visited[nxt] = True
                current = nxt

            # close the tour back to start_city if possible
            if length != np.inf and map[current][start_city] > 0:
                length += map[current][start_city]
                path.append(start_city)

                all_paths.append(path)
                all_lengths.append(length)

                # update global best
                if length < total_cost:
                    best_path = path.copy()
                    total_cost = length

        # evaporate
        pheromone *= (1 - evaporation)

        # deposit new pheromone on each full cycle
        for path, length in zip(all_paths, all_lengths):
            delta = Q / length  # how much pheromone is deposited on each edge of that path
                                # shorter path = more pheromone
            for i in range(len(path) - 1):
                pheromone[path[i], path[i+1]] += delta # add pheromone to the edge

    if not best_path:
        return [], -1
    return best_path, total_cost


def main(num_cities, density, symmetric, debug):
    #generate cities
    n = num_cities
    cities = gen_cities(n)
    if debug:
        [print(c) for c in cities]

    #generate map
    map = gen_map(cities, n, density, symmetric)
    if debug:
        print("Map:")
        map_print(n, map)
    

    #TSP
    start_city = 0 
    
    start_time = time.time()
    path, cost = aco(map, 
                     start_city, 
                     n_ants=100, 
                     n_iterations=1000, 
                     alpha=1, # controls how strongly the ant relies on the pheromone trail
                     beta=1, # controls how strongly the ant relies on heuristic (distance)
                     evaporation = 0.1, 
                     Q=1 # pheromone deposit
                     )
    end_time = time.time()
    aco_time = end_time - start_time
    print(f"ACO\nTime: {aco_time}\nCost: {cost:.4f}\nPath: {path}\n")

    start_time = time.time()
    path, cost = astar(map, start_city)
    end_time = time.time()
    astar_time = end_time - start_time
    print(f"A*\nTime: {astar_time}\nCost: {cost:.4f}\nPath: {path}\n")

    start_time = time.time()
    bfs_path, bfs_cost = tsp_bfs(map, start_city)
    end_time = time.time()
    bfs_time = end_time - start_time
    print(f"BFS\nTime: {bfs_time}\nCost: {bfs_cost:.4f}\nPath: {bfs_path}\n")

    start_time = time.time()
    dfs_path, dfs_cost = tsp_dfs(map, start_city)
    end_time = time.time()
    dfs_time = end_time - start_time
    print(f"DFS\nTime: {dfs_time}\nCost: {dfs_cost:.4f}\nPath: {dfs_path}\n")

    start_time = time.time()
    nn_path, nn_cost = tsp_nn(map, start_city)
    end_time = time.time()
    nn_time = end_time - start_time
    print(f"NN\nTime: {nn_time}\nCost: {nn_cost:.4f}\nPath: {nn_path}\n")

    start_time = time.time()
    greedy_path, greedy_cost = tsp_greedy(map, start_city)
    end_time = time.time()
    greedy_time = end_time - start_time
    print(f"Greedy\nTime: {greedy_time}\nCost: {greedy_cost:.4f}\nPath: {greedy_path}\n")


if __name__ == "__main__":
    # main(num_cities = 7, density=1.0, symmetric=True, debug=False)
    n_cities = 9

    print("\n100% connections, Symmetric")
    main(num_cities=n_cities, density=1.0, symmetric=True, debug=True)
    
    # print("\n80% connections, Symmetric")
    # main(num_cities=n_cities, density=0.8, symmetric=True, debug=False)
    
    # print("\n100% connections, Asymmetric")
    # main(num_cities=n_cities, density=1.0, symmetric=False, debug=False)
    
    # print("\n80% connections, Asymmetric")
    # main(num_cities=n_cities, density=0.8, symmetric=False, debug=False)