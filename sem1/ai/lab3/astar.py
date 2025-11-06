import time

from lab1.route_search import tsp_bfs, tsp_dfs
from lab2.greedy import tsp_nn, tsp_greedy
from tsp_utils import *

   
def min_path(map):
    n = len(map)
    h = float('inf')
    for i in range(n):
        for j in range(n):
            if map[i][j] != 0 and i != j and map[i][j]<h:
                h = map[i][j]
    return h

def avg_edge(map):
    n = len(map)
    total, count = 0.0, 0
    for i in range(n):
        for j in range(n):
            if map[i][j] != 0 and i != j:
                total += map[i][j]
                count += 1
    return (total / count)

class Node:
    def __init__(self, parent, city, g, path):
        self.parent = parent
        self.city = city
        self.g = g         # cost from start
        self.h = 0         # heuristic cost
        self.f = 0         # total cost = g + h
        self.path = path   # visited sequence of cities

    def __eq__(self, other):
        return self.path == other.path
    

def astar(map, start_city):
    n = len(map)
    h_cost = min_path(map) # smallest edge in graph
    start = Node(None, start_city, 0, [start_city])
    open, closed = [start], []

    
    while open:
        current = min(open, key=lambda node: node.f) # node with lowest f
        open.remove(current)
        closed.append(current)
        
        # goal: visited all cities and can return to start
        if len(current.path) == n and map[current.city][start_city] > 0:
            final_cost = current.g + map[current.city][start_city]
            return current.path + [start_city], final_cost

        for child in range(n):
            
            # skip if no path or already visited
            if map[current.city][child] == 0 or child in current.path:
                continue
            
            g2 = current.g + map[current.city][child]
            h2 = h_cost * (n - len(current.path))   # admissible heuristic
            f2 = g2 + h2
            path2 = current.path + [child]
            child = Node(current, child, g2, path2)
            child.h, child.f = h2, f2

            # skip if a better or equal path already seen
            better_f = lambda lst: any(node.path == child.path and node.f <= child.f for node in lst)
            if better_f(open) or better_f(closed):
                continue

            open.append(child)

    return None, -1


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
    print(f"DFS\nTime: {dfs_time}\nCost: {dfs_cost:.4f}\nPath: {dfs_path}")

    start_time = time.time()
    nn_path, nn_cost = tsp_nn(map, start_city)
    end_time = time.time()
    nn_time = end_time - start_time
    print(f"NN\nTime: {nn_time}\nCost: {nn_cost:.4f}\nPath: {nn_path}")

    start_time = time.time()
    greedy_path, greedy_cost = tsp_greedy(map, start_city)
    end_time = time.time()
    greedy_time = end_time - start_time
    print(f"Greedy\nTime: {greedy_time}\nCost: {greedy_cost:.4f}\nPath: {greedy_path}")


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