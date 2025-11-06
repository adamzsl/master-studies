import random
import time
import pandas as pd

class City:
    def __init__(self, name, x, y ,z):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
    def __str__(self):
        return f"{self.name} ({self.x}, {self.y}, {self.z})"
    

def gen_coordinates():
    x = random.uniform(-100, 100)
    y = random.uniform(-100, 100)
    z = random.uniform(0, 50)

    return x, y, z

def gen_cities(n):
    cities = []
    for i in range(n):
        x, y, z = gen_coordinates()
        name = f"city{i}"
        cities.append(City(name, x, y, z))
    return cities

def map_print(n, map):
    df = pd.DataFrame(map)
        
    df.index = [f"City {i}" for i in range(n)]
    df.columns = [f"City {j}" for j in range(n)]
    
    pd.set_option('display.precision', 2)
    pd.set_option('display.float_format', '{:.2f}'.format)
    
    print(df)

def euclidean_distance(city1, city2):
    distance = ((city1.x - city2.x)**2 + (city1.y - city2.y)**2 + (city1.z - city2.z)**2)**0.5
    return distance

def gen_map(cities, n, density, symmetric):
    map = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if random.random() <= density:
                if symmetric:
                    map[i][j] = euclidean_distance(cities[i], cities[j])
                    map[j][i] = map[i][j]
                else:
                    if i != j:
                        distance = euclidean_distance(cities[i], cities[j])
                        if cities[i].z < cities[j].z:
                            distance *= 1.1
                        elif cities[i].z > cities[j].z:
                            distance *= 0.9
                        map[i][j] = distance
    return map

def tsp_bfs(map, start_city):
    n = len(map)

    queue = [(start_city, [start_city], 0)]

    best_path = None
    min_cost = float('inf')

    while queue:
        city, path, cost = queue.pop(0) # FIFO

        if len(path) == n:
            if map[city][start_city] > 0:
                cost += map[city][start_city]
                if cost < min_cost:
                    min_cost = cost
                    best_path = path + [start_city]
        else:
            for next_city in range(n):
                if next_city not in path and map[city][next_city] != 0:
                    queue.append((next_city, path + [next_city], cost + map[city][next_city]))
    return best_path, min_cost

def tsp_dfs(map, start_city):
    n = len(map)

    stack = [(start_city, [start_city], 0)] # LIFO

    best_path = None
    min_cost = float('inf')

    while stack:
        city, path, cost = stack.pop()

        if len(path) == n:
            if map[city][start_city] > 0:
                cost += map[city][start_city]
                if cost < min_cost:
                    min_cost = cost
                    best_path = path + [start_city]
        else:
            for next_city in range(n-1, -1, -1):
                if next_city not in path and map[city][next_city] > 0:
                    stack.append((next_city, path + [next_city], cost + map[city][next_city]))
    return best_path, min_cost

def tsp_nn(map, start_city):
    n = len(map)
    path = [start_city]
    cost = 0

    while len(path) < n:
        city = path[-1]
        min_dist = float('inf')
        next_city = None

        for i in range(n):
            if i not in path and map[city][i] > 0:
                if map[city][i] < min_dist:
                    min_dist = map[city][i]
                    next_city = i
        
        if next_city is None:
            return None, -1
        
        path.append(next_city)
        cost += min_dist

    if map[path[-1]][start_city] > 0:
        cost += map[path[-1]][start_city]
        path.append(start_city)
        return path, cost
    else:
        return None, -1
    
def tsp_greedy(map, start_city):
    n = len(map)
    path = [start_city]
    cost = 0

    while len(path) < n:
        city = path[-1]
        min_dist = float('inf')
        next_city = None

        # check all unvisited cities
        for city1 in range(n):
            if city1 not in path and map[city][city1] > 0:
                cost1 = map[city][city1]
                
                # if its last city, dont check second city
                if len(path) == n-1:
                    if cost1 < min_dist:
                        min_dist = cost1
                        best_next_city = city1
                    continue
                
                # check second city
                min_dist2 = float('inf')
                for city2 in range(n):
                    if city2 not in path and city2 != city1 and map[city1][city2] > 0:
                        cost2 = map[city1][city2]
                        if cost2 < min_dist2:
                            min_dist2 = cost2

                # if no path to second city
                if min_dist2 == float('inf'):
                    total_cost = cost1
                else:
                    total_cost = cost1 + min_dist2

                if total_cost < min_dist:
                    min_dist = total_cost
                    best_next_city = city1

        # if no path to any city
        if best_next_city is None:
            return None, -1
        
        # add just 1st city to path
        path.append(best_next_city)
        cost += map[city][best_next_city]
    
    # check if path is complete
    if map[path[-1]][start_city] > 0:
        cost += map[path[-1]][start_city]
        path.append(start_city)
        return path, cost
    else:
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
    main(num_cities=n_cities, density=1.0, symmetric=True, debug=False)
    
    # print("\n80% connections, Symmetric")
    # main(num_cities=n_cities, density=0.8, symmetric=True, debug=False)
    
    # print("\n100% connections, Asymmetric")
    # main(num_cities=n_cities, density=1.0, symmetric=False, debug=False)
    
    # print("\n80% connections, Asymmetric")
    # main(num_cities=n_cities, density=0.8, symmetric=False, debug=False)