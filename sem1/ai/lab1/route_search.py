import random
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

    bfs_path, bfs_cost = tsp_bfs(map, start_city)
    print(f"BFS\nCost: {bfs_cost:.4f}\nPath: {bfs_path}\n")

    dfs_path, dfs_cost = tsp_dfs(map, start_city)
    print(f"DFS\nCost: {dfs_cost:.4f}\nPath: {dfs_path}")




if __name__ == "__main__":
    # main(num_cities = 7, density=1.0, symmetric=True, debug=False)
    n_cities = 7

    print("\n100% connections, Symmetric")
    main(num_cities=n_cities, density=1.0, symmetric=True, debug=False)
    
    print("\n80% connections, Symmetric")
    main(num_cities=n_cities, density=0.8, symmetric=True, debug=False)
    
    print("\n100% connections, Asymmetric")
    main(num_cities=n_cities, density=1.0, symmetric=False, debug=False)
    
    print("\n80% connections, Asymmetric")
    main(num_cities=n_cities, density=0.8, symmetric=False, debug=False)