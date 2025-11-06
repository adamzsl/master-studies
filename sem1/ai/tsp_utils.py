import random
import pandas as pd

class City:
    def __init__(self, name, x, y, z):
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