import numpy as np

FOOD_MIN, FOOD_MAX = 0.0, 10.0
SERV_MIN, SERV_MAX = 0.0, 10.0
TIP_MIN, TIP_MAX = 0.0, 15.0

def clamp(x, a, b):
    # clip x in [a, b] range
    return max(a, min(b, x))

def tri(x, a, b, c):
    # triangular membership
    if a == b and b == c:
        return 1.0 if x == a else 0.0
    if x <= a or x >= c: # outside range
        return 0.0
    if x == b: # peak
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b != a else 0.0
    return (c - x) / (c - b) if c != b else 0.0

def trap(x, a, b, c, d):
    # trapezoidal membership
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a) if b != a else 0.0
    return (d - x) / (d - c) if d != c else 0.0

def food_low(x):    return trap(x, 0.0, 0.0, 2.5, 5.0)
def food_med(x):    return tri(x, 3.0, 5.0, 7.0)
def food_high(x):   return trap(x, 5.0, 7.5, 10.0, 10.0)

def serv_low(x):    return trap(x, 0.0, 0.0, 2.5, 5.0)
def serv_med(x):    return tri(x, 3.0, 5.0, 7.0)
def serv_high(x):   return trap(x, 5.0, 7.5, 10.0, 10.0)

def AND(a, b):
    return min(a, b)

def OR(a, b):
    return max(a, b)

# TSK - 0 
rules = [
    ("low service OR low food -> tip=2",
     lambda f, s: OR(serv_low(s), food_low(f)), 2.0),

    ("medium service AND medium food -> tip=8",
     lambda f, s: AND(serv_med(s), food_med(f)), 8.0),

    ("high service AND high food -> tip=14",
     lambda f, s: AND(serv_high(s), food_high(f)), 14.0),

    ("high service AND medium food -> tip=12",
     lambda f, s: AND(serv_high(s), food_med(f)), 12.0),

    ("medium service AND high food -> tip=12",
     lambda f, s: AND(serv_med(s), food_high(f)), 12.0),

    ("low service AND medium food -> tip=5",
     lambda f, s: AND(serv_low(s), food_med(f)), 5.0),

    ("medium service AND low food -> tip=5",
     lambda f, s: AND(serv_med(s), food_low(f)), 5.0),
]

def tsk_infer(food, service, verbose=True):

    f = clamp(float(food), FOOD_MIN, FOOD_MAX)
    s = clamp(float(service), SERV_MIN, SERV_MAX)

    weights = []
    consequents = []
    details = []

    for name, antecedent, c in rules:
        w = float(antecedent(f, s))  # poprzednik reguly
        weights.append(w)
        consequents.append(c)
        details.append((name, w, c))

    W = sum(weights)
    if W == 0.0:
        y = 0.0
    else:
        y = sum(w * c for w, c in zip(weights, consequents)) / W

    y = clamp(y, TIP_MIN, TIP_MAX)

    if verbose:
        print(f"food={f:.1f}, service={s:.1f} -> tip={y:.2f}")
        print("rules (w, tip):")
        for i, (w, c) in enumerate(zip(weights, consequents), 1):
            print(f"  r{i}: w={w:.2f}, tip={c}")
        print()

    return y

if __name__ == "__main__":
    examples = [
        (2.0, 3.0),
        (5.0, 5.0),
        (9.0, 8.0),
        (7.0, 5.5),
        (3.5, 6.0),
    ]
    for food, service in examples:
        tsk_infer(food, service, verbose=True)