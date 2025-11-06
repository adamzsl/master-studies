import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Input variables
d = ctrl.Antecedent(np.arange(0, 101, 1), 'd')         # range [m]
a = ctrl.Antecedent(np.arange(10, 81, 1), 'a')         # angle [deg]
k = ctrl.Antecedent(np.arange(0, 0.51, 0.01), 'k')     # air resistance coefficient
m = ctrl.Antecedent(np.arange(0.1, 5.1, 0.1), 'm')     # mass [kg]

# Output variable
v0 = ctrl.Consequent(np.arange(0, 51, 1), 'v0')        # initial velocity [m/s]

# Membership functions
d['small'] = fuzz.trimf(d.universe, [0, 0, 30])
d['medium'] = fuzz.trimf(d.universe, [20, 50, 80])
d['large'] = fuzz.trimf(d.universe, [60, 100, 100])

a['low'] = fuzz.trimf(a.universe, [10, 10, 30])
a['medium'] = fuzz.trimf(a.universe, [20, 45, 70])
a['high'] = fuzz.trimf(a.universe, [60, 80, 80])

k['small'] = fuzz.trimf(k.universe, [0, 0, 0.15])
k['medium'] = fuzz.trimf(k.universe, [0.1, 0.25, 0.4])
k['large'] = fuzz.trimf(k.universe, [0.3, 0.5, 0.5])

m['light'] = fuzz.trimf(m.universe, [0.1, 0.1, 1.5])
m['medium'] = fuzz.trimf(m.universe, [1, 2.5, 4])
m['heavy'] = fuzz.trimf(m.universe, [3, 5, 5])

v0['low'] = fuzz.trimf(v0.universe, [0, 0, 15])
v0['medium'] = fuzz.trimf(v0.universe, [10, 25, 40])
v0['high'] = fuzz.trimf(v0.universe, [30, 50, 50])

# Fuzzy rules
rule1 = ctrl.Rule(d['small'] & a['medium'] & k['small'] & m['light'], v0['low'])
rule2 = ctrl.Rule(d['medium'] & a['medium'] & k['small'] & m['medium'], v0['medium'])
rule3 = ctrl.Rule(d['large'] & a['medium'] & k['small'] & m['heavy'], v0['high'])
rule4 = ctrl.Rule(d['large'] & a['high'] & k['large'] & m['light'], v0['high'])
rule5 = ctrl.Rule(d['small'] & a['low'] & k['large'] & m['light'], v0['medium'])
rule6 = ctrl.Rule(d['medium'] & a['medium'] & k['medium'] & m['medium'], v0['medium'])
rule7 = ctrl.Rule(d['large'] & a['low'] & k['small'] & m['medium'], v0['high'])
rule8 = ctrl.Rule(d['small'] & a['high'] & k['small'] & m['heavy'], v0['low'])
# rule9  = ctrl.Rule(d['small']  & a['medium'] & k['small']  & m['light'],  v0['medium'])
rule10 = ctrl.Rule(d['small']  & a['medium'] & k['medium'] & m['light'],  v0['medium'])
rule11 = ctrl.Rule(d['small']  & a['medium'] & k['large']  & m['light'],  v0['high'])
rule12 = ctrl.Rule(d['medium'] & (a['low'] | a['high']) & k['small']  & m['medium'], v0['high'])
rule13 = ctrl.Rule(d['medium'] & a['medium'] & k['large'] & m['medium'], v0['high'])
rule14 = ctrl.Rule(d['large']  & (a['low'] | a['high']) & k['small']  & m['heavy'],  v0['high'])

# Control system
v0_ctrl = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule10, rule11, rule12, rule13, rule14
])
v0_sim = ctrl.ControlSystemSimulation(v0_ctrl)

g = 9.81

def v0_no_drag(R, ang_deg):
    ang = np.deg2rad(ang_deg)
    s2a = np.sin(2*ang)
    if s2a <= 0:
        return np.nan
    return np.sqrt(R * g / s2a)

def y_at(v0_guess, R, ang_deg, k_coeff, mass):
    ang = np.deg2rad(ang_deg)
    beta = k_coeff / mass if mass > 0 else 0.0
    V0x = v0_guess * np.cos(ang)
    V0y = v0_guess * np.sin(ang)
    if beta < 1e-8:
        if V0x <= 0:
            return -1
        t = R / V0x
        return V0y * t - 0.5 * g * t * t
    ratio = beta * R / V0x if V0x > 0 else 2.0
    if ratio >= 1.0:
        return -1
    t = -1.0 / beta * np.log(1 - ratio)
    e = np.exp(-beta * t)
    # y(t): y = (1/beta)*(V0y + g/beta)*(1 - e^{-beta t}) - g t / beta
    return ( (V0y + g / beta) * (1 - e) / beta ) - (g * t / beta)

def v0_with_drag(R, ang_deg, k_coeff, mass):
    if k_coeff < 1e-8:
        return v0_no_drag(R, ang_deg)
    lo, hi = 0.1, 100.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        ym = y_at(mid, R, ang_deg, k_coeff, mass)
        if ym > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

# Przykładowe dane wejściowe
d_in = 20
a_in = 45
k_in = 0.1
m_in = 1.0

# Fuzzy
v0_sim.input['d'] = d_in
v0_sim.input['a'] = a_in
v0_sim.input['k'] = k_in
v0_sim.input['m'] = m_in
v0_sim.compute()

if 'v0' in v0_sim.output:
    print(f"Fuzzy v0: {v0_sim.output['v0']:.2f} m/s")
else:
    print("Brak wyniku fuzzy.")

# Analityczne
v_no_drag  = v0_no_drag(d_in, a_in)
v_with_drag = v0_with_drag(d_in, a_in, k_in, m_in)

print(f"Analytical no drag:    {v_no_drag:.2f} m/s")
print(f"Analytical with drag:  {v_with_drag:.2f} m/s")

# Podgląd
v0.view(sim=v0_sim)
plt.show()