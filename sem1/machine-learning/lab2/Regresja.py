# %% [markdown]
# Wczytanie najważniejszych bibliotek
# %%
import numpy as np
import einops  # Pozwala dużo czytelniej manipulować tensorami

np.set_printoptions(suppress=True, linewidth=120)
# %%
# "Jedyna Słuszna Metoda" inicjalizacji generatora liczb losowych
rng = np.random.default_rng()

# %% [markdown]
# Generowanie losowych danych
# %%
# Batch z 8 próbkami 2D
X = rng.random((8, 2))  # rozkład jednolity [0.0, 1.0)
print(f"{X.shape=}")
# %%
X = rng.standard_normal((8, 2))  # standardowy rozkład normalny (μ=0, σ=1)
print(f"{X.shape=}")
# %%
# Batch z 42 próbkami 7D
X = rng.random((42, 7))
print(f"{X.shape=}")
# %%
# Batch z 42 próbkami 7Dx3D
X = rng.random((42, 7, 3))
print(f"{X.shape=}")

# %% [markdown]
# Generowanie sekwencji
x = np.arange(3, 10, 2)
print(f"{x=}")
# %%
x = np.linspace(3, 10, 2)
print(f"{x=}")
# %%
x = np.linspace(3, 10, 20)
print(f"{x=}")

# %% [markdown]
# Transpozycja tensorów
X = rng.random((8, 2))
print(f"{X.T.shape=}")

# %% [markdown]
# Manipulacja tensorami
# %%
X = rng.random((42, 7, 3))
# %%
# Wersja NumPy
A = np.transpose(X, (2, 0, 1))
print(f"{A.shape=}")

# %%
# Wesja einops
A = einops.rearrange(X, "batch time features -> features batch time")
print(f"{A.shape=}")
# %%
einops.reduce(X, "batch time features -> batch features", "sum").shape
# %%
np.sum(X, axis=1).shape
# %%
einops.repeat(X, "batch time features -> batch time nreps features", nreps=5).shape
# %%
# np.tile()
# np.repeat()
# np.einsum

# %%
einops.rearrange(X, "(a1 a2) time features -> a1 time features a2", a1=2).shape

# %%
einops.rearrange(X, "batch time features -> (features batch) time").shape

# %% [markdown]
# Mnożenia tensorami

# %%
# 8 dwuwymiarowych próbek
X = rng.random((8, 2))
# dwuwwartościowy wektor wag
w = rng.random(2)

# %%
# dot product
# [8, 2] @ [2, ] ⇒ [8, ]
print(f"{(X @ w).shape=}")
# %%
# [2, ] @ [8, 2].T ⇒ [8,]
print(f"{(w @ X.T).shape=}")
# %%
# wektor [2, ] != [2, 1]
# [2, 1] @ [8, 2].T ⇒ ERROR
print(f"{(w[:, np.newaxis] @ X.T).shape=}")
# %%
w[:, np.newaxis].shape

# %%
einops.rearrange(w, "features -> features 1").shape

# %%
einops.rearrange(w, "features -> features 1").squeeze().shape

# %%
# [8, 2] @ [2, 1] ⇒ [8, 1]
print(f"{(X @ w[:, np.newaxis]).shape=}")

# %%
# Hadamard product
# %%
# [8, 2] @ [8, 2] ⇒ ERROR
print(f"{(X @ X).shape=}")
# %%
# [8, 2] * [8, 2] ⇒ OK
print(f"{(X * X).shape=}")
# %%
# [8, 2] * [2, ] ⇒ [8, 2] <- broadcasing
print(f"{(X * w).shape=}")

# %%
# Broadcasting działa sprawdzając zgodność od ostatniego wymiaru (równy lub 1)
# %% ostatni wymiar jest 1, może być dowolnie broadcastowany
print(f"{(rng.random((2, 1)) * rng.random((2, 1000))).shape=}")
# %% ostatni wymiar jest zgodny, przedostatni jest 1
print(f"{(rng.random((1, 2)) * rng.random((1000, 2))).shape=}")
# %% broadcasting działa na wielu wymiarach
print(f"{(rng.random((2, 1, 1, 1)) * rng.random((2, 10, 20, 30))).shape=}")
# %% ostatni wymiar nie jest 1, nie może być dowolnie broadcastowany
print(f"{(rng.random((2,)) * rng.random((2, 1000))).shape=}")

# %% [markdown]
# Wizualizacja
# %%
# statyczne wykresy
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# interaktywne wykresy
import plotly.graph_objects as go

# %% [markdown]
# Prosty model liniowy
# %%
x = np.linspace(-20, 20, 1000)
t = 4 * x + 2 * x**2 + 1 * x**3 + 0.5 * x**4
ts = np.sin(0.5 * x)

# %%
plt.plot(x, t)
# plt.xlim(-10, 10)
# plt.ylim(-10, 10)
plt.show()

# %%
sns.set_context("poster")
# sns.set_context("notebook")
sns.set_style("darkgrid")
# sns.set_style("ticks")
# sns.set_style("whitegrid")

sns.lineplot(x=x, y=t)
# plt.plot(x, t)
plt.show()

# %%
fig = go.Figure(go.Scatter(x=x, y=t, mode="lines+markers"))
fig.show()

# %% [markdown]
# Regersja liniowa


# %%
# prosta metryka
def MSE(true, pred):
    return np.mean((true - pred) ** 2)


# %%
x_ = np.c_[np.ones_like(x), x].T  # transpose żeby pasowało do wzoru
print(f"{x_.shape=}")

# %%
# y = Ax + b -> y = θ x
w = rng.standard_normal((2,))
y = x_.T @ w
print(f"{w=}")
print(f"{y.shape=}")
print(f"{MSE(t, y)=}")
plt.plot(x, t, label="Target")
plt.plot(x, y, label="Predict")
plt.legend()
plt.show()
# %%
# θ = (x x.T)^-1 x y
w = np.linalg.inv(x_ @ x_.T) @ x_ @ t
y = x_.T @ w
print(f"{w=}")
print(f"{y.shape=}")
print(f"{MSE(t,y)}")
plt.plot(x, t, label="Target")
plt.plot(x, y, label="Predict")
plt.legend()
plt.show()
# Nadal nie jest to oczekiwany wynik - nasza funkcja jest zbyt prosta i nie ma jak się dopasować

# %%
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(5, include_bias=True)
print(f"{x.shape=}")
# x musi miec kształt (n_samples, n_features)
# dostawienie x_0 = "1" jest w trakcie transformacji
xl = einops.rearrange(x, "n_samples -> n_samples 1")
print(f"{xl.shape=}")
xp = poly.fit_transform(xl)
print(f"{xp.shape=}")

# %%
w = np.linalg.inv(xp.T @ xp) @ xp.T @ t
y = xp @ w
print(f"{w=}")
print(f"{y.shape=}")
print(f"{MSE(t,y)=}")
plt.plot(x, t, label="Target")
plt.plot(x, y, label="Predict")
plt.legend()
plt.show()
# %%
poly = PolynomialFeatures(7, include_bias=True)
print(f"{xl.shape=}")
xx = np.linspace(-21, 21, 1000)
xx = einops.rearrange(xx, "time -> time 1")
xxp = poly.fit_transform(xx)
xp = poly.fit_transform(xl)
print(f"{xp.shape=}")

w = np.linalg.inv(xp.T @ xp) @ xp.T @ ts
#y = xp @ w
y = xxp @ w # wielomian "odlatuje" jak tylko wyjdzie poza zakres
print(f"{w=}")
print(f"{y.shape=}")
print(f"{MSE(ts,y)=}")
plt.plot(x, ts, label="Target")
plt.plot(x, y, label="Predict")
plt.legend()
plt.show()

# %%
