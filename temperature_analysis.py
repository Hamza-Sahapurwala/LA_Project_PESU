import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)


def info(label: str, value) -> None:
    print(f"{label:<40}: {value}")


def cwr(concept: str, why: str, result: str) -> None:
    print("\nConcept -> Purpose -> Outcome")
    print(f"- Concept : {concept}")
    print(f"- Purpose : {why}")
    print(f"- Outcome : {result}")


def rref(matrix: np.ndarray, tol: float = 1e-10):
    m = matrix.astype(float).copy()
    rows, cols = m.shape
    r = 0
    pivots = []

    for c in range(cols):
        if r >= rows:
            break

        pivot = np.argmax(np.abs(m[r:, c])) + r
        if abs(m[pivot, c]) < tol:
            continue

        m[[r, pivot]] = m[[pivot, r]]
        m[r] = m[r] / m[r, c]

        for rr in range(rows):
            if rr != r:
                m[rr] -= m[rr, c] * m[r]

        pivots.append(c)
        r += 1

    m[np.abs(m) < tol] = 0.0
    return m, pivots


# STEP 1: DATA COLLECTION
section("STEP 1: DATA COLLECTION")

df = pd.read_csv("GlobalLandTemperaturesByMajorCity.csv", parse_dates=["dt"])
df = df.dropna(subset=["AverageTemperature"])
df["Year"] = df["dt"].dt.year

cities = ["London", "New York", "Mumbai", "Beijing", "Cairo", "Sydney", "Moscow", "Tokyo"]
city_year_count = df.groupby("City")["Year"].nunique()
selected_cities = [c for c in cities if c in city_year_count.index and city_year_count[c] >= 100]

print("Available Cities:", ", ".join(selected_cities))
user_city = input("Enter city for prediction (default: first city): ").strip()

if user_city in selected_cities:
    focus_city = user_city
else:
    focus_city = selected_cities[0]

annual_full = (
    df[df["City"].isin(selected_cities)]
    .groupby(["Year", "City"])["AverageTemperature"]
    .mean()
    .unstack("City")
)

annual = annual_full.dropna()

years = annual.index.to_numpy()
A = annual.to_numpy()
city_labels = list(annual.columns)
focus_idx = city_labels.index(focus_city)

info("Selected Cities", ", ".join(selected_cities))
info("Focus City", focus_city)
info("Matrix Shape (Years x Cities)", A.shape)

cwr("Real-world data", "Analyze global temperature trends", "Matrix formed from climate dataset")


# STEP 2: MATRIX REPRESENTATION
section("STEP 2: MATRIX REPRESENTATION")

A_mean = A.mean(axis=0)
A_centered = A - A_mean

print("\nMatrix A (first 5 rows):")
print(pd.DataFrame(A, index=years, columns=city_labels).head())

cwr("Matrix Representation", "Convert data into linear algebra form", "Temperature matrix A created")


# STEP 3: RREF
section("STEP 3: RREF")

rref_matrix, pivots = rref(A_centered)
rank = len(pivots)

print("\nRREF Matrix (first 5 rows):")
print(pd.DataFrame(rref_matrix, index=years, columns=city_labels).head())

info("Rank", rank)

cwr("RREF", "Find independent city patterns", f"Rank = {rank}")


# STEP 4: VECTOR SPACE
section("STEP 4: VECTOR SPACE")

cwr(
    "Vector Space",
    "Understand data structure",
    "Row space = yearly trends, Column space = city patterns"
)


# STEP 5: BASIS + ORTHOGONALIZATION
section("STEP 5: BASIS + ORTHOGONALIZATION")

_, basis_indices = rref(A_centered.T)
B = A_centered[:, basis_indices]
Q, _ = np.linalg.qr(B)

print("\nBasis Matrix B (first 5 rows):")
print(pd.DataFrame(B, index=years).head())

print("\nOrthogonal Matrix Q (first 5 rows):")
print(pd.DataFrame(Q, index=years).head())

cwr("Basis", "Remove redundancy", "Independent features selected")
cwr("Orthogonalization", "Remove correlation", "Orthogonal climate components obtained")


# STEP 6: PROJECTION
section("STEP 6: PROJECTION")

P = Q @ Q.T
A_proj = P @ A_centered + A_mean

print("\nProjected Matrix (first 5 rows):")
print(pd.DataFrame(A_proj, index=years, columns=city_labels).head())

cwr("Projection", "Denoise data", "Projected onto clean subspace")


# STEP 7: LEAST SQUARES
section("STEP 7: LEAST SQUARES")

t = (years - years.mean()) / years.std()
X = np.column_stack([np.ones_like(t), t, t**2])

print("\nDesign Matrix X (first 5 rows):")
print(pd.DataFrame(X).head())

Beta = np.linalg.lstsq(X, A, rcond=None)[0]
A_pred = X @ Beta

print("\nPredicted Matrix (first 5 rows):")
print(pd.DataFrame(A_pred, index=years, columns=city_labels).head())

cwr("Least Squares", "Predict trends", "Best-fit polynomial model created")

# FUTURE YEARS (HARDCODED)
future_years = np.arange(2024, 2051)
t_future = (future_years - years.mean()) / years.std()
X_future = np.column_stack([np.ones_like(t_future), t_future, t_future**2])
A_future = X_future @ Beta


# STEP 8: EIGEN ANALYSIS
section("STEP 8: EIGEN ANALYSIS")

C = np.cov(A_centered.T)
eigvals, eigvecs = np.linalg.eigh(C)

eigvals = eigvals[::-1]
eigvecs = eigvecs[:, ::-1]

print("\nCovariance Matrix:")
print(pd.DataFrame(C, index=city_labels, columns=city_labels))

print("\nEigenvalues:", eigvals)

explained = 100 * eigvals / eigvals.sum()

cwr("Eigenvalues", "Find dominant trends", f"PC1 explains {explained[0]:.2f}% variance")


# STEP 9: FINAL OUTPUT
section("STEP 9: FINAL OUTPUT")

out_df = pd.DataFrame(A_future, index=future_years, columns=city_labels)
out_df.to_csv("future_predictions.csv")

print(f"\nPredictions for {focus_city}:")
for i, year in enumerate(future_years):
    print(f"{year}: {A_future[i, focus_idx]:.2f} °C")

cwr("Final Output", "Provide predictions", "CSV file generated + results displayed")


# VISUALIZATION
section("VISUALIZATION")

fig, axs = plt.subplots(3, 1, figsize=(10, 12))

axs[0].plot(years, A[:, focus_idx])
axs[0].set_title("Actual Temperature Trend")

axs[1].plot(years, A[:, focus_idx], label="Actual")
axs[1].plot(years, A_pred[:, focus_idx], label="Predicted")
axs[1].legend()
axs[1].set_title("Actual vs Predicted")

axs[2].plot(explained)
axs[2].set_title("Variance Explained")

plt.tight_layout()
plt.show()


# CONCLUSION
section("CONCLUSION")

print("Temperature shows increasing trend.")
print("Predictions indicate future rise.")
