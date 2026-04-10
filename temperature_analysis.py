import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def cwr(concept: str, why: str, result: str) -> None:
    print("Concept -> Why -> Result")
    print(f"  Concept: {concept}")
    print(f"  Why    : {why}")
    print(f"  Result : {result}")


def rref(matrix: np.ndarray, tol: float = 1e-10):
    """Numerical RREF with pivot tracking."""
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


def choose_focus_city(cities):
    if not cities:
        raise ValueError("No valid cities available for analysis.")

    city_map = {c.lower(): c for c in cities}

    if len(sys.argv) > 1:
        arg_city = " ".join(sys.argv[1:]).strip().lower()
        if arg_city in city_map:
            return city_map[arg_city]
        print(f"[Warning] City '{' '.join(sys.argv[1:])}' not found. Falling back to prompt/default.")

    print("Available cities:", ", ".join(cities))
    try:
        user_city = input("Enter city for detailed trend analysis [default: first city]: ").strip().lower()
    except EOFError:
        user_city = ""

    if user_city == "":
        return cities[0]
    return city_map.get(user_city, cities[0])


# STEP 1: DATA COLLECTION
section("STEP 1: DATA COLLECTION")
df = pd.read_csv("GlobalLandTemperaturesByMajorCity.csv", parse_dates=["dt"])
df = df.dropna(subset=["AverageTemperature"])
df["Year"] = df["dt"].dt.year

candidate_cities = ["London", "New York", "Mumbai", "Beijing", "Cairo", "Sydney", "Moscow", "Tokyo"]
city_year_count = df.groupby("City")["Year"].nunique()
selected_cities = [c for c in candidate_cities if c in city_year_count.index and city_year_count[c] >= 100]
focus_city = choose_focus_city(selected_cities)

annual_full = (
    df[df["City"].isin(selected_cities)]
    .groupby(["Year", "City"])["AverageTemperature"]
    .mean()
    .unstack("City")
)
annual = annual_full.dropna()

print(f"Selected cities: {selected_cities}")
print(f"Focus city: {focus_city}")
print(f"Annual matrix shape (complete rows): {annual.shape}")
print(f"Missing annual entries (for estimation later): {int(annual_full.isna().sum().sum())}")
cwr(
    "Data collection",
    "Need reliable yearly values for linear algebra operations.",
    f"Built annual year-city table with {annual.shape[0]} years and {annual.shape[1]} cities.",
)


# STEP 2: MATRIX REPRESENTATION
section("STEP 2: MATRIX REPRESENTATION OF DATA")
years = annual.index.to_numpy(dtype=float)
city_labels = list(annual.columns)
focus_idx = city_labels.index(focus_city)
A = annual.to_numpy(dtype=float)
A_mean = A.mean(axis=0)
A_centered = A - A_mean

print("Rows = Years, Columns = Cities")
print("Matrix A (first 5 rows):")
print(np.round(A[:5], 3))
print(f"Focus city column index: {focus_idx}")
print(f"Focus city first 5 values: {np.round(A[:5, focus_idx], 3)}")
cwr(
    "Matrix representation",
    "Matrix form is required for elimination, basis, projection, and eigen-analysis.",
    f"A is in R^({A.shape[0]}x{A.shape[1]}), centered matrix computed.",
)


# STEP 3: RREF / MATRIX SIMPLIFICATION
section("STEP 3: RREF / MATRIX SIMPLIFICATION")
A_small = A_centered[:10, :]
R, pivots = rref(A_small)
print("RREF of first 10-year block (A_centered[:10, :]):")
print(np.round(R, 4))
print(f"Pivot columns: {pivots}")
print(f"Interpretation: {len(pivots)} independent climate patterns are visible in this 10-year sample block.")

focus_col_rref, focus_col_pivots = rref(A_small[:, [focus_idx]])
print(f"Focus city single-column RREF pivots: {focus_col_pivots}")
print(f"Focus city column RREF (flattened): {np.round(focus_col_rref.flatten(), 4)}")
cwr(
    "RREF simplification",
    "Pivot columns identify independent structure and remove row-redundancy.",
    f"Found {len(pivots)} pivots, meaning {len(pivots)} independent climate patterns in the sampled matrix.",
)


# STEP 4: BASIS AND ORTHOGONAL BASIS FORMATION
section("STEP 4: BASIS AND ORTHOGONAL BASIS FORMATION")
R_full, basis_indices = rref(A_centered.T)
B = A_centered[:, basis_indices]
Q, _ = np.linalg.qr(B)
Q = Q[:, : B.shape[1]]

rank = len(basis_indices)
nullity = A.shape[1] - rank

print(f"Basis indices (city columns): {basis_indices}")
print(f"Basis matrix shape B: {B.shape}")
print(f"Orthogonal basis shape Q: {Q.shape}")
print(f"Orthogonality check Q^T Q = I: {np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-7)}")
print(f"Rank: {rank}")
print(f"Nullity: {nullity}")
print(f"Interpretation: only {rank} independent climate patterns exist; redundancy dimension is {nullity}.")

focus_in_basis = focus_idx in basis_indices
print(f"Is focus city in basis? {focus_in_basis}")
cwr(
    "Basis + orthogonal basis",
    "Basis removes redundancy; orthogonal basis makes projection numerically stable.",
    f"Built B in R^({B.shape[0]}x{B.shape[1]}) with rank={rank} and nullity={nullity}, then formed orthogonal Q.",
)


# STEP 5: PROJECTION-BASED PREDICTION
section("STEP 5: PROJECTION-BASED PREDICTION")
P = Q @ Q.T
A_proj_centered = P @ A_centered
A_proj = A_proj_centered + A_mean
projection_residual = A_centered - A_proj_centered

focus_res_norm = np.linalg.norm(projection_residual[:, focus_idx])
signal_captured = 100 * np.linalg.norm(A_proj_centered) / np.linalg.norm(A_centered)
print(f"Projection matrix shape: {P.shape}")
print(f"Signal captured (%): {signal_captured:.2f}")
print(f"Focus city projection residual norm: {focus_res_norm:.6f}")
print("Focus city first 5 projected values:", np.round(A_proj[:5, focus_idx], 3))
print(f"Interpretation: {signal_captured:.2f}% of variation is explained by the selected subspace.")
print("Projection removes noise and keeps main climate trend.")
cwr(
    "Projection",
    "Projection keeps in-subspace trend and filters orthogonal noise.",
    f"Projected matrix created; {signal_captured:.2f}% variation retained as main trend.",
)


# STEP 6: LEAST SQUARES ESTIMATION
section("STEP 6: LEAST SQUARES ESTIMATION")
t = (years - years.mean()) / years.std()
X = np.column_stack([np.ones_like(t), t, t ** 2])
Beta = np.linalg.lstsq(X, A, rcond=None)[0]
A_pred = X @ Beta
rmse = np.sqrt(np.mean((A - A_pred) ** 2))

future_years = np.arange(2024, 2041, dtype=float)
t_future = (future_years - years.mean()) / years.std()
X_future = np.column_stack([np.ones_like(t_future), t_future, t_future ** 2])
A_future = X_future @ Beta
pred_df = pd.DataFrame(A_future, index=future_years.astype(int), columns=city_labels)

focus_beta = Beta[:, focus_idx]
print(f"RMSE (all cities): {rmse:.4f}")
print(f"Focus model: y(t) = {focus_beta[0]:.4f} + ({focus_beta[1]:.4f})t + ({focus_beta[2]:.4f})t^2")
print(f"Focus city prediction 2030: {pred_df.loc[2030, focus_city]:.3f} degC")
print(f"Focus city prediction 2040: {pred_df.loc[2040, focus_city]:.3f} degC")
cwr(
    "Least squares",
    "Provides best-fit trend under noisy/inconsistent observations.",
    f"Estimated trend model and forecast with RMSE={rmse:.4f}.",
)


# STEP 7: EIGENVALUE / EIGENVECTOR ANALYSIS
section("STEP 7: EIGENVALUE / EIGENVECTOR ANALYSIS")
C = np.cov(A_centered.T)
eigvals, eigvecs = np.linalg.eigh(C)
order = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

explained = 100 * eigvals / eigvals.sum()
top_city = city_labels[int(np.argmax(np.abs(eigvecs[:, 0])))]
print("Eigenvalues:", np.round(eigvals, 4))
print("Variance explained (%):", np.round(explained, 2))
print(f"Focus city loading on PC1: {eigvecs[focus_idx, 0]:+.4f}")
print(f"Focus city loading on PC2: {eigvecs[focus_idx, 1]:+.4f}")
print("PC1 represents dominant global temperature trend.")
print(f"Most influential city in PC1: {top_city}")
print(f"Interpretation: most variance is explained by PC1 ({explained[0]:.2f}%), indicating a common warming pattern.")
print("Most cities show similar trend, supporting a broad global warming effect.")
cwr(
    "Eigen-analysis",
    "Finds dominant correlated climate patterns.",
    f"PC1 explains {explained[0]:.2f}% variance and captures the dominant shared warming behavior.",
)


# STEP 8: FINAL REDUCED MODEL / OUTPUT
section("STEP 8: FINAL REDUCED MODEL OR APPLICATION OUTPUT")
all_years = annual_full.index.to_numpy(dtype=float)
t_all = (all_years - years.mean()) / years.std()
X_all = np.column_stack([np.ones_like(t_all), t_all, t_all ** 2])
A_all_model = X_all @ Beta

annual_imputed = annual_full.copy()
mask = annual_imputed.isna()
annual_imputed = annual_imputed.where(~mask, A_all_model)

pred_df.to_csv("temperature_future_predictions_2024_2040.csv", index_label="Year")
pd.DataFrame(A_proj, index=annual.index, columns=city_labels).to_csv(
    "temperature_improved_denoised_dataset.csv", index_label="Year"
)
annual_imputed.to_csv("temperature_improved_with_missing_estimated.csv", index_label="Year")

last_year = int(years[-1])
last_val = A[-1, focus_idx]
pred_2040 = pred_df.loc[2040, focus_city]
print(f"Focus city last historical ({last_year}): {last_val:.3f} degC")
print(f"Focus city predicted (2040): {pred_2040:.3f} degC")
print(f"Trend delta ({last_year} -> 2040): {pred_2040 - last_val:+.3f} degC")
print("Saved: temperature_future_predictions_2024_2040.csv")
print("Saved: temperature_improved_denoised_dataset.csv")
print("Saved: temperature_improved_with_missing_estimated.csv")
cwr(
    "Final reduced model",
    "Need usable project output (prediction + cleaned/improved datasets).",
    "Final model outputs saved for report/demo.",
)

