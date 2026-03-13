"""
=============================================================================
     Global Temperature Trend Analysis using Linear Algebra
=============================================================================
Dataset : GlobalLandTemperaturesByMajorCity.csv
          (Monthly average temperatures for major cities worldwide)

Linear Algebra Pipeline:
  1.  Data Acquisition & Preprocessing
  2.  Matrix Representation
  3.  Gaussian Elimination & RREF
  4.  Vector Space Analysis  (Rank, Column Space, Null Space)
  5.  Linearly Independent Basis  (Remove Redundancy)
  6.  Gram-Schmidt Orthogonalisation
  7.  Projection onto Orthogonal Subspace
  8.  Least-Squares Regression  (Trend & Prediction)
  9.  Eigenvalue / PCA Analysis  (Pattern Discovery)
  10. Visualisation & Final Output
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    """Print a prominent section banner."""
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def subsection(title: str) -> None:
    """Print a sub-section header."""
    print(f"\n--- {title} ---")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA ACQUISITION
# ═════════════════════════════════════════════════════════════════════════════
section("STEP 1: DATA ACQUISITION")

# Load the raw dataset
df = pd.read_csv("GlobalLandTemperaturesByMajorCity.csv", parse_dates=["dt"])
print(f"Loaded  : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Columns : {list(df.columns)}")
print("\nFirst 3 rows:")
print(df.head(3).to_string(index=False))

# Drop rows with missing temperature readings
df = df.dropna(subset=["AverageTemperature"])
print(f"\nAfter dropping NaN temperatures: {df.shape[0]:,} rows remain")

# Extract calendar year from the date column
df["Year"] = df["dt"].dt.year

# ─────────────────────────────────────────────────────────────────────────────
# Choose 8 geographically diverse cities for the matrix columns.
# We only retain cities that appear in the dataset AND have at least 100 years
# of annual records so the matrix is well-populated.
# ─────────────────────────────────────────────────────────────────────────────
CANDIDATE_CITIES = [
    "London", "New York", "Mumbai", "Beijing",
    "Cairo",  "Sydney",  "Moscow", "Tokyo",
]

city_year_counts = df.groupby("City")["Year"].nunique()
available = set(city_year_counts[city_year_counts >= 100].index)
CITIES = [c for c in CANDIDATE_CITIES if c in available]

print(f"\nSelected cities ({len(CITIES)}): {CITIES}")

# Compute annual average temperature per city (mean over all months of each year)
annual = (
    df[df["City"].isin(CITIES)]
    .groupby(["Year", "City"])["AverageTemperature"]
    .mean()
    .unstack("City")   # rows = Year, columns = City
)

# Retain only years where every selected city has a value
annual = annual.dropna()
print(f"\nClean matrix shape : {annual.shape}  (years × cities)")
print(f"Year range         : {annual.index.min()} – {annual.index.max()}")
print("\nSample rows:")
print(annual.head(5).round(3).to_string())


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — MATRIX REPRESENTATION
# ═════════════════════════════════════════════════════════════════════════════
section("STEP 2: MATRIX REPRESENTATION")

years = annual.index.values.astype(float)   # shape (n,)
A     = annual.values.astype(float)         # shape (n × m),  n=years, m=cities

print(f"\nMatrix A — shape : {A.shape}")
print(f"  • Each ROW    represents one year's temperatures across all cities.")
print(f"  • Each COLUMN represents one city's temperature time-series.")

# Centre the matrix by subtracting each city's long-run mean.
# This removes the baseline "climate offset" and keeps only the variation.
A_mean     = A.mean(axis=0)            # shape (m,)
A_centered = A - A_mean               # shape (n × m)

print("\nColumn (city) mean temperatures  [°C]:")
for city, mean_val in zip(annual.columns, A_mean):
    print(f"  {city:12s}: {mean_val:.2f} °C")

print(f"\nA_centered — first 5 rows (mean-subtracted):\n{A_centered[:5].round(3)}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — GAUSSIAN ELIMINATION & RREF
# ═════════════════════════════════════════════════════════════════════════════
section("STEP 3: GAUSSIAN ELIMINATION & ROW REDUCED ECHELON FORM (RREF)")


def gaussian_elimination(M: np.ndarray) -> np.ndarray:
    """
    Perform Gaussian Elimination (forward pass only) on matrix M.

    Algorithm:
      For each column, find the row with the largest absolute value (partial
      pivoting for numerical stability), swap it to the current pivot row,
      then eliminate all entries below the pivot.

    Returns the upper-triangular form of M.
    """
    M = M.astype(float).copy()
    nrows, ncols = M.shape
    pivot_row = 0

    for col in range(ncols):
        # ── Partial pivoting: find row with largest |value| in this column
        sub_col   = np.abs(M[pivot_row:, col])
        max_local = np.argmax(sub_col)
        max_row   = max_local + pivot_row

        if np.isclose(M[max_row, col], 0.0, atol=1e-9):
            continue   # entire column (below pivot) is zero → skip

        # ── Swap pivot row into position
        M[[pivot_row, max_row]] = M[[max_row, pivot_row]]

        # ── Eliminate all rows below the pivot
        for row in range(pivot_row + 1, nrows):
            if not np.isclose(M[pivot_row, col], 0.0, atol=1e-9):
                factor    = M[row, col] / M[pivot_row, col]
                M[row]   -= factor * M[pivot_row]

        pivot_row += 1
        if pivot_row >= nrows:
            break

    return M


def rref(M: np.ndarray, tol: float = 1e-9):
    """
    Compute the Row Reduced Echelon Form (RREF) of matrix M.

    Steps:
      1. Forward pass  — same as Gaussian elimination (pivot + eliminate below).
      2. Backward pass — eliminate entries ABOVE each pivot.
      3. Scale         — make each pivot entry equal to 1.

    Returns:
      R     : RREF matrix.
      pivots: list of column indices that contain a leading 1.
    """
    M = M.astype(float).copy()
    nrows, ncols = M.shape
    pivot_row = 0
    pivots    = []

    for col in range(ncols):
        # ── Find pivot in this column at or below current pivot_row
        sub_col   = np.abs(M[pivot_row:, col])
        max_local = np.argmax(sub_col)
        max_row   = max_local + pivot_row

        if np.abs(M[max_row, col]) < tol:
            continue   # no non-zero entry → free variable column

        # ── Swap
        M[[pivot_row, max_row]] = M[[max_row, pivot_row]]

        # ── Scale pivot row so that the pivot entry = 1
        M[pivot_row] = M[pivot_row] / M[pivot_row, col]

        # ── Eliminate all OTHER rows (above AND below)
        for row in range(nrows):
            if row != pivot_row:
                M[row] -= M[row, col] * M[pivot_row]

        pivots.append(col)
        pivot_row += 1
        if pivot_row >= nrows:
            break

    return M, pivots


# ── Apply to a small sub-matrix (first 10 years) for readable output
A_sub = A_centered[:10, :]
print(f"\nSub-matrix A_sub (first 10 years × {A_sub.shape[1]} cities)")

# Gaussian Elimination
GE = gaussian_elimination(A_sub)
print("\nGaussian Elimination result (upper-triangular form):")
print(np.round(GE, 4))

# RREF on A_sub.T  — treats each CITY vector as a "row"
# → reveals which cities are linearly dependent on others
R_sub, pivot_cols_sub = rref(A_sub.T)
print(f"\nRREF of A_sub.T  |  Pivot columns: {pivot_cols_sub}")
print(f"Number of pivots: {len(pivot_cols_sub)}  "
      f"→ {len(pivot_cols_sub)} linearly independent city-rows in the sub-matrix")
print("\nRREF matrix (first pivot+1 rows shown):")
print(np.round(R_sub[:len(pivot_cols_sub) + 1, :], 4))


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — VECTOR SPACE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
section("STEP 4: VECTOR SPACE ANALYSIS  (Rank · Column Space · Null Space)")

# ── Singular Value Decomposition gives us everything in one shot:
#    A = U Σ V^T
#    • U  : left singular vectors  (column space basis)
#    • Σ  : singular values        (tells us rank)
#    • V^T: right singular vectors (row space / null space)
U, S, Vt = np.linalg.svd(A_centered, full_matrices=True)

tol_rank = S.max() * max(A_centered.shape) * np.finfo(float).eps
rank     = int(np.sum(S > tol_rank))

print(f"\nMatrix A_centered  shape : {A_centered.shape}")
print(f"Singular values          : {np.round(S, 4)}")
print(f"Rank threshold (ε)       : {tol_rank:.2e}")
print(f"Rank of A                : {rank}")

subsection("Column Space  (range of A)")
col_space_basis = U[:, :rank]
print(f"Dimension : {rank}  (= number of independent temperature patterns across years)")
print(f"Basis shape : {col_space_basis.shape}  — each column is a year-direction vector")
print("Interpretation: any temperature observation lies in (or near) this {}-D subspace.".format(rank))

subsection("Null Space  (kernel of A)")
null_space = Vt[rank:, :].T   # columns of V corresponding to near-zero singular values
null_dim   = null_space.shape[1]
if null_dim == 0:
    print(f"Null space dimension : 0")
    print("→ The city columns are linearly INDEPENDENT — no city is a linear combination of others.")
else:
    print(f"Null space dimension : {null_dim}")
    print("→ There exist non-trivial city combinations that produce zero temperature variation.")
    print(f"\nNull space basis vectors (shape {null_space.shape}):")
    print(np.round(null_space, 4))

subsection("Interpretation")
n_cities = A_centered.shape[1]
print(f"  Rank {rank}  →  {rank} independent temperature patterns.")
print(f"  Nullity {n_cities - rank}  →  {n_cities - rank} dimension(s) of redundancy in the city data.")
print(f"  The column space is a {rank}-D subspace of R^{A_centered.shape[0]}.")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — LINEARLY INDEPENDENT BASIS  (Remove Redundancy)
# ═════════════════════════════════════════════════════════════════════════════
section("STEP 5: LINEARLY INDEPENDENT BASIS  (Removing Redundancy)")

# Run RREF on A_centered.T so that each row corresponds to a city.
# Pivot columns of A.T reveal which CITIES are linearly independent.
R_full, pivot_city_idx = rref(A_centered.T)

city_labels       = list(annual.columns)
independent_cities = [city_labels[i] for i in pivot_city_idx]

print(f"Pivot city column indices : {pivot_city_idx}")
print(f"Linearly independent cities: {independent_cities}")

# B is the basis matrix — contains only the independent city columns
B = A_centered[:, pivot_city_idx]
print(f"\nBasis matrix B  shape : {B.shape}  (years × {len(pivot_city_idx)} independent cities)")
print("These columns span the same column space as A_centered with no redundancy.")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — GRAM-SCHMIDT ORTHOGONALISATION
# ═════════════════════════════════════════════════════════════════════════════
section("STEP 6: GRAM–SCHMIDT ORTHOGONALISATION")


def gram_schmidt(V: np.ndarray) -> np.ndarray:
    """
    Classical Gram-Schmidt process.

    Converts a set of linearly independent column vectors in V into an
    orthonormal set Q such that Q^T Q = I and span(Q) = span(V).

    For each vector v_j:
      1. Subtract projections onto all previously computed basis vectors:
            u_j = v_j - Σ_{i<j} proj_{q_i}(v_j)
            where proj_{q}(v) = (q · v) * q   (since q is already unit-length)
      2. Normalise:  q_j = u_j / ‖u_j‖

    Parameters
    ----------
    V : (n × k) matrix whose columns are the vectors to orthogonalise.

    Returns
    -------
    Q : (n × k) matrix with orthonormal columns.
    """
    n, k = V.shape
    Q = np.zeros_like(V, dtype=float)

    for j in range(k):
        q = V[:, j].copy().astype(float)

        # ── Subtract projection onto each previously found basis vector
        for i in range(j):
            projection = np.dot(Q[:, i], V[:, j]) * Q[:, i]
            q -= projection

        # ── Normalise
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            Q[:, j] = q / norm
        else:
            # Vector is linearly dependent on previous ones — insert zero column
            Q[:, j] = np.zeros(n)

    return Q


Q = gram_schmidt(B)
print(f"\nOrthonormal basis Q  shape : {Q.shape}")
print(f"Each column is a unit vector: {np.allclose(np.linalg.norm(Q, axis=0), 1.0)}")

# ── Orthogonality verification  →  Q^T Q should equal I_k
QTQ = Q.T @ Q
print(f"\nOrthogonality check  Q^T · Q  (should be identity matrix):")
print(np.round(QTQ, 6))
is_ortho = np.allclose(QTQ, np.eye(Q.shape[1]), atol=1e-6)
print(f"Orthonormality verified : {is_ortho}")

# ── Cross-check with NumPy's built-in QR decomposition
Q_np, _ = np.linalg.qr(B)
# Signs may differ, so compare absolute values
max_diff = np.max(np.abs(np.abs(Q) - np.abs(Q_np[:, :Q.shape[1]])))
print(f"Max |difference| vs NumPy QR : {max_diff:.2e}  (near zero → consistent)")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — PROJECTION ONTO ORTHOGONAL SUBSPACE
# ═════════════════════════════════════════════════════════════════════════════
section("STEP 7: PROJECTION ONTO ORTHOGONAL SUBSPACE")

# ── Projection matrix:  P = Q Q^T
#    For any vector y,  P y  is the closest point to y inside the subspace.
#    P is symmetric (P = P^T) and idempotent (P² = P).
P = Q @ Q.T
print(f"Projection matrix P  shape : {P.shape}")

# ── Verify idempotency:  P² = P
idempotency_err = np.linalg.norm(P @ P - P)
print(f"Idempotency check  ‖P² – P‖ = {idempotency_err:.2e}  (should be ≈ 0)")

# ── Project A_centered onto the subspace
#    A_projected ≈ A_centered  if A_centered lies within the subspace.
#    The "residual" is the out-of-subspace noise / unexplained variation.
A_projected = P @ A_centered      # best approximation inside the subspace
residual     = A_centered - A_projected

print(f"\n‖Original   A_centered‖  = {np.linalg.norm(A_centered):.4f}")
print(f"‖Projected  A_centered‖  = {np.linalg.norm(A_projected):.4f}")
print(f"‖Residual              ‖  = {np.linalg.norm(residual):.4f}")
frac_captured = np.linalg.norm(A_projected) / np.linalg.norm(A_centered)
print(f"Fraction of signal captured by subspace : {100*frac_captured:.1f}%")

print(f"\nSample comparison for {city_labels[0]} (first 5 years):")
print(f"{'Year':>6}  {'Original':>10}  {'Projected':>10}  {'Residual':>10}")
for yr, orig, proj in zip(years[:5], A_centered[:5, 0], A_projected[:5, 0]):
    print(f"{int(yr):>6}  {orig:>10.4f}  {proj:>10.4f}  {orig - proj:>10.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 — LEAST-SQUARES REGRESSION  (Trend & Prediction)
# ═════════════════════════════════════════════════════════════════════════════
section("STEP 8: LEAST-SQUARES REGRESSION FOR TREND PREDICTION")

# ── Build design matrix X for polynomial regression of degree 2:
#       y ≈ β₀ + β₁·t + β₂·t²
#
#    We normalise time to [-~3, +~3] for numerical stability.
t        = (years - years.mean()) / years.std()   # standardised time axis
X_design = np.column_stack([
    np.ones_like(t),   # β₀  (intercept)
    t,                 # β₁  (linear  trend)
    t ** 2,            # β₂  (quadratic curvature)
])
print(f"\nDesign matrix X  shape : {X_design.shape}  columns = [1, t, t²]")

# ── Solve:  X β ≈ A    (over-determined; one solution per city column)
#    Normal equations:  β = (X^T X)^{-1} X^T A
#
#    numpy.linalg.lstsq solves this efficiently using SVD internally.
Beta, residuals_ls, ls_rank, ls_sv = np.linalg.lstsq(X_design, A, rcond=None)
# Beta shape: (3 × m)

print(f"Coefficient matrix β  shape : {Beta.shape}  (3 coefficients × {len(city_labels)} cities)")
print("\nLeast-squares coefficients per city:")
print(f"  {'City':12s}  {'β₀ (intercept)':>16}  {'β₁ (linear)':>13}  {'β₂ (quadratic)':>15}")
for city, b in zip(city_labels, Beta.T):
    print(f"  {city:12s}  {b[0]:>16.4f}  {b[1]:>13.4f}  {b[2]:>15.4f}")

# ── Fitted values over the historical range
A_fitted = X_design @ Beta   # shape (n × m)

# ── Residual quality metrics
res_all = A - A_fitted
mae  = np.mean(np.abs(res_all))
rmse = np.sqrt(np.mean(res_all ** 2))
print(f"\nResidual statistics (across all cities & years):")
print(f"  Mean Absolute Error (MAE) : {mae:.4f} °C")
print(f"  Root Mean Square Error    : {rmse:.4f} °C")

# ── Predict temperatures for future years 2024–2040
future_years = np.arange(2024, 2041, dtype=float)
t_future     = (future_years - years.mean()) / years.std()
X_future     = np.column_stack([np.ones_like(t_future), t_future, t_future ** 2])
A_future     = X_future @ Beta   # shape (len(future_years) × m)

print("\nPredicted annual average temperatures [°C]:")
pred_df = pd.DataFrame(A_future, index=future_years.astype(int), columns=city_labels)
print(pred_df.loc[[2025, 2030, 2035, 2040]].round(2).to_string())


# ═════════════════════════════════════════════════════════════════════════════
# STEP 9 — EIGENVALUE / PCA ANALYSIS  (Pattern Discovery)
# ═════════════════════════════════════════════════════════════════════════════
section("STEP 9: PATTERN DISCOVERY USING EIGENVALUES & EIGENVECTORS (PCA)")

# ── Covariance matrix  C = (1 / n-1) · A_centered^T · A_centered
#    Shape: (m × m) — captures how cities' temperatures co-vary.
n_years  = A_centered.shape[0]
C = (A_centered.T @ A_centered) / (n_years - 1)
print(f"\nCovariance matrix C  shape : {C.shape}")

# ── Eigendecomposition.
#    numpy.linalg.eigh is used because C is symmetric (Hermitian).
#    Returns eigenvalues in ascending order → we reverse for descending.
eigenvalues, eigenvectors = np.linalg.eigh(C)
idx          = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("\nEigenvalues and variance explained per Principal Component:")
total_var = eigenvalues.sum()
cumvar    = 0.0
for i, ev in enumerate(eigenvalues):
    cumvar += ev
    print(f"  PC{i+1}: λ = {ev:8.4f}  |  "
          f"var = {100*ev/total_var:5.1f}%  |  "
          f"cumulative = {100*cumvar/total_var:5.1f}%")

# ── Project data onto principal components  (score matrix)
PCA_scores = A_centered @ eigenvectors   # shape (n × m)
print(f"\nPC score matrix  shape : {PCA_scores.shape}  (years × components)")
print("\nFirst 5 year-rows of PC scores:")
score_df = pd.DataFrame(
    PCA_scores[:5, :],
    index=years[:5].astype(int),
    columns=[f"PC{i+1}" for i in range(PCA_scores.shape[1])],
)
print(score_df.round(4).to_string())

# ── PC1 loadings tell us which cities drive the dominant pattern
print("\nPC1 eigenvector loadings  (dominant global temperature pattern per city):")
for city, loading in zip(city_labels, eigenvectors[:, 0]):
    bar = "█" * int(abs(loading) * 40)
    sign = "+" if loading >= 0 else "-"
    print(f"  {city:12s}: {sign}{abs(loading):.4f}  {bar}")

print("\nInterpretation:")
print("  PC1 captures the dominant correlated warming/cooling across all cities.")
print("  High |loading| → that city strongly participates in the global trend.")
print(f"  PC1 alone explains {100*eigenvalues[0]/total_var:.1f}% of total temperature variance.")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 10 — VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════
section("STEP 10: VISUALISATION & FINAL OUTPUT")

plt.style.use("seaborn-v0_8-darkgrid")

fig = plt.figure(figsize=(22, 28))
fig.suptitle(
    "Global Temperature Trend Analysis using Linear Algebra\n"
    f"Dataset: Major Cities  ·  Years: {int(years.min())} – {int(years.max())}",
    fontsize=17, fontweight="bold", y=0.99,
)

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.62, wspace=0.35)
colors = plt.cm.tab10(np.linspace(0, 1, len(city_labels)))

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Historical annual average temperatures  (full width)
# ─────────────────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
for i, city in enumerate(city_labels):
    ax1.plot(years, A[:, i], alpha=0.65, linewidth=0.9,
             label=city, color=colors[i])
ax1.set_title("Historical Annual Average Temperatures by City", fontsize=13, fontweight="bold")
ax1.set_xlabel("Year", fontsize=11)
ax1.set_ylabel("Temperature (°C)", fontsize=11)
ax1.legend(fontsize=8, ncol=4, loc="upper left")
ax1.annotate("Raw data from dataset", xy=(0.99, 0.02),
             xycoords="axes fraction", ha="right", fontsize=8, color="gray")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Least-squares polynomial fit
# ─────────────────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
for i, city in enumerate(city_labels):
    ax2.scatter(years, A[:, i], s=3, alpha=0.25, color=colors[i])
    ax2.plot(years, A_fitted[:, i], linewidth=1.8,
             label=city, color=colors[i])
ax2.set_title("Least-Squares Polynomial Fit (Degree 2)", fontsize=12, fontweight="bold")
ax2.set_xlabel("Year", fontsize=10)
ax2.set_ylabel("Temperature (°C)", fontsize=10)
ax2.legend(fontsize=7, ncol=2)
ax2.text(0.02, 0.97, f"MAE = {mae:.3f} °C\nRMSE = {rmse:.3f} °C",
         transform=ax2.transAxes, va="top", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Future temperature prediction  (last 30 historic years + forecast)
# ─────────────────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
last_30 = -30
for i, city in enumerate(city_labels):
    ax3.plot(years[last_30:], A[last_30:, i],
             linewidth=1.5, color=colors[i], label=city)
    ax3.plot(future_years, A_future[:, i],
             linewidth=1.8, linestyle="--", color=colors[i], alpha=0.85)
ax3.axvline(x=years[-1], color="black", linestyle=":", linewidth=1.5,
            label="Forecast boundary")
ax3.fill_betweenx(
    [ax3.get_ylim()[0] if ax3.get_ylim()[0] != 0 else A.min() - 2,
     A.max() + 2],
    future_years[0], future_years[-1],
    alpha=0.07, color="orange", label="Forecast zone",
)
ax3.set_title("Temperature Forecast  2024–2040\n(Dashed = Prediction)", fontsize=12, fontweight="bold")
ax3.set_xlabel("Year", fontsize=10)
ax3.set_ylabel("Temperature (°C)", fontsize=10)
ax3.legend(fontsize=6, ncol=2)

# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Singular values (visualising rank / column space)
# ─────────────────────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
bar_colors_sv = ["steelblue" if s > tol_rank else "lightgray" for s in S]
ax4.bar(range(1, len(S) + 1), S, color=bar_colors_sv, edgecolor="black", alpha=0.85)
ax4.axhline(y=tol_rank, color="red", linestyle="--", linewidth=1.5,
            label=f"Tolerance threshold")
ax4.set_title(f"Singular Values of A_centered  (Rank = {rank})", fontsize=12, fontweight="bold")
ax4.set_xlabel("Component Index", fontsize=10)
ax4.set_ylabel("Singular Value", fontsize=10)
ax4.legend(fontsize=9)
ax4.text(0.97, 0.95, f"Blue bars → rank {rank}\nGray bars → null space",
         transform=ax4.transAxes, ha="right", va="top", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: PCA scree plot
# ─────────────────────────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
exp_var = 100 * eigenvalues / total_var
cum_var = np.cumsum(exp_var)
x_ticks = range(1, len(eigenvalues) + 1)

ax5.bar(x_ticks, exp_var, color="coral", edgecolor="black", alpha=0.85, label="Individual %")
ax5_twin = ax5.twinx()
ax5_twin.plot(x_ticks, cum_var, "k-o", markersize=6, linewidth=1.5, label="Cumulative %")
ax5_twin.set_ylabel("Cumulative Variance Explained (%)", fontsize=10)
ax5_twin.set_ylim(0, 110)
ax5_twin.axhline(y=90, color="green", linestyle=":", linewidth=1.2, alpha=0.7)
ax5.set_title("PCA — Eigenvalue Scree Plot", fontsize=12, fontweight="bold")
ax5.set_xlabel("Principal Component", fontsize=10)
ax5.set_ylabel("Variance Explained (%)", fontsize=10)
ax5.legend(loc="upper left", fontsize=9)
ax5_twin.legend(loc="center right", fontsize=9)

# ─────────────────────────────────────────────────────────────────────────────
# Plot 6: PC1 vs PC2 score plot  (temporal trajectory coloured by year)
# ─────────────────────────────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[3, 0])
sc = ax6.scatter(PCA_scores[:, 0], PCA_scores[:, 1],
                 c=years, cmap="plasma", s=18, alpha=0.85, edgecolors="none")
cbar = plt.colorbar(sc, ax=ax6)
cbar.set_label("Year", fontsize=9)
ax6.set_title("PCA Score Plot: PC1 vs PC2\n(Colour encodes Year)", fontsize=12, fontweight="bold")
ax6.set_xlabel(f"PC1  ({exp_var[0]:.1f}% var)", fontsize=10)
ax6.set_ylabel(f"PC2  ({exp_var[1]:.1f}% var)", fontsize=10)

# ─────────────────────────────────────────────────────────────────────────────
# Plot 7: PC1 loadings (dominant climate pattern)
# ─────────────────────────────────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[3, 1])
loadings  = eigenvectors[:, 0]
bar_clrs  = ["#2ca02c" if l > 0 else "#d62728" for l in loadings]
ax7.barh(city_labels, loadings, color=bar_clrs, edgecolor="black", alpha=0.85)
ax7.axvline(0, color="black", linewidth=0.9)
ax7.set_title(f"PC1 Eigenvector Loadings\n(Dominant Global Temperature Pattern  —  {exp_var[0]:.1f}% variance)",
              fontsize=12, fontweight="bold")
ax7.set_xlabel("Loading Value", fontsize=10)
for bar_rect, val in zip(ax7.patches, loadings):
    ax7.text(val + 0.002 * np.sign(val), bar_rect.get_y() + bar_rect.get_height() / 2,
             f"{val:+.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)

plt.savefig("temperature_analysis_results.png", dpi=150, bbox_inches="tight")
print("Plot saved → temperature_analysis_results.png")
plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
section("PROJECT SUMMARY — Linear Algebra Concepts Applied")

summary = f"""
  Dataset   : GlobalLandTemperaturesByMajorCity.csv
  Cities    : {city_labels}
  Year span : {int(years.min())} – {int(years.max())}  ({len(years)} years)
  Matrix A  : {A.shape[0]} × {A.shape[1]}  (years × cities)

  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP  │  Concept                    │  Result / Key Finding        │
  ├─────────────────────────────────────────────────────────────────────┤
  │   2    │  Matrix Representation      │  A ∈ R^({A.shape[0]}×{A.shape[1]})                      │
  │   3    │  Gaussian Elim. + RREF      │  Pivot cols → {pivot_city_idx}            │
  │   4    │  Rank / Null Space          │  rank(A) = {rank},  nullity = {A.shape[1]-rank}            │
  │   5    │  Independent Basis          │  {len(pivot_city_idx)} independent city vectors          │
  │   6    │  Gram-Schmidt               │  Orthonormal Q ∈ R^({B.shape[0]}×{Q.shape[1]})          │
  │   7    │  Projection                 │  {100*frac_captured:.1f}% signal captured              │
  │   8    │  Least Squares (β₀+β₁t+β₂t²)│  RMSE = {rmse:.3f} °C               │
  │   9    │  Eigenvalues / PCA          │  PC1 = {exp_var[0]:.1f}% variance            │
  │  10    │  Visualisation              │  Saved → results.png          │
  └─────────────────────────────────────────────────────────────────────┘
"""
print(summary)
