import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import textwrap
import re

# ===================== CONFIGURATION =====================
# Input file & columns
FILE_PATH = Path("tps_usable_hydrogen_storage_capacity_gcmcv2.xlsx")

# Targets (multi-target white-box)
TARGETS = ["UG at TPS ", "UV at TPS "]

# Core features to use (order matters for equation readability)
CORE_FEATURE_KEYS = ["Density ", "GSA ", "VSA ", "VF ", "PV ", "LCD ", "PLD "]

# White-box settings
DEGREE       = 2            # polynomial degree
TEST_SIZE    = 0.20         # 80/20 split (default used by run_whitebox_for_target)
RANDOM_STATE = 42
TOP_SCATTER  = 7            # show scatter

# DOE-style screening
DOE_ENABLE       = True
DOE_TARGET_GRAV  = 5.5      # wt.% (dipakai untuk UG at TPS)
DOE_TARGET_VOL   = 40.0     # g H2/L (dipakai untuk UV at TPS)
DOE_SHOW_PLOTS   = True

# Outputs
OUTDIR = Path("pipeline_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
# =========================================================

# -------------------- Aliases (& toleransi penamaan) --------------------
FEATURE_ALIASES = {
    "Density": [
        "Density", "density", "Density [g/cm^3]", "density [g/cm^3]", "density [g/cm³]"
    ],
    "GSA": [
        "GSA", "GSA [m^2/g]", "GSA [m2/g]", "grav_surface_area", "grav surface area"
    ],
    "VSA": [
        "VSA", "VSA [m^2/cm^3]", "VSA [m2/cm3]", "Volumetric SA [m^2/cm^3]", "vol_surface_area [m^2/cm^3]"
    ],
    "VF": [
        "VF", "void_fraction", "void fraction", "Void Fraction", "void_fraction [-]"
    ],
    "PV": [
        "PV", "POAV", "pore_volume", "pore volume", "total pore volume",
        "PV [cm^3/g]", "POAV [cm^3/g]", "POAV [cm³/g]", "pore_volume [cm^3/g]"
    ],
    "LCD": [
        "LCD", "Largest cavity diameter", "largest_cavity_diameter", "lcd"
    ],
    "PLD": [
        "PLD", "pore_limiting_diameter", "Pore Limiting Diameter", "pld"
    ],
}

TARGET_ALIASES = {
    "UG at TPS ": ["UG at TPS", "UG @ TPS", "UG_TPS", "UG usable at TPS", "UG (TPS)"],
    "UV at TPS ": ["UV at TPS", "UV @ TPS", "UV_TPS", "UV usable at TPS", "UV (TPS)"],
}

# -------------------- Utilities --------------------
def sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)

def resolve_first_present(df: pd.DataFrame, candidates: list) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def ensure_column_with_alias(df: pd.DataFrame, canonical: str, aliases_map: dict) -> bool:
    """If any alias exists, copy into canonical name. Return True if present after resolution."""
    if canonical in df.columns:
        return True
    alias = resolve_first_present(df, aliases_map.get(canonical, []))
    if alias is not None:
        if alias != canonical:
            df[canonical] = df[alias]
        return True
    return False

def collect_features(df: pd.DataFrame, feature_keys: list, aliases_map: dict) -> list:
    present = []
    missing = []
    for key in feature_keys:
        ok = ensure_column_with_alias(df, key, aliases_map)
        if ok:
            present.append(key)
        else:
            missing.append(key)
    if missing:
        alias_info = "\n".join([f"- {k}: {aliases_map.get(k, [])}" for k in missing])
        raise ValueError(
            "Kolom fitur berikut tidak ditemukan / tidak dikenali:\n"
            + ", ".join(missing)
            + "\n\nCek ejaan kolom pada file input.\nAlias yang didukung:\n"
            + alias_info
        )
    return present

def resolve_target(df: pd.DataFrame, target_name: str) -> str:
    if ensure_column_with_alias(df, target_name, TARGET_ALIASES):
        return target_name
    # fallback: coba alias lain—kalau ada yang muncul, kembalikan canonical
    for cand in TARGET_ALIASES.get(target_name, []):
        if cand in df.columns:
            df[target_name] = df[cand]
            return target_name
    raise ValueError(f"Target '{target_name}' tidak ditemukan. Cek alias: {TARGET_ALIASES.get(target_name, [])}")

def build_equation(target_name, intercept, feature_names_poly, coefs):
    # coefs di sini sudah termasuk intercept (idx 0) jika kita rangkai manual
    terms = [f"{intercept:.6g}"]
    # feature_names_poly sudah mengandung '1' (bias) di indeks 0
    for name, coef in zip(feature_names_poly[1:], coefs[1:]):  # skip bias term
        if np.isfinite(coef) and abs(coef) >= 1e-12:
            sign = " + " if coef >= 0 else " - "
            terms.append(f"{sign}{abs(coef):.6g}*({name})")
    return f"{target_name} ≈ " + "".join(terms)

def pearson_manual(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xbar, ybar = x.mean(), y.mean()
    cov = np.mean((x - xbar) * (y - ybar))
    stdx = x.std(ddof=0)
    stdy = y.std(ddof=0)
    if stdx == 0 or stdy == 0:
        return np.nan
    return cov / (stdx * stdy)

# ---------- Plot helpers ----------
def wb_plot_pearson_bar(corr_series: pd.Series, title="Pearson correlation vs target"):
    corr_sorted = corr_series.sort_values(key=lambda s: s.abs(), ascending=False)
    plt.figure()
    plt.bar(range(len(corr_sorted)), corr_sorted.values)
    plt.xticks(range(len(corr_sorted)), corr_sorted.index, rotation=45, ha="right")
    plt.ylabel("Pearson r")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def wb_plot_heatmap(df_corr: pd.DataFrame, title="Correlation heatmap"):
    plt.figure()
    im = plt.imshow(df_corr.values, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(df_corr.shape[1]), df_corr.columns, rotation=45, ha="right")
    plt.yticks(range(df_corr.shape[0]), df_corr.index)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def wb_plot_scatter_with_fit(x, y, xlabel, ylabel):
    coef = np.polyfit(x, y, 1)
    xp = np.linspace(np.min(x), np.max(x), 100)
    yp = np.polyval(coef, xp)
    plt.figure()
    plt.scatter(x, y, s=14)
    plt.plot(xp, yp)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs {xlabel}")
    plt.tight_layout()
    plt.show()

# -------------------- White-box (per target) --------------------
def run_whitebox_for_target(df: pd.DataFrame, target_col: str) -> None:
    # Pastikan target & fitur hadir (pakai alias)
    target_col = resolve_target(df, target_col)
    features = collect_features(df, CORE_FEATURE_KEYS, FEATURE_ALIASES)

    # Filter numeric & dropna
    cols_needed = features + [target_col]
    data = df[cols_needed].select_dtypes(include=[np.number]).dropna()
    if data.empty:
        raise ValueError(f"Tidak ada data numerik valid untuk target '{target_col}' setelah dropna. Cek NaN/tipe kolom.")

    X = data[features].values
    y = data[target_col].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    poly = PolynomialFeatures(degree=DEGREE, include_bias=True)
    Xtr_poly = poly.fit_transform(Xtr)
    Xte_poly = poly.transform(Xte)

    lr = LinearRegression()
    lr.fit(Xtr_poly, ytr)

    ytrp = lr.predict(Xtr_poly)
    ytep = lr.predict(Xte_poly)

    mae_tr = mean_absolute_error(ytr, ytrp)
    rmse_tr = mean_squared_error(ytr, ytrp) ** 0.5
    r2_tr = r2_score(ytr, ytrp)

    mae_te = mean_absolute_error(yte, ytep)
    rmse_te = mean_squared_error(yte, ytep) ** 0.5
    r2_te = r2_score(yte, ytep)

    feat_poly_names = poly.get_feature_names_out(features)
    # Rangakai koefisien seperti tampilan white-box: intercept + koef lainnya
    intercept = lr.intercept_
    # Removed: coefs_full = np.concatenate(([intercept], lr.coef_))

    # The 'coefs' argument in build_equation expects the list of coefficients corresponding
    # to feat_poly_names *excluding* the intercept, which is passed separately.
    # The previous code used np.concatenate(([0.0], lr.coef_[1:])), which makes sense if lr.coef_[0] is for the '1' term.
    equation = build_equation(target_col, intercept, feat_poly_names, np.concatenate(([0.0], lr.coef_[1:])))

    # Output dir per target
    tdir = OUTDIR / sanitize_name(target_col)
    tdir.mkdir(parents=True, exist_ok=True)

    # Simpan artefak teks/CSV
    (tdir / "whitebox_equation.txt").write_text(
        f"White-box polynomial model (degree={DEGREE})\n\n"
        f"Target : {target_col}\n"
        f"Features: {features}\n\n"
        f"Equation:\n{equation}\n\n"
        "Metrics:\n"
        f"- Train: MAE={mae_tr:.4f}, RMSE={rmse_tr:.4f}, R^2={r2_tr:.4f}\n"
        f"- Test : MAE={mae_te:.4f}, RMSE={rmse_te:.4f}, R^2={r2_te:.4f}\n"
        f"(Split: {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}, random_state={RANDOM_STATE})\n"
    )

    pd.DataFrame({
        "term": feat_poly_names,
        "coefficient": np.concatenate(([lr.intercept_], lr.coef_[1:])) # FIX: Reverted to previous working version
    }).to_csv(tdir / "whitebox_coefficients.csv", index=False)

    # Pearson (pandas & manual) terhadap target
    pearson_pandas = {f: data[f].corr(data[target_col]) for f in features}
    pd.Series(pearson_pandas, name="pearson_corr").to_csv(tdir / "pearson_correlations_vs_target.csv")
    pearson_manual_dict = {f: pearson_manual(data[f].values, data[target_col].values) for f in features}
    pd.Series(pearson_manual_dict, name="pearson_corr_manual").to_csv(tdir / "pearson_correlations_vs_target_manual.csv")

    # Plot: bar Pearson, heatmap fitur+target, scatter top N, pred vs actual
    wb_plot_pearson_bar(pd.Series(pearson_pandas), title=f"Pearson vs {target_col}")

    corr_mat = data[features + [target_col]].corr()
    wb_plot_heatmap(corr_mat, title=f"Heatmap (features + {target_col})")

    top_by_abs = pd.Series(pearson_pandas).abs().sort_values(ascending=False).index[:TOP_SCATTER]
    for f in top_by_abs:
        wb_plot_scatter_with_fit(data[f].values, data[target_col].values, xlabel=f, ylabel=target_col)

    plt.figure()
    plt.scatter(yte, ytep, s=18)
    plt.xlabel(f"Aktual {target_col} (test)")
    plt.ylabel("Prediksi")
    plt.title(f"Prediksi vs Aktual ({target_col}, degree={DEGREE})")
    mn = float(np.min([yte.min(), ytep.min()]))
    mx = float(np.max([yte.max(), ytep.max()]))
    plt.plot([mn, mx], [mn, mx])
    plt.tight_layout()
    plt.show()

    print(f"\n=== WHITE-BOX MODEL for {target_col} ===")
    print("Features:", features)
    print(f"Train: MAE={mae_tr:.4f}, RMSE={rmse_tr:.4f}, R^2={r2_tr:.4f}")
    print(f"Test : MAE={mae_te:.4f}, RMSE={rmse_te:.4f}, R^2={r2_te:.4f}")
    print("Equation (snippet):")
    print(textwrap.shorten(equation, width=160, placeholder=" ..."))
    print("\nSaved files:")
    print("-", (tdir / "whitebox_equation.txt").resolve())
    print("-", (tdir / "whitebox_coefficients.csv").resolve())
    print("-", (tdir / "pearson_correlations_vs_target.csv").resolve())
    print("-", (tdir / "pearson_correlations_vs_target_manual.csv").resolve())

# -------------------- Cross-validation with multiple train:test ratios --------------------
def run_train_test_ratio_cv(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Jalankan evaluasi model white-box (polynomial regression) pada beberapa rasio train:test.
    Rasio yang diuji: 90:10, 80:20, 70:30, 60:40, 50:50.
    Return dataframe berisi hasil tiap rasio.
    """
    ratios = [0.10, 0.20, 0.30, 0.40, 0.50]  # test sizes
    results = []

    print(f"\n=== Train:Test Ratio Sweep untuk target: {target_col} ===")
    print("Rasio train:test yang diuji ->", [f"{int((1-r)*100)}:{int(r*100)}" for r in ratios])

    # Pastikan fitur dan target siap
    target_col = resolve_target(df, target_col)
    features = collect_features(df, CORE_FEATURE_KEYS, FEATURE_ALIASES)

    # Filter numeric dan drop NA
    cols_needed = features + [target_col]
    data = df[cols_needed].select_dtypes(include=[np.number]).dropna()
    if data.empty:
        raise ValueError(f"Tidak ada data numerik valid untuk target '{target_col}' setelah dropna.")

    X = data[features].values
    y = data[target_col].values

    for test_size in ratios:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE
        )

        poly = PolynomialFeatures(degree=DEGREE, include_bias=True)
        Xtr_poly = poly.fit_transform(Xtr)
        Xte_poly = poly.transform(Xte)

        lr = LinearRegression()
        lr.fit(Xtr_poly, ytr)

        ytrp = lr.predict(Xtr_poly)
        ytep = lr.predict(Xte_poly)

        mae_tr = mean_absolute_error(ytr, ytrp)
        rmse_tr = mean_squared_error(ytr, ytrp) ** 0.5
        r2_tr = r2_score(ytr, ytrp)

        mae_te = mean_absolute_error(yte, ytep)
        rmse_te = mean_squared_error(yte, ytep) ** 0.5
        r2_te = r2_score(yte, ytep)

        results.append({
            "train:test": f"{int((1-test_size)*100)}:{int(test_size*100)}",
            "MAE_train": mae_tr,
            "RMSE_train": rmse_tr,
            "R2_train": r2_tr,
            "MAE_test": mae_te,
            "RMSE_test": rmse_te,
            "R2_test": r2_te
        })

    df_results = pd.DataFrame(results)
    print("\nHasil cross-validation per rasio:")
    print(df_results.to_string(index=False, float_format="%.4f"))

    # Tentukan performa terbaik berdasarkan R2 tertinggi & error terendah
    best_r2 = df_results.loc[df_results["R2_test"].idxmax()]
    best_err = df_results.loc[df_results["RMSE_test"].idxmin()]

    print("\n>> Rasio dengan R² tertinggi:", best_r2["train:test"])
    print("   R²_test = %.4f, RMSE_test = %.4f" % (best_r2["R2_test"], best_r2["RMSE_test"]))
    print(">> Rasio dengan RMSE terendah:", best_err["train:test"])
    print("   R²_test = %.4f, RMSE_test = %.4f" % (best_err["R2_test"], best_err["RMSE_test"]))

    # Simpan hasil
    outpath = OUTDIR / sanitize_name(f"cv_results_{target_col}.csv")
    df_results.to_csv(outpath, index=False)
    print("\nSaved CV results to:", outpath.resolve())

    return df_results

# -------------------- DOE-style Screening --------------------
def run_doe_screening(df: pd.DataFrame) -> None:
    # Pastikan target ada
    ug_col = resolve_target(df, "UG at TPS ")
    uv_col = resolve_target(df, "UV at TPS ")

    df_filtered = df.dropna(subset=[ug_col, uv_col]).copy()
    print(f"Total rows (all): {len(df):,}")
    print(f"Total rows (with {ug_col} & {uv_col}): {len(df_filtered):,}")

    meets_grav = df_filtered[ug_col] >= DOE_TARGET_GRAV
    meets_vol  = df_filtered[uv_col] >= DOE_TARGET_VOL
    df_screened = df_filtered[meets_grav & meets_vol].copy()
    print(f"Number of MOFs meeting DOE-like thresholds: {len(df_screened):,}")

    # Save table
    xlsx_path = OUTDIR / "mofs_meet_doe_like_tps.xlsx"
    df_screened.to_excel(xlsx_path, index=False)
    print("Excel saved at:", xlsx_path.resolve())

    # Scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(df_filtered[ug_col], df_filtered[uv_col], alpha=0.5, label="All entries")
    if not df_screened.empty:
        plt.scatter(df_screened[ug_col], df_screened[uv_col], alpha=0.9, label="Meets thresholds")
    plt.axvline(DOE_TARGET_GRAV, linestyle="--", label=f"UG threshold ({DOE_TARGET_GRAV} wt%)")
    plt.axhline(DOE_TARGET_VOL,  linestyle="--", label=f"UV threshold ({DOE_TARGET_VOL} g H₂/L)")
    plt.xlabel("UG at TPS [wt.%]")
    plt.ylabel("UV at TPS [g H₂/L]")
    plt.title("Screening vs DOE 2025-style Thresholds (using TPS columns)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if DOE_SHOW_PLOTS:
        plt.show()
    else:
        plt.savefig(OUTDIR / "doe_like_screening_scatter_tps.png", dpi=200)
        plt.close()

    # Optional Colab download
    try:
        from google.colab import files  # type: ignore
        files.download(str(xlsx_path))
    except Exception:
        pass

# -------------------- Main Orchestrator --------------------
def main():
    # Load
    if not FILE_PATH.exists():
        raise FileNotFoundError(f"Input file tidak ditemukan: {FILE_PATH.resolve()}")
    df = pd.read_excel(FILE_PATH)

    # Jalankan white-box untuk tiap target
    for tgt in TARGETS:
        run_whitebox_for_target(df.copy(), tgt)
        # Tambahan: jalankan sweep train:test ratios untuk per-target
        run_train_test_ratio_cv(df.copy(), tgt)

    # DOE-like screening (opsional)
    if DOE_ENABLE:
        run_doe_screening(df.copy())

if __name__ == "__main__":
    main()
