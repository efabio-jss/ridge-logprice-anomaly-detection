import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import matplotlib.pyplot as plt


def make_data(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    """Generate synthetic dataset with brand, age, condition and price."""
    rng = np.random.default_rng(seed)
    brands = np.array(["A", "B", "C", "D"])
    condition = np.array(["poor", "good", "like_new"])

    df = pd.DataFrame({
        "brand": rng.choice(brands, size=n, p=[0.35, 0.3, 0.2, 0.15]),
        "age_years": rng.gamma(shape=3.0, scale=3.0, size=n).clip(0.3, 40.0),
        "condition": rng.choice(condition, size=n, p=[0.15, 0.55, 0.30]),
        "diameter": rng.normal(40, 3.0, size=n).clip(32, 48),
        "has_box": rng.integers(0, 2, size=n),
        "has_papers": rng.integers(0, 2, size=n),
    })

    # True DGP (log-price) with heteroscedastic noise
    brand_eff = {"A": 9.3, "B": 9.0, "C": 9.6, "D": 9.4}
    cond_eff = {"poor": -0.35, "good": 0.0, "like_new": 0.20}

    mu = (
        df["brand"].map(brand_eff).astype(float)
        + df["age_years"].mul(-0.02)
        + (df["diameter"] - 40) * 0.02
        + df["condition"].map(cond_eff).astype(float)
        + 0.03 * df["has_box"]
        + 0.04 * df["has_papers"]
    )

    noise = rng.normal(0, 0.10 + 0.01 * df["age_years"], size=n)
    log_price = mu + noise
    df["price"] = np.exp(log_price).round(0)

    # Inject anomalies (outliers in price and diameter)
    out_idx = rng.choice(df.index, size=25, replace=False)
    df.loc[out_idx[:12], "price"] *= rng.choice([0.35, 3.2], size=12)
    df.loc[out_idx[12:], "diameter"] *= rng.choice([0.6, 1.5], size=13)

    return df


def build_model() -> Pipeline:
    """Build preprocessing + Ridge regression pipeline."""
    cat = ["brand", "condition"]
    num = ["age_years", "diameter", "has_box", "has_papers"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])
    model = Ridge(alpha=1.0, solver="sag", random_state=7)
    return Pipeline([("pre", pre), ("model", model)])


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    B: int = 300,
    seed: int = 7
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Bootstrap confidence intervals for MAE and R²."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    maes, r2s = [], []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        mae = mean_absolute_error(y_true[idx], y_pred[idx])
        r2 = r2_score(y_true[idx], y_pred[idx])
        maes.append(mae)
        r2s.append(r2)
    mae_ci = (float(np.quantile(maes, 0.025)), float(np.quantile(maes, 0.975)))
    r2_ci = (float(np.quantile(r2s, 0.025)), float(np.quantile(r2s, 0.975)))
    return mae_ci, r2_ci


def robust_z(x: np.ndarray) -> np.ndarray:
    """Compute robust z-scores based on MAD (median absolute deviation)."""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return 0.6745 * (x - med) / mad


def main() -> None:
    print(">> Generating synthetic dataset...")
    df = make_data()
    X = df.drop(columns=["price"])
    y = df["price"].values

    print(">> Train/test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )

    print(">> Fit Ridge on log(1+price)")
    model = build_model()
    model.fit(X_train, np.log1p(y_train))

    # Smearing estimator for bias correction when retransforming
    resid_log = np.log1p(y_train) - model.predict(X_train)
    smear = float(np.mean(np.exp(resid_log)))
    pred_test = np.expm1(model.predict(X_test)) * smear

    # Holdout evaluation
    mae = mean_absolute_error(y_test, pred_test)
    r2 = r2_score(y_test, pred_test)
    (mae_lo, mae_hi), (r2_lo, r2_hi) = bootstrap_ci(y_test, pred_test)

    print(f"-- Holdout MAE: {mae:.2f}  (95% CI: {mae_lo:.2f} .. {mae_hi:.2f})")
    print(f"-- Holdout R^2: {r2:.3f} (95% CI: {r2_lo:.3f} .. {r2_hi:.3f})")

    # Cross-validation on log scale
    print(">> 5-fold CV MAE (log-scale target)")
    kf = KFold(n_splits=5, shuffle=True, random_state=7)
    cv_scores = cross_val_score(
        model, X, np.log1p(y), scoring="neg_mean_absolute_error", cv=kf, n_jobs=-1
    )
    print(f"-- CV mean MAE (log): {(-cv_scores).mean():.4f}  (+/- {(-cv_scores).std():.4f})")

    # Residual-based anomaly detection
    print(">> Residual-based anomaly detection")
    resid = y_test - pred_test
    rz = robust_z(resid)
    df_res = X_test.copy()
    df_res = df_res.assign(price_true=y_test, price_pred=pred_test, resid=resid, resid_rz=rz)

    contamination = 0.05  # anomaly proportion
    iso = IsolationForest(random_state=7, contamination=contamination)
    iso.fit(df_res[["price_pred", "resid"]])
    iso_flag = iso.predict(df_res[["price_pred", "resid"]])  # -1 anomaly, 1 normal
    df_res["iforest"] = iso_flag
    df_res["anomaly_score"] = np.abs(df_res["resid_rz"]) + (df_res["iforest"] == -1).astype(int)

    top_anom = df_res.sort_values("anomaly_score", ascending=False).head(20)

    # Save outputs
    out_dir = "artifacts_python_demo"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "dataset.csv"), index=False)
    top_anom.to_csv(os.path.join(out_dir, "anomalies_top20.csv"), index=False)

    # Residuals vs predicted
    plt.figure()
    plt.scatter(pred_test, resid, alpha=0.5)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted price")
    plt.ylabel("Residual (y - ŷ)")
    plt.title("Residuals vs Predicted (Ridge log-price)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "residuals_vs_pred.png"))
    plt.close()

    # Histogram of robust z-scores
    plt.figure()
    plt.hist(rz, bins=40)
    plt.title("Robust Z-scores of Residuals")
    plt.xlabel("Robust z")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "residuals_rz_hist.png"))
    plt.close()

    # True vs predicted
    plt.figure()
    plt.scatter(y_test, pred_test, alpha=0.5)
    lo = min(float(np.min(y_test)), float(np.min(pred_test)))
    hi = max(float(np.max(y_test)), float(np.max(pred_test)))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("True price")
    plt.ylabel("Predicted price")
    plt.title("True vs Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "true_vs_pred.png"))
    plt.close()

    print("\n>> Summary")
    print("Artifacts saved to ./artifacts_python_demo:")
    print("- dataset.csv, anomalies_top20.csv")
    print("- residuals_vs_pred.png, residuals_rz_hist.png, true_vs_pred.png")

if __name__ == "__main__":
    main()
