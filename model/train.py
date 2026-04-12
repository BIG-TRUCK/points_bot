"""Train a CHL points prediction model using Leave-One-Out Cross-Validation.

Usage:
    python -m model.train                     # loads latest per_card pkl
    python -m model.train data/my_data.pkl    # explicit path
"""

import logging
import os
import pickle
import sys
from glob import glob

import lightgbm as lgb
import mord
import numpy as np
import xgboost as xgb
import pandas as pd
import shap
from sklearn.model_selection import LeaveOneOut

from model.features import build_feature_matrix
from model.feedback import apply_feedback_to_df

logger = logging.getLogger(__name__)

DATA_DIR = "data"
MODEL_PATH = os.path.join(DATA_DIR, "chl_model.pkl")
VALID_POINTS = {0, 1, 2, 3, 5, 7, 8}


def _load_per_card_df(path: str | None = None) -> pd.DataFrame:
    if path:
        with open(path, "rb") as f:
            return pickle.load(f)
    candidates = sorted(glob(os.path.join(DATA_DIR, "chl_dataset_per_card_*.pkl")))
    if not candidates:
        raise FileNotFoundError("No per_card dataset found. Run pipeline.py first.")
    latest = candidates[-1]
    logger.info(f"Loading {latest}")
    with open(latest, "rb") as f:
        return pickle.load(f)


def _snap_to_valid(value: float) -> int:
    """Rounds a continuous prediction to the nearest valid point value."""
    return min(VALID_POINTS, key=lambda v: abs(v - value))


def _lgbm_model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )


def _ordinal_model() -> mord.OrdinalRidge:
    return mord.OrdinalRidge(alpha=1.0)


def _xgb_model() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=0,
    )


def run_loocv(X: pd.DataFrame, y: pd.Series, model_factory) -> dict:
    """Runs LOOCV and returns a results dict with predictions and metrics."""
    loo = LeaveOneOut()
    preds_raw = np.zeros(len(y))
    card_names = y.index.tolist()

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        model = model_factory()
        model.fit(X_train, y_train)
        preds_raw[test_idx] = model.predict(X_test)

    preds_snapped = [_snap_to_valid(p) for p in preds_raw]
    errors = np.abs(preds_raw - y.values)

    # Per-point-value breakdown
    breakdown = {}
    for pt in sorted(VALID_POINTS):
        mask = y.values == pt
        if mask.any():
            breakdown[pt] = float(np.mean(np.abs(preds_raw[mask] - y.values[mask])))

    return {
        "card_names": card_names,
        "y_true": y.values.tolist(),
        "y_pred_raw": preds_raw.tolist(),
        "y_pred_snapped": preds_snapped,
        "mae": float(np.mean(errors)),
        "breakdown_by_points": breakdown,
    }


def train_final_model(X: pd.DataFrame, y: pd.Series, model_factory) -> object:
    """Trains on the full labeled set and returns the fitted model."""
    model = model_factory()
    model.fit(X, y)
    return model


def compute_shap(model: lgb.LGBMRegressor, X: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe of mean |SHAP| values per feature, descending."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.Series(mean_abs, index=X.columns)
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .to_frame()
    )


def main(data_path: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO)

    df = _load_per_card_df(data_path)
    df = apply_feedback_to_df(df)
    labeled = df[df["points"].notna()].copy()
    logger.info(f"Labeled cards (incl. feedback): {len(labeled)}")

    X_labeled = build_feature_matrix(labeled)
    y = labeled["points"].astype(float)

    # Naive baseline
    naive_mae = float(np.mean(np.abs(y.values - y.mode()[0])))
    logger.info(f"Naive baseline MAE (always predict mode={y.mode()[0]}): {naive_mae:.3f}")

    # LOOCV — LightGBM
    logger.info("Running LOOCV — LightGBM...")
    lgbm_results = run_loocv(X_labeled, y, _lgbm_model)
    logger.info(f"LightGBM LOOCV MAE: {lgbm_results['mae']:.3f}")
    logger.info(f"  Per-point breakdown: {lgbm_results['breakdown_by_points']}")

    # LOOCV — XGBoost
    logger.info("Running LOOCV — XGBoost...")
    xgb_results = run_loocv(X_labeled, y, _xgb_model)
    logger.info(f"XGBoost LOOCV MAE: {xgb_results['mae']:.3f}")
    logger.info(f"  Per-point breakdown: {xgb_results['breakdown_by_points']}")

    # LOOCV — Ordinal regression
    logger.info("Running LOOCV — Ordinal Ridge...")
    ord_results = run_loocv(X_labeled, y, _ordinal_model)
    logger.info(f"Ordinal Ridge LOOCV MAE: {ord_results['mae']:.3f}")
    logger.info(f"  Per-point breakdown: {ord_results['breakdown_by_points']}")

    # Print per-card predictions from the best model
    all_results = [
        ("LightGBM", lgbm_results),
        ("XGBoost", xgb_results),
        ("OrdinalRidge", ord_results),
    ]
    best_name, best_results = min(all_results, key=lambda x: x[1]["mae"])
    logger.info(f"Best model: {best_name} (MAE={best_results['mae']:.3f})")
    print("\n--- LOOCV predictions (best model) ---")
    for name, true, pred in zip(
        best_results["card_names"],
        best_results["y_true"],
        best_results["y_pred_snapped"],
    ):
        flag = " <--" if abs(true - pred) >= 2 else ""
        print(f"  {name:<35} true={int(true)}  pred={pred}{flag}")

    # Train final model on all labeled data
    logger.info("Training final LightGBM model on full labeled set...")
    final_model = train_final_model(X_labeled, y, _lgbm_model)

    # SHAP feature importance
    shap_df = compute_shap(final_model, X_labeled)
    print("\n--- Top 20 features by mean |SHAP| ---")
    print(shap_df.head(20).to_string())

    # Save model + metadata
    artifact = {
        "model": final_model,
        "lgbm_loocv": lgbm_results,
        "xgb_loocv": xgb_results,
        "ordinal_loocv": ord_results,
        "naive_baseline_mae": naive_mae,
        "feature_columns": list(X_labeled.columns),
        "shap_importance": shap_df,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    logger.info(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
