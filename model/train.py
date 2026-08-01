"""Train the two-stage CHL points model using Leave-One-Out Cross-Validation.

Stage 1 — Classifier: "should this card be pointed?" (binary, all cards)
Stage 2 — Regressor:  "how many points?"            (only pointed cards)

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
import pandas as pd
import shap
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

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
    return min(VALID_POINTS, key=lambda v: abs(v - value))


def _needs_scaling(model) -> bool:
    """Distance/margin-based models (Logistic, SVM) need standardised input;
    tree models and OrdinalRidge do not."""
    return isinstance(model, (LogisticRegression, SVC, SVR))


# ---------------------------------------------------------------------------
# Model factories — classifiers
# ---------------------------------------------------------------------------

def _lgbm_classifier(scale_pos_weight: float) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
    )


def _xgb_classifier(scale_pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0,
        eval_metric="logloss",
    )


def _logistic_classifier() -> LogisticRegression:
    return LogisticRegression(C=0.1, max_iter=1000, random_state=42)


def _svm_classifier() -> SVC:
    return SVC(C=1.0, kernel="rbf", probability=True, random_state=42)


# ---------------------------------------------------------------------------
# Model factories — regressors
# ---------------------------------------------------------------------------

def _lgbm_regressor() -> lgb.LGBMRegressor:
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


def _xgb_regressor() -> xgb.XGBRegressor:
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


def _ordinal_regressor() -> mord.OrdinalRidge:
    return mord.OrdinalRidge(alpha=1.0)


def _svm_regressor() -> SVR:
    return SVR(C=1.0, kernel="rbf")


# ---------------------------------------------------------------------------
# LOOCV — classifier
# ---------------------------------------------------------------------------

def run_loocv_classifier(
    X_all: pd.DataFrame,
    y_binary: pd.Series,
    model_factory,
) -> dict:
    """LOOCV over positive examples only.

    For each pointed card, flip its label to 0 for training, then measure
    what probability the model assigns to it. The mean probability across all
    held-out positives is the key metric: higher = better.

    Args:
        X_all:         Feature matrix for ALL cards (not just labeled ones).
        y_binary:      Binary series: 1 for pointed cards, 0 for everything else.
        model_factory: Callable returning a fresh classifier instance.

    Returns:
        dict with per-card probabilities and summary metrics.
    """
    positive_mask = y_binary == 1
    positive_indices = list(y_binary[positive_mask].index)
    card_names = list(y_binary[positive_mask].index)

    probs = []
    for idx in positive_indices:
        y_train = y_binary.copy()
        y_train.loc[idx] = 0  # temporarily treat as unpointed
        model = model_factory()
        if _needs_scaling(model):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_all)
            model.fit(X_scaled, y_train)
            prob = model.predict_proba(X_scaled[X_all.index.get_loc(idx)].reshape(1, -1))[0, 1]
        else:
            model.fit(X_all, y_train)
            prob = model.predict_proba(X_all.loc[[idx]])[0, 1]
        probs.append(float(prob))

    mean_prob = float(np.mean(probs))
    # Rank metric: for each held-out positive, what fraction of all cards
    # had a lower predicted probability? (1.0 = always top ranked)
    rank_pcts = []
    for idx, prob in zip(positive_indices, probs):
        y_train = y_binary.copy()
        y_train.loc[idx] = 0
        model = model_factory()
        if _needs_scaling(model):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_all)
            model.fit(X_scaled, y_train)
            all_probs = model.predict_proba(X_scaled)[:, 1]
        else:
            model.fit(X_all, y_train)
            all_probs = model.predict_proba(X_all)[:, 1]
        rank_pct = float((all_probs < prob).mean())
        rank_pcts.append(rank_pct)

    return {
        "card_names": card_names,
        "probs": probs,
        "mean_prob": mean_prob,
        "mean_rank_pct": float(np.mean(rank_pcts)),
    }


# ---------------------------------------------------------------------------
# LOOCV — regressor
# ---------------------------------------------------------------------------

def run_loocv_regressor(X: pd.DataFrame, y: pd.Series, model_factory) -> dict:
    """Standard LOOCV for the regressor over pointed cards only."""
    loo = LeaveOneOut()
    preds_raw = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        model = model_factory()
        if _needs_scaling(model):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        model.fit(X_train, y_train)
        preds_raw[test_idx] = model.predict(X_test)

    preds_snapped = [_snap_to_valid(p) for p in preds_raw]
    errors = np.abs(preds_raw - y.values)

    breakdown = {}
    for pt in sorted(VALID_POINTS):
        mask = y.values == pt
        if mask.any():
            breakdown[pt] = float(np.mean(np.abs(preds_raw[mask] - y.values[mask])))

    return {
        "card_names": list(y.index),
        "y_true": y.values.tolist(),
        "y_pred_raw": preds_raw.tolist(),
        "y_pred_snapped": preds_snapped,
        "mae": float(np.mean(errors)),
        "breakdown_by_points": breakdown,
    }


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def compute_shap(model, X: pd.DataFrame) -> pd.DataFrame:
    """Returns mean |SHAP| per feature for tree-based models."""
    explainer = shap.TreeExplainer(model)
    vals = explainer.shap_values(X)
    if isinstance(vals, list):
        vals = vals[1]  # binary classifier returns [neg, pos]
    mean_abs = np.abs(vals).mean(axis=0)
    return (
        pd.Series(mean_abs, index=X.columns)
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .to_frame()
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_path: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO)

    df = _load_per_card_df(data_path)
    df = apply_feedback_to_df(df)

    pointed = df[df["points"].notna() & (df["points"] > 0)].copy()
    logger.info(f"Pointed cards (stage 2 training set): {len(pointed)}")

    # Build feature matrices
    X_all, tfidf_vectorizer = build_feature_matrix(df)
    X_pointed = X_all.loc[pointed.index]
    y_binary = pd.Series(
        (df["points"].fillna(0) > 0).astype(int).values,
        index=df.index,
    )
    y_points = pointed["points"].astype(float)

    n_pos = int(y_binary.sum())
    n_neg = len(y_binary) - n_pos
    spw = n_neg / n_pos
    logger.info(f"Class balance — pointed: {n_pos}, unpointed: {n_neg}, scale_pos_weight: {spw:.1f}")

    # ------------------------------------------------------------------
    # Stage 1: Classifier LOOCV
    # ------------------------------------------------------------------
    logger.info("Stage 1 — Classifier LOOCV...")

    clf_candidates = [
        ("LightGBM", lambda: _lgbm_classifier(spw)),
        ("XGBoost",  lambda: _xgb_classifier(spw)),
        ("Logistic", _logistic_classifier),
        ("SVM",      _svm_classifier),
    ]

    clf_results = {}
    for name, factory in clf_candidates:
        logger.info(f"  Running classifier LOOCV — {name}...")
        result = run_loocv_classifier(X_all, y_binary, factory)
        clf_results[name] = result
        logger.info(f"  {name}: mean_prob={result['mean_prob']:.3f}  mean_rank_pct={result['mean_rank_pct']:.3f}")

    best_clf_name = max(clf_results, key=lambda k: clf_results[k]["mean_rank_pct"])
    logger.info(f"Best classifier: {best_clf_name} (mean_rank_pct={clf_results[best_clf_name]['mean_rank_pct']:.3f})")

    print("\n--- Classifier LOOCV (probability assigned to held-out pointed cards) ---")
    best_clf_res = clf_results[best_clf_name]
    for name, prob in zip(best_clf_res["card_names"], best_clf_res["probs"]):
        print(f"  {str(name):<35} prob={prob:.3f}")

    # Train final classifier
    best_clf_factory = dict(clf_candidates)[best_clf_name]
    final_clf = best_clf_factory()
    if _needs_scaling(final_clf):
        scaler = StandardScaler()
        X_all_scaled = scaler.fit_transform(X_all)
        final_clf.fit(X_all_scaled, y_binary)
    else:
        scaler = None
        final_clf.fit(X_all, y_binary)

    # ------------------------------------------------------------------
    # Stage 2: Regressor LOOCV
    # ------------------------------------------------------------------
    logger.info("Stage 2 — Regressor LOOCV...")

    mode_val = float(y_points.mode().iloc[0])
    naive_mae = float(np.mean(np.abs(np.array(y_points, dtype=float) - mode_val)))
    logger.info(f"Naive baseline MAE (always predict mode={mode_val}): {naive_mae:.3f}")

    reg_candidates = [
        ("LightGBM",     _lgbm_regressor),
        ("XGBoost",      _xgb_regressor),
        ("OrdinalRidge", _ordinal_regressor),
        ("SVM",          _svm_regressor),
    ]

    reg_results = {}
    for name, factory in reg_candidates:
        logger.info(f"  Running regressor LOOCV — {name}...")
        result = run_loocv_regressor(X_pointed, y_points, factory)
        reg_results[name] = result
        logger.info(f"  {name}: MAE={result['mae']:.3f}  breakdown={result['breakdown_by_points']}")

    best_reg_name = min(reg_results, key=lambda k: reg_results[k]["mae"])
    logger.info(f"Best regressor: {best_reg_name} (MAE={reg_results[best_reg_name]['mae']:.3f})")

    print("\n--- Regressor LOOCV (best model) ---")
    best_reg_res = reg_results[best_reg_name]
    for name, true, pred in zip(
        best_reg_res["card_names"],
        best_reg_res["y_true"],
        best_reg_res["y_pred_snapped"],
    ):
        flag = " <--" if abs(true - pred) >= 2 else ""
        print(f"  {str(name):<35} true={int(true)}  pred={pred}{flag}")

    # Train final regressor
    best_reg_factory = dict(reg_candidates)[best_reg_name]
    final_reg = best_reg_factory()
    if _needs_scaling(final_reg):
        reg_scaler = StandardScaler()
        X_pointed_scaled = reg_scaler.fit_transform(X_pointed)
        final_reg.fit(X_pointed_scaled, y_points)
    else:
        reg_scaler = None
        final_reg.fit(X_pointed, y_points)

    # SHAP (regressor only — most interpretable for point magnitude).
    # TreeExplainer only supports tree-based models.
    if isinstance(final_reg, (lgb.LGBMRegressor, xgb.XGBRegressor)):
        shap_df = compute_shap(final_reg, X_pointed)
        print("\n--- Top 20 features by mean |SHAP| (regressor) ---")
        print(shap_df.head(20).to_string())
    else:
        shap_df = None

    artifact = {
        "classifier": final_clf,
        "classifier_scaler": scaler,
        "regressor": final_reg,
        "regressor_scaler": reg_scaler,
        "tfidf_vectorizer": tfidf_vectorizer,
        "clf_results": clf_results,
        "reg_results": reg_results,
        "naive_baseline_mae": naive_mae,
        "feature_columns": list(X_all.columns),
        "shap_importance": shap_df,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    logger.info(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
