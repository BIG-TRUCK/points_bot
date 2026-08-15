"""Train the two-stage CHL points model using Leave-One-Out Cross-Validation.

Stage 1 — Classifier: "should this card be pointed?" (binary, all cards)
Stage 2 — Regressor:  "how many points?"            (only pointed cards)

Usage:
    python -m model.train                     # loads latest per_card pkl
    python -m model.train data/my_data.pkl    # explicit path

Each run is logged to a local MLflow store (./mlflow.db, artifacts under
./mlruns/) — params (dataset size, class balance, winning model + its tuned
hyperparameters), metrics (AUPRC, F1, MAE, ...), and the model artifact
itself. Browse past runs with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import logging
import os
import pickle
import subprocess
import sys
from glob import glob

import lightgbm as lgb
import mlflow
import mord
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from model.features import build_feature_matrix
from model.feedback import apply_feedback_to_df

logger = logging.getLogger(__name__)

DATA_DIR = "data"
MODEL_PATH = os.path.join(DATA_DIR, "chl_model.pkl")
VALID_POINTS = {0, 1, 2, 3, 5, 7, 8}

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT = "chl_points_bot"


def _git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


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
# Hyperparameter tuning — a small random search per candidate model type,
# run once up front on a cheap K-fold split. This is separate from (and much
# cheaper than) the LOOCV below, which compares model *types* against each
# other with an unbiased held-out estimate; tuning only picks each type's
# best internal settings before that comparison runs.
# ---------------------------------------------------------------------------

_GBM_PARAM_DIST = {
    "n_estimators": [200, 400, 600],
    "learning_rate": [0.03, 0.05, 0.08, 0.1],
    "min_child_samples": [1, 2, 5],  # LightGBM; XGBoost uses min_child_weight (below)
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0.0, 0.1, 0.5],
    "reg_lambda": [0.0, 0.1, 0.5],
}
_LGBM_PARAM_DIST = {**_GBM_PARAM_DIST, "num_leaves": [7, 15, 31]}
_XGB_PARAM_DIST = {
    k: v for k, v in _GBM_PARAM_DIST.items() if k != "min_child_samples"
} | {"max_depth": [2, 3, 4, 5], "min_child_weight": [1, 2, 5]}
_LOGISTIC_PARAM_DIST = {"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}
_SVC_PARAM_DIST = {
    "C": [0.1, 0.3, 1.0, 3.0, 10.0],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
}
_SVR_PARAM_DIST = {**_SVC_PARAM_DIST, "epsilon": [0.05, 0.1, 0.2, 0.5]}
_ORDINAL_PARAM_DIST = {"alpha": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]}

TUNE_N_ITER = 12


def _tune(
    base_model,
    param_dist: dict,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str,
    cv,
    n_iter: int = TUNE_N_ITER,
    random_state: int = 42,
) -> dict:
    """Randomized search over `param_dist` for one model type. Distance
    models get StandardScaler baked into the CV pipeline so scaling never
    leaks across folds (unlike the LOOCV below, which fits the scaler on
    the full data once — an existing, accepted approximation there)."""
    needs_scaling = _needs_scaling(base_model)
    estimator = Pipeline([("scaler", StandardScaler()), ("model", base_model)]) if needs_scaling else base_model
    params = {f"model__{k}": v for k, v in param_dist.items()} if needs_scaling else param_dist
    search = RandomizedSearchCV(
        estimator,
        params,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=1,
    )
    search.fit(X, y)
    return {k.removeprefix("model__"): v for k, v in search.best_params_.items()}


def _tuned_factory(base_model, tuned_params: dict):
    def factory():
        return clone(base_model).set_params(**tuned_params)

    return factory


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

_FAST_SHAP_MODELS = (
    lgb.LGBMClassifier, lgb.LGBMRegressor,
    xgb.XGBClassifier, xgb.XGBRegressor,
    LogisticRegression,
)


def compute_shap(model, X: pd.DataFrame, background: pd.DataFrame | None = None) -> pd.DataFrame:
    """Returns mean |SHAP| per feature for any of this project's model types.

    Tree models (LightGBM/XGBoost) get the exact TreeExplainer. Logistic
    Regression gets the exact LinearExplainer. Everything else (SVM,
    OrdinalRidge) has no closed-form SHAP, so falls back to KernelExplainer
    against a small k-means-summarized background — much slower, which is
    why callers should keep `X` small for these (see _shap_subsample).
    """
    if isinstance(model, (lgb.LGBMClassifier, lgb.LGBMRegressor, xgb.XGBClassifier, xgb.XGBRegressor)):
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(X)
    elif isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, background if background is not None else X)
        vals = explainer.shap_values(X)
    else:
        # No closed-form SHAP for this model type (SVM, OrdinalRidge). nsamples
        # is capped well below KernelExplainer's "auto" default (~2*n_features)
        # — with ~1,000 features that default means thousands of model
        # evaluations per explained row, which is far too slow for this to run
        # as a routine part of training. This trades some SHAP precision for
        # runtime; fine for an interpretability sidebar, not used by the model.
        bg_source = background if background is not None else X
        bg = shap.kmeans(bg_source, min(25, len(bg_source)))
        predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        explainer = shap.KernelExplainer(predict_fn, bg)
        vals = explainer.shap_values(X, nsamples=200, silent=True)

    if isinstance(vals, list):
        vals = vals[1] if len(vals) > 1 else vals[0]  # binary classifier -> [neg, pos]
    vals = np.asarray(vals)
    if vals.ndim > 2:
        vals = vals[:, :, -1]  # some explainers add a trailing output/class axis
    mean_abs = np.abs(vals).mean(axis=0)
    return (
        pd.Series(mean_abs, index=X.columns)
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .to_frame()
    )


def _shap_subsample(X: pd.DataFrame, y: pd.Series, model, max_rows: int) -> pd.DataFrame:
    """Tree/Logistic models get exact, cheap explainers regardless of size;
    everything else falls back to the much slower KernelExplainer, so those
    are explained on a stratified subsample (all positives + a random sample
    of negatives) rather than the full dataset."""
    if isinstance(model, _FAST_SHAP_MODELS) or len(X) <= max_rows:
        return X
    rng = np.random.RandomState(42)
    y_aligned = y.loc[X.index]
    pos_idx = list(X.index[y_aligned == 1])
    neg_pool = list(X.index[y_aligned == 0])
    n_neg = max(max_rows - len(pos_idx), 0)
    neg_idx = list(rng.choice(neg_pool, size=min(n_neg, len(neg_pool)), replace=False))
    return X.loc[pos_idx + neg_idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_path: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        commit = _git_commit_hash()
        if commit:
            mlflow.set_tag("git_commit", commit)

        df = _load_per_card_df(data_path)
        df = apply_feedback_to_df(df)

        pointed = df[df["points"].notna() & (df["points"] > 0)].copy()
        logger.info(f"Pointed cards (stage 2 training set): {len(pointed)}")

        # Build feature matrices
        X_all, preprocessors = build_feature_matrix(df)
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

        mlflow.log_params({
            "n_pos": n_pos,
            "n_neg": n_neg,
            "scale_pos_weight": round(spw, 3),
            "tune_n_iter": TUNE_N_ITER,
            "n_features": X_all.shape[1],
        })

        # --------------------------------------------------------------
        # Stage 1: Classifier LOOCV
        # --------------------------------------------------------------
        logger.info("Stage 1 — Classifier LOOCV...")

        logger.info(f"Tuning classifier hyperparameters ({TUNE_N_ITER}-iteration random search per candidate)...")
        clf_tune_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        clf_candidates_untuned = [
            ("LightGBM", _lgbm_classifier(spw), _LGBM_PARAM_DIST),
            ("XGBoost", _xgb_classifier(spw), _XGB_PARAM_DIST),
            ("Logistic", _logistic_classifier(), _LOGISTIC_PARAM_DIST),
            ("SVM", _svm_classifier(), _SVC_PARAM_DIST),
        ]
        clf_tuned_params: dict[str, dict] = {}
        clf_candidates = []
        for name, base_model, param_dist in clf_candidates_untuned:
            best_params = _tune(base_model, param_dist, X_all, y_binary, scoring="average_precision", cv=clf_tune_cv)
            clf_tuned_params[name] = best_params
            logger.info(f"  {name} tuned params: {best_params}")
            clf_candidates.append((name, _tuned_factory(base_model, best_params)))

        clf_results = {}
        for name, factory in clf_candidates:
            logger.info(f"  Running classifier LOOCV — {name}...")
            result = run_loocv_classifier(X_all, y_binary, factory)
            clf_results[name] = result
            logger.info(f"  {name}: mean_prob={result['mean_prob']:.3f}  mean_rank_pct={result['mean_rank_pct']:.3f}")

        best_clf_name = max(clf_results, key=lambda k: clf_results[k]["mean_rank_pct"])
        logger.info(f"Best classifier: {best_clf_name} (mean_rank_pct={clf_results[best_clf_name]['mean_rank_pct']:.3f})")

        mlflow.log_param("best_classifier", best_clf_name)
        mlflow.log_params({f"clf_best__{k}": v for k, v in clf_tuned_params[best_clf_name].items()})
        mlflow.log_metric("clf_mean_rank_pct", clf_results[best_clf_name]["mean_rank_pct"])

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

        # SHAP for every classifier candidate, not just the winner. Tree models
        # and Logistic have exact explainers; SVM falls back to KernelExplainer,
        # which is why it's explained on a stratified subsample rather than the
        # full ~1,624-row dataset (see _shap_subsample).
        SHAP_MAX_ROWS_SLOW = 60
        logger.info("Computing SHAP for every classifier candidate...")
        shap_importance_clf: dict[str, pd.DataFrame] = {}
        for name, factory in clf_candidates:
            if name == best_clf_name:
                model, model_scaler = final_clf, scaler
            else:
                model = factory()
                model_scaler = StandardScaler() if _needs_scaling(model) else None
                if model_scaler is not None:
                    model.fit(model_scaler.fit_transform(X_all), y_binary)
                else:
                    model.fit(X_all, y_binary)
            X_c = X_all if model_scaler is None else pd.DataFrame(
                model_scaler.transform(X_all), index=X_all.index, columns=X_all.columns
            )
            X_explain = _shap_subsample(X_c, y_binary, model, SHAP_MAX_ROWS_SLOW)
            logger.info(f"  SHAP — {name} ({len(X_explain)} of {len(X_c)} rows)...")
            shap_importance_clf[name] = compute_shap(model, X_explain)
        print(f"\n--- Top 15 features by mean |SHAP| (classifier, {best_clf_name}) ---")
        print(shap_importance_clf[best_clf_name].head(15).to_string())

        # F1 / AUPRC (average precision): positives use their LOOCV-held-out
        # probability (unbiased); negatives use the final model's in-sample
        # probability, since full LOOCV over the ~1,584 unpointed cards was
        # never run (see run_loocv_classifier — only positives are held out,
        # given the class imbalance). So these numbers are a mildly optimistic
        # upper bound on true held-out performance, not a strict estimate of it.
        # AUPRC (not AUROC) is the headline ranking metric here since it's the
        # one that doesn't get flattered by the ~40:1,584 class imbalance.
        probs_all = final_clf.predict_proba(scaler.transform(X_all) if scaler is not None else X_all)[:, 1]
        probs_for_eval = probs_all.copy()
        row_index = {name: i for i, name in enumerate(y_binary.index)}
        for name, held_out_prob in zip(best_clf_res["card_names"], best_clf_res["probs"]):
            probs_for_eval[row_index[name]] = held_out_prob

        auprc = average_precision_score(y_binary, probs_for_eval)
        auroc = roc_auc_score(y_binary, probs_for_eval)
        preds_at_half = (probs_for_eval >= 0.5).astype(int)

        thresholds = np.linspace(0.01, 0.99, 99)
        f1s = [f1_score(y_binary, (probs_for_eval >= t).astype(int), zero_division=0) for t in thresholds]
        best_t_idx = int(np.argmax(f1s))

        classifier_eval = {
            "auprc": float(auprc),
            "f1_at_0.5": float(f1_score(y_binary, preds_at_half, zero_division=0)),
            "precision_at_0.5": float(precision_score(y_binary, preds_at_half, zero_division=0)),
            "recall_at_0.5": float(recall_score(y_binary, preds_at_half, zero_division=0)),
            "best_f1": float(f1s[best_t_idx]),
            "best_f1_threshold": float(thresholds[best_t_idx]),
            "auroc": float(auroc),  # supplementary — inflates easily under this class imbalance, prefer AUPRC
            "note": (
                "Positives (pointed cards) use LOOCV-held-out probabilities; negatives "
                "(unpointed cards) use in-sample probabilities from the final model, "
                "since full LOOCV over the unpointed cards wasn't run given the class "
                "imbalance. Treat these as a mildly optimistic upper bound, not a "
                "strict held-out estimate. AUPRC is the headline ranking metric - "
                "AUROC is included for reference but reads misleadingly high under "
                "~40:1,584 class imbalance."
            ),
        }
        logger.info(
            f"Classifier eval — AUPRC={auprc:.3f}  F1@0.5={classifier_eval['f1_at_0.5']:.3f}  "
            f"best F1={classifier_eval['best_f1']:.3f} @ t={classifier_eval['best_f1_threshold']:.2f}  "
            f"(AUROC={auroc:.3f})"
        )

        mlflow.log_metrics({
            "clf_auprc": classifier_eval["auprc"],
            "clf_auroc": classifier_eval["auroc"],
            "clf_best_f1": classifier_eval["best_f1"],
            "clf_best_f1_threshold": classifier_eval["best_f1_threshold"],
        })

        # --------------------------------------------------------------
        # Stage 2: Regressor LOOCV
        # --------------------------------------------------------------
        logger.info("Stage 2 — Regressor LOOCV...")

        mode_val = float(y_points.mode().iloc[0])
        naive_mae = float(np.mean(np.abs(np.array(y_points, dtype=float) - mode_val)))
        logger.info(f"Naive baseline MAE (always predict mode={mode_val}): {naive_mae:.3f}")

        logger.info(f"Tuning regressor hyperparameters ({TUNE_N_ITER}-iteration random search per candidate)...")
        reg_tune_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        reg_candidates_untuned = [
            ("LightGBM", _lgbm_regressor(), _LGBM_PARAM_DIST),
            ("XGBoost", _xgb_regressor(), _XGB_PARAM_DIST),
            ("OrdinalRidge", _ordinal_regressor(), _ORDINAL_PARAM_DIST),
            ("SVM", _svm_regressor(), _SVR_PARAM_DIST),
        ]
        reg_tuned_params: dict[str, dict] = {}
        reg_candidates = []
        for name, base_model, param_dist in reg_candidates_untuned:
            best_params = _tune(
                base_model, param_dist, X_pointed, y_points, scoring="neg_mean_absolute_error", cv=reg_tune_cv
            )
            reg_tuned_params[name] = best_params
            logger.info(f"  {name} tuned params: {best_params}")
            reg_candidates.append((name, _tuned_factory(base_model, best_params)))

        reg_results = {}
        for name, factory in reg_candidates:
            logger.info(f"  Running regressor LOOCV — {name}...")
            result = run_loocv_regressor(X_pointed, y_points, factory)
            reg_results[name] = result
            logger.info(f"  {name}: MAE={result['mae']:.3f}  breakdown={result['breakdown_by_points']}")

        best_reg_name = min(reg_results, key=lambda k: reg_results[k]["mae"])
        logger.info(f"Best regressor: {best_reg_name} (MAE={reg_results[best_reg_name]['mae']:.3f})")

        mlflow.log_param("best_regressor", best_reg_name)
        mlflow.log_params({f"reg_best__{k}": v for k, v in reg_tuned_params[best_reg_name].items()})
        mlflow.log_metric("reg_mae", reg_results[best_reg_name]["mae"])

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

        # SHAP for every regressor candidate. X_pointed is small (~40 rows), so
        # no subsampling is needed even for the KernelExplainer fallback.
        logger.info("Computing SHAP for every regressor candidate...")
        shap_importance_reg: dict[str, pd.DataFrame] = {}
        for name, factory in reg_candidates:
            if name == best_reg_name:
                model, model_scaler = final_reg, reg_scaler
            else:
                model = factory()
                model_scaler = StandardScaler() if _needs_scaling(model) else None
                if model_scaler is not None:
                    model.fit(model_scaler.fit_transform(X_pointed), y_points)
                else:
                    model.fit(X_pointed, y_points)
            X_c = X_pointed if model_scaler is None else pd.DataFrame(
                model_scaler.transform(X_pointed), index=X_pointed.index, columns=X_pointed.columns
            )
            logger.info(f"  SHAP — {name}...")
            shap_importance_reg[name] = compute_shap(model, X_c)
        print(f"\n--- Top 15 features by mean |SHAP| (regressor, {best_reg_name}) ---")
        print(shap_importance_reg[best_reg_name].head(15).to_string())

        artifact = {
            "classifier": final_clf,
            "classifier_scaler": scaler,
            "classifier_eval": classifier_eval,
            "regressor": final_reg,
            "regressor_scaler": reg_scaler,
            "preprocessors": preprocessors,
            "clf_results": clf_results,
            "reg_results": reg_results,
            "clf_tuned_params": clf_tuned_params,
            "reg_tuned_params": reg_tuned_params,
            "naive_baseline_mae": naive_mae,
            "feature_columns": list(X_all.columns),
            "shap_importance_clf": shap_importance_clf,
            "shap_importance_reg": shap_importance_reg,
            "best_clf_name": best_clf_name,
            "best_reg_name": best_reg_name,
        }
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(artifact, f)
        logger.info(f"Model saved to {MODEL_PATH}")

        mlflow.log_artifact(MODEL_PATH)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
