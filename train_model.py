#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training pipeline for ensemble models with in-fold target encoding and XGB scaling.
使用方法
python train_model.py --base-dir . --config config/train_model.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import lightgbm as lgb
import xgboost as xgb

try:  # CatBoost is optional to keep local setup lightweight
    from catboost import CatBoostRegressor
except ModuleNotFoundError:  # pragma: no cover - handled gracefully at runtime
    CatBoostRegressor = None  # type: ignore

try:
    from optuna.integration import TqdmCallback as OptunaTqdmCallback
except ImportError:  # pragma: no cover - fall back to default behaviour
    OptunaTqdmCallback = None  # type: ignore

from utils import metrics, reporting, weights

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
STAGE_NAMES = [
    "Load Config",
    "Prepare Data",
    "Build Folds",
    "Train Models",
    "Optimize Ensemble",
    "Finalize",
]
IDENTIFIER_COLUMNS = ["sector_id", "panel_month", "month_idx", "is_future_horizon"]


@dataclass
class FoldInfo:
    """Stores indices and metadata for a single time-based fold."""

    fold_id: int
    train_idx: np.ndarray
    valid_idx: np.ndarray
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    train_weights: np.ndarray
    valid_weights: np.ndarray
    train_target_min: float
# ---------------------------------------------------------------------------
# Configuration & utility helpers
# ---------------------------------------------------------------------------


def load_config(base_dir: Path, config_path: Optional[str]) -> Dict[str, object]:
    """Load YAML config, resolve relative paths, and normalize target encoding settings."""

    base_dir = base_dir.resolve()
    if config_path is None:
        config_path = base_dir / "config" / "train_model.yaml"
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    config["base_dir"] = str(base_dir)
    for key in ["features_path", "panel_path", "artifacts_dir", "logs_dir"]:
        if key in config:
            resolved = Path(config[key])
            if not resolved.is_absolute():
                resolved = (base_dir / resolved).resolve()
            config[key] = str(resolved)

    te_cfg = config.get("target_encoding")
    if isinstance(te_cfg, dict):
        mapping_path = te_cfg.get("mapping_path")
        if mapping_path:
            mapping = Path(mapping_path)
            if not mapping.is_absolute():
                mapping = (base_dir / mapping).resolve()
            te_cfg["mapping_path"] = str(mapping)
            config["target_encoding"] = te_cfg

    return config


def setup_logging(log_dir: Path) -> None:
    """Configure structured logging for console + file."""

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train_model.log"
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=handlers)
    logging.info("Training pipeline started; log file: %s", log_file)


def save_config_snapshot(config: Dict[str, object], log_dir: Path) -> Path:
    """Persist the resolved configuration for traceability."""

    snapshot_path = log_dir / "train_model_config.yaml"
    with open(snapshot_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, allow_unicode=True, sort_keys=True)
    logging.info("Configuration snapshot written to %s", snapshot_path)
    return snapshot_path


def read_features(path: Path, target_column: str, te_cfg: Dict[str, object]) -> List[str]:
    """Read selected feature names, excluding the raw target and appending TE output if needed."""

    if not path.exists():
        raise FileNotFoundError(f"Feature list not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        features = [line.strip() for line in fh if line.strip()]

    feature_set: List[str] = []
    for name in features:
        if name == target_column:
            continue
        if name not in feature_set:
            feature_set.append(name)

    if te_cfg.get("enabled", False):
        te_column = te_cfg.get("column", "sector_id")
        output_col = te_cfg.get("output_column", f"{te_column}_te")
        if output_col not in feature_set:
            feature_set.append(output_col)

    logging.info("Loaded %d candidate features", len(feature_set))
    return feature_set


def prepare_dataset(
    config: Dict[str, object],
    feature_cols: List[str],
    te_cfg: Dict[str, object],
) -> pd.DataFrame:
    """Load panel parquet, filter by train end, and validate feature availability."""

    panel_path = Path(config["panel_path"])
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel data not found: {panel_path}")

    df = pd.read_parquet(panel_path)
    df["panel_month"] = pd.to_datetime(df["panel_month"])
    train_end = pd.Timestamp(config["train_end_month"])
    train_df = df[df["panel_month"] <= train_end].copy()
    train_df.reset_index(drop=True, inplace=True)

    dynamic_features: set[str] = set()
    if te_cfg.get("enabled", False):
        te_column = te_cfg.get("column", "sector_id")
        output_col = te_cfg.get("output_column", f"{te_column}_te")
        dynamic_features.add(output_col)

    missing_features = [col for col in feature_cols if col not in train_df.columns and col not in dynamic_features]
    if missing_features:
        raise KeyError(f"Missing features in training data: {missing_features}")

    target_column = config["target_column"]
    target_na = train_df[target_column].isna()
    if target_na.any():
        logging.warning("Detected %d rows with NaN target; dropping", int(target_na.sum()))
        train_df = train_df.loc[~target_na].reset_index(drop=True)

    logging.info("Training rows: %d", len(train_df))
    return train_df
# ---------------------------------------------------------------------------
# Target encoding helpers
# ---------------------------------------------------------------------------


def _resolve_te_default(te_cfg: Dict[str, object], train_df: pd.DataFrame, target_column: str) -> float:
    fallback = te_cfg.get("handle_unknown", "mean")
    if isinstance(fallback, (int, float)):
        return float(fallback)
    if str(fallback).lower() == "zero":
        return 0.0
    return float(train_df[target_column].mean())


def apply_target_encoding(
    train_df: pd.DataFrame,
    target_column: str,
    te_cfg: Dict[str, object],
    valid_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, object]]:
    """Apply (optionally) target encoding to train/valid frames and return mapping metadata."""

    if not te_cfg.get("enabled", False):
        return train_df, valid_df, {}

    input_col = te_cfg.get("column", "sector_id")
    output_col = te_cfg.get("output_column", f"{input_col}_te")
    if input_col not in train_df.columns:
        raise KeyError(f"Target encoding column '{input_col}' not found in training data")

    encoded_train = train_df.copy()
    if output_col in encoded_train.columns:
        encoded_train = encoded_train.drop(columns=[output_col])

    agg = encoded_train.groupby(input_col)[target_column].mean()
    default_value = _resolve_te_default(te_cfg, encoded_train, target_column)

    encoded_train[output_col] = encoded_train[input_col].map(agg).fillna(default_value)

    if valid_df is not None:
        encoded_valid = valid_df.copy()
        if output_col in encoded_valid.columns:
            encoded_valid = encoded_valid.drop(columns=[output_col])
        encoded_valid[output_col] = encoded_valid[input_col].map(agg).fillna(default_value)
    else:
        encoded_valid = None

    mapping_payload = {
        "input_column": input_col,
        "output_column": output_col,
        "default_value": default_value,
        "mapping": {
            (str(key) if isinstance(key, (pd.Timestamp, Path)) else key): (float(val) if pd.notna(val) else None)
            for key, val in agg.items()
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return encoded_train, encoded_valid, mapping_payload


def save_target_encoding_mapping(
    te_cfg: Dict[str, object],
    mapping: Dict[str, object],
    default_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Persist the final target-encoding mapping for inference."""

    if not mapping or not te_cfg.get("enabled", False):
        return None

    mapping_value = te_cfg.get("mapping_path")
    if mapping_value:
        mapping_path = Path(mapping_value)
    else:
        mapping_path = Path("models/sector_te_mapping.json")
        if default_dir is not None:
            mapping_path = default_dir / mapping_path

    if not mapping_path.is_absolute():
        mapping_path = mapping_path.resolve()

    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, ensure_ascii=False, indent=2)
    logging.info("Saved target-encoding mapping to %s", mapping_path)
    return mapping_path
# ---------------------------------------------------------------------------
# CV fold creation & target transforms
# ---------------------------------------------------------------------------


def create_folds(train_df: pd.DataFrame, config: Dict[str, object], target_column: str) -> List[FoldInfo]:
    """Create deterministic folds based on configured time windows."""

    cv_cfg = config.get("cv", {})
    folds_cfg = cv_cfg.get("folds", [])
    minimum_months = int(cv_cfg.get("minimum_training_months", 0))
    weight_cfg = config.get("weights", {})
    folds: List[FoldInfo] = []

    for fold_id, fold_cfg in enumerate(folds_cfg, start=1):
        train_end = pd.Timestamp(fold_cfg["train_end"])
        valid_start = pd.Timestamp(fold_cfg["valid_start"])
        valid_end = pd.Timestamp(fold_cfg["valid_end"])

        train_mask = train_df["panel_month"] <= train_end
        valid_mask = (train_df["panel_month"] >= valid_start) & (train_df["panel_month"] <= valid_end)

        train_months = train_df.loc[train_mask, "panel_month"].nunique()
        if train_months < minimum_months:
            logging.warning(
                "Skipping fold %d; insufficient training months (%d < %d)",
                fold_id,
                train_months,
                minimum_months,
            )
            continue
        if valid_mask.sum() == 0:
            logging.warning("Skipping fold %d; no validation rows", fold_id)
            continue

        train_idx = train_df.index[train_mask].to_numpy()
        valid_idx = train_df.index[valid_mask].to_numpy()

        train_weights = weights.compute_sample_weights(
            train_df.loc[train_idx, "panel_month"],
            weight_cfg,
            train_end,
        )
        valid_weights = weights.compute_sample_weights(
            train_df.loc[valid_idx, "panel_month"],
            weight_cfg,
            valid_end,
        )
        train_target_min = float(train_df.loc[train_idx, target_column].min())

        folds.append(
            FoldInfo(
                fold_id=fold_id,
                train_idx=train_idx,
                valid_idx=valid_idx,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                train_weights=train_weights,
                valid_weights=valid_weights,
                train_target_min=train_target_min,
            )
        )

    if not folds:
        raise RuntimeError("No valid folds configured; check cv settings")
    logging.info("Constructed %d folds", len(folds))
    return folds


def transform_target(values: np.ndarray, log_cfg: Dict[str, object]) -> np.ndarray:
    """Log-transform target if requested."""

    if not log_cfg.get("enabled", False):
        return values.astype(np.float64)
    epsilon = float(log_cfg.get("epsilon", 1e-3))
    safe_values = np.maximum(values, 0.0) + epsilon
    return np.log(safe_values)


def inverse_transform(values: np.ndarray, log_cfg: Dict[str, object]) -> np.ndarray:
    """Inverse log transformation."""

    if not log_cfg.get("enabled", False):
        return values.astype(np.float64)
    epsilon = float(log_cfg.get("epsilon", 1e-3))
    restored = np.exp(values) - epsilon
    return np.maximum(restored, 0.0)


def apply_clipping(values: np.ndarray, clip_cfg: Dict[str, object]) -> np.ndarray:
    """Enforce minimum prediction values if configured."""

    if clip_cfg.get("enabled", False):
        min_value = float(clip_cfg.get("min_value", 0.0))
        values = np.maximum(values, min_value)
    return values
# ---------------------------------------------------------------------------
# Model construction and Optuna search
# ---------------------------------------------------------------------------


def sample_params(trial: optuna.Trial, param_spec: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    """Sample model hyperparameters from Optuna trial based on config spec."""

    params: Dict[str, object] = {}
    for name, spec in param_spec.items():
        ptype = spec.get("type")
        if ptype == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        elif ptype == "uniform":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
        elif ptype == "loguniform":
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=True,
            )
        elif ptype == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unsupported parameter type '{ptype}' for {name}")
    return params


def build_estimator(model_name: str, params: Dict[str, object], seed: int) -> object:
    """Instantiate estimator for the given model name."""

    if model_name == "lightgbm":
        final_params = params.copy()
        final_params.setdefault("random_state", seed)
        final_params.setdefault("n_jobs", -1)
        final_params.setdefault("verbosity", -1)
        return lgb.LGBMRegressor(**final_params)

    if model_name == "xgboost":
        final_params = {
            "objective": "reg:squarederror",
            "random_state": seed,
            "n_jobs": -1,
        }
        final_params.update(params)
        device_value = final_params.get("device")
        if isinstance(device_value, str):
            final_params["device"] = device_value.lower()
        return xgb.XGBRegressor(**final_params)

    if model_name == "catboost":
        if CatBoostRegressor is None:
            raise RuntimeError("CatBoost is not installed; cannot train catboost model")
        final_params = params.copy()
        final_params.setdefault("loss_function", "RMSE")
        final_params.setdefault("random_seed", seed)
        final_params.setdefault("verbose", False)
        final_params.setdefault("allow_writing_files", False)
        return CatBoostRegressor(**final_params)

    if model_name == "random_forest":
        final_params = params.copy()
        final_params.setdefault("random_state", seed)
        final_params.setdefault("n_jobs", -1)
        return RandomForestRegressor(**final_params)

    raise ValueError(f"Unsupported model '{model_name}'")
# ---------------------------------------------------------------------------
# Fold evaluation & validation aggregation
# ---------------------------------------------------------------------------


def evaluate_fold(
    model_name: str,
    params: Dict[str, object],
    fold: FoldInfo,
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_column: str,
    te_cfg: Dict[str, object],
    log_cfg: Dict[str, object],
    clip_cfg: Dict[str, object],
    seed: int,
) -> Dict[str, object]:
    """Train and evaluate a single fold with target encoding + optional scaler."""

    fold_train = train_df.loc[fold.train_idx].copy()
    fold_valid = train_df.loc[fold.valid_idx].copy()

    fold_train, fold_valid, _ = apply_target_encoding(fold_train, target_column, te_cfg, fold_valid)

    X_train = fold_train[feature_cols]
    y_train = fold_train[target_column].to_numpy(dtype=np.float64)
    X_valid = fold_valid[feature_cols]
    y_valid = fold_valid[target_column].to_numpy(dtype=np.float64)

    current_params = params.copy()
    if (
        model_name == "random_forest"
        and current_params.get("criterion") == "poisson"
        and fold.train_target_min <= 0
    ):
        logging.debug(
            "Fold %d: switching random_forest criterion to squared_error due to non-positive targets",
            fold.fold_id,
        )
        current_params["criterion"] = "squared_error"

    estimator = build_estimator(model_name, current_params, seed)

    scaler: Optional[StandardScaler] = None
    if model_name == "xgboost":
        scaler = StandardScaler()
        X_train_matrix = scaler.fit_transform(X_train)
        X_valid_matrix = scaler.transform(X_valid)
    else:
        X_train_matrix = X_train
        X_valid_matrix = X_valid

    y_train_trans = transform_target(y_train, log_cfg)
    estimator.fit(X_train_matrix, y_train_trans, sample_weight=fold.train_weights)

    raw_pred = estimator.predict(X_valid_matrix)
    raw_pred = np.asarray(raw_pred, dtype=np.float64)
    y_pred = inverse_transform(raw_pred, log_cfg)
    y_pred = apply_clipping(y_pred, clip_cfg)

    ape = metrics.ape_vector(y_valid, y_pred)
    report = metrics.score_report(y_valid, y_pred)

    return {
        "fold_id": fold.fold_id,
        "y_true": y_valid,
        "y_pred": y_pred,
        "sector_id": fold_valid["sector_id"].to_numpy(),
        "panel_month": fold_valid["panel_month"].to_numpy(),
        "weights": fold.valid_weights,
        "ape": ape,
        "report": report,
    }


def collect_validation_predictions(
    model_name: str,
    params: Dict[str, object],
    folds: List[FoldInfo],
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_column: str,
    te_cfg: Dict[str, object],
    log_cfg: Dict[str, object],
    clip_cfg: Dict[str, object],
    seed: int,
) -> pd.DataFrame:
    """Aggregate validation predictions across folds into a tidy dataframe."""

    records: List[pd.DataFrame] = []
    for fold in folds:
        result = evaluate_fold(
            model_name,
            params,
            fold,
            train_df,
            feature_cols,
            target_column,
            te_cfg,
            log_cfg,
            clip_cfg,
            seed,
        )
        fold_df = pd.DataFrame(
            {
                "model": model_name,
                "fold": result["fold_id"],
                "sector_id": result["sector_id"],
                "panel_month": result["panel_month"],
                "y_true": result["y_true"],
                "y_pred": result["y_pred"],
                "ape": result["ape"],
                "weights": result["weights"],
            }
        )
        records.append(fold_df)
    return pd.concat(records, ignore_index=True)


def summarize_validation(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Delegate summarisation to reporting helper."""

    return reporting.summarize_predictions(pred_df, group_keys=["model", "fold"])


def build_trial_history(study: optuna.Study) -> pd.DataFrame:
    """Flatten Optuna trial history for logging."""

    rows: List[Dict[str, object]] = []
    for trial in study.get_trials(deepcopy=False):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        row: Dict[str, object] = {
            "trial": trial.number,
            "score": float(trial.value),
        }
        for key, val in trial.params.items():
            row[f"param_{key}"] = val
        fold_reports = trial.user_attrs.get("fold_reports", [])
        for idx, report in enumerate(fold_reports, start=1):
            row[f"fold{idx}_score"] = report.get("score")
            row[f"fold{idx}_bad_ratio"] = report.get("bad_ratio")
        rows.append(row)
    return pd.DataFrame(rows)


def optimize_model(
    model_cfg: Dict[str, object],
    folds: List[FoldInfo],
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_column: str,
    te_cfg: Dict[str, object],
    log_cfg: Dict[str, object],
    clip_cfg: Dict[str, object],
    seed: int,
) -> optuna.Study:
    """Run Optuna hyperparameter search maximizing competition metric."""

    model_name = model_cfg["name"]
    param_spec = model_cfg.get("params", {})
    optimizer_cfg = model_cfg.get("optimizer", {})
    n_trials = int(optimizer_cfg.get("n_trials", 20))
    timeout = optimizer_cfg.get("timeout")

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, param_spec)
        scores: List[float] = []
        reports: List[Dict[str, object]] = []
        for fold in folds:
            result = evaluate_fold(
                model_name,
                params,
                fold,
                train_df,
                feature_cols,
                target_column,
                te_cfg,
                log_cfg,
                clip_cfg,
                seed,
            )
            scores.append(result["report"]["score"])
            reports.append(result["report"])
        trial.set_user_attr("fold_reports", reports)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", study_name=f"{model_name}_study")
    callbacks: List[object] = []
    if OptunaTqdmCallback is not None:
        callbacks.append(OptunaTqdmCallback(total=n_trials, leave=False))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=callbacks)
    logging.info(
        "Model %s best Optuna score: %.4f",
        model_name,
        float(study.best_value if study.best_trial else 0),
    )
    return study
# ---------------------------------------------------------------------------
# Final training and ensemble optimisation
# ---------------------------------------------------------------------------


def train_full_model(
    model_name: str,
    params: Dict[str, object],
    encoded_train_df: pd.DataFrame,
    feature_cols: List[str],
    target_column: str,
    log_cfg: Dict[str, object],
    seed: int,
    weight_cfg: Dict[str, object],
    train_end_month: pd.Timestamp,
) -> Tuple[object, Optional[StandardScaler]]:
    """Fit estimator on full dataset and optionally return fitted scaler."""

    X = encoded_train_df[feature_cols]
    y = encoded_train_df[target_column].to_numpy(dtype=np.float64)

    sample_weights = weights.compute_sample_weights(
        encoded_train_df["panel_month"],
        weight_cfg,
        train_end_month,
    )

    estimator = build_estimator(model_name, params, seed)

    scaler: Optional[StandardScaler] = None
    if model_name == "xgboost":
        scaler = StandardScaler()
        X_matrix = scaler.fit_transform(X)
    else:
        X_matrix = X

    y_trans = transform_target(y, log_cfg)
    estimator.fit(X_matrix, y_trans, sample_weight=sample_weights)
    return estimator, scaler


def optimize_ensemble(
    validation_predictions: Dict[str, pd.DataFrame],
    ensemble_cfg: Dict[str, object],
    logs_dir: Path,
    models_dir: Path,
) -> Dict[str, object]:
    """Solve constrained optimisation for ensemble weights."""

    model_names = list(validation_predictions.keys())
    if not model_names:
        raise RuntimeError("No validation predictions available for ensemble optimisation")

    merged: Optional[pd.DataFrame] = None
    for name in model_names:
        df = validation_predictions[name].copy()
        df = df[["sector_id", "panel_month", "fold", "y_true", "y_pred"]]
        df = df.rename(columns={"y_pred": f"pred_{name}"})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["sector_id", "panel_month", "fold", "y_true"], how="inner")

    if merged is None or merged.empty:
        raise RuntimeError("Unable to construct ensemble training matrix")

    y_true = merged["y_true"].to_numpy(dtype=np.float64)
    base_preds = [merged[f"pred_{name}"].to_numpy(dtype=np.float64) for name in model_names]

    bounds_cfg = ensemble_cfg.get("weight_bounds", [0.0, 1.0])
    lower, upper = float(bounds_cfg[0]), float(bounds_cfg[1])
    init_weights = ensemble_cfg.get("initial_weights", [])

    if len(init_weights) != len(model_names):
        init_weights = [1.0 / len(model_names)] * len(model_names)
    else:
        init_weights = np.asarray(init_weights, dtype=np.float64)
        init_weights = init_weights / init_weights.sum()

    def objective(weights: np.ndarray) -> float:
        pred = np.zeros_like(y_true)
        for w, base in zip(weights, base_preds):
            pred += w * base
        score = metrics.competition_metric(y_true, pred)["score"]
        return -score

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(lower, upper) for _ in model_names]

    result = minimize(
        objective,
        np.asarray(init_weights, dtype=np.float64),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": int(ensemble_cfg.get("optimization", {}).get("max_iter", 1000))},
    )

    if not result.success:
        logging.warning("Ensemble optimisation failed: %s; reverting to uniform weights", result.message)
        weights_array = np.ones(len(model_names), dtype=np.float64) / len(model_names)
    else:
        weights_array = result.x

    ensemble_pred = np.zeros_like(y_true)
    for weight, base in zip(weights_array, base_preds):
        ensemble_pred += weight * base
    uniform_pred = np.mean(np.vstack(base_preds), axis=0)

    merged["pred_ensemble"] = ensemble_pred
    merged["pred_uniform"] = uniform_pred

    ensemble_path = logs_dir / "ensemble_validation_predictions.parquet"
    merged.to_parquet(ensemble_path, index=False)
    logging.info("Saved ensemble validation predictions to %s", ensemble_path)

    ensemble_score = metrics.competition_metric(y_true, ensemble_pred)["score"]
    uniform_score = metrics.competition_metric(y_true, uniform_pred)["score"]

    weight_payload = {
        "model_names": model_names,
        "weights": weights_array.tolist(),
        "score": float(ensemble_score),
        "uniform_score": float(uniform_score),
        "solver_success": bool(result.success),
        "solver_message": result.message if hasattr(result, "message") else "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    weight_path = Path(ensemble_cfg.get("save", models_dir / "ensemble_weights.json"))
    if not weight_path.is_absolute():
        weight_path = models_dir / weight_path
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    with open(weight_path, "w", encoding="utf-8") as fh:
        json.dump(weight_payload, fh, ensure_ascii=False, indent=2)
    logging.info("Ensemble weights saved to %s", weight_path)

    return {
        "weights_path": weight_path,
        "weights": weights_array,
        "score": ensemble_score,
        "uniform_score": uniform_score,
        "merged_predictions": merged,
    }
# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_pipeline(
    base_dir: Path,
    config_path: Optional[str],
    model_filter: Optional[Sequence[str]] = None,
) -> Dict[str, Path]:
    """Execute end-to-end training according to specification."""

    progress = tqdm(total=len(STAGE_NAMES), desc="Training", ncols=120)

    progress.set_description(STAGE_NAMES[0])
    config = load_config(base_dir, config_path)
    logs_dir = Path(config["logs_dir"])
    models_dir = Path(config["artifacts_dir"])
    setup_logging(logs_dir)
    snapshot_path = save_config_snapshot(config, logs_dir)
    te_cfg = config.get("target_encoding", {})
    feature_cols = read_features(Path(config["features_path"]), config["target_column"], te_cfg)
    models_dir.mkdir(parents=True, exist_ok=True)
    progress.update(1)

    progress.set_description(STAGE_NAMES[1])
    train_df = prepare_dataset(config, feature_cols, te_cfg)
    progress.update(1)

    progress.set_description(STAGE_NAMES[2])
    folds = create_folds(train_df, config, config["target_column"])
    progress.update(1)

    encoded_full_df, _, te_mapping = apply_target_encoding(
        train_df.copy(),
        config["target_column"],
        te_cfg,
    )
    mapping_path = save_target_encoding_mapping(te_cfg, te_mapping, models_dir)

    numeric_feature_cols = [
        col
        for col in feature_cols
        if col in encoded_full_df.columns
        and pd.api.types.is_numeric_dtype(encoded_full_df[col])
    ]
    dropped_non_numeric = [col for col in feature_cols if col not in numeric_feature_cols]
    if dropped_non_numeric:
        logging.info("Dropping non-numeric features: %s", dropped_non_numeric)
    if not numeric_feature_cols:
        raise RuntimeError("No numeric features available after target encoding; check selected_features.txt")
    feature_cols = numeric_feature_cols

    models_cfg: List[Dict[str, object]] = config.get("models", [])
    if model_filter:
        allow = {name.lower() for name in model_filter}
        models_cfg = [cfg for cfg in models_cfg if cfg.get("name", "").lower() in allow]
    if not models_cfg:
        raise RuntimeError("No models selected for training")

    validation_predictions: Dict[str, pd.DataFrame] = {}
    model_summaries: Dict[str, Dict[str, float]] = {}
    weight_cfg = config.get("weights", {})
    log_cfg = config.get("log_transform", {})
    clip_cfg = config.get("prediction_clipping", {})
    seed = int(config.get("seed", 42))
    train_end_month = pd.Timestamp(config["train_end_month"])

    progress.set_description(STAGE_NAMES[3])
    for model_cfg in models_cfg:
        model_name = model_cfg["name"]
        logging.info("=== Optimising model: %s ===", model_name)
        study = optimize_model(
            model_cfg,
            folds,
            train_df,
            feature_cols,
            config["target_column"],
            te_cfg,
            log_cfg,
            clip_cfg,
            seed,
        )
        best_params = study.best_params

        validation_df = collect_validation_predictions(
            model_name,
            best_params,
            folds,
            train_df,
            feature_cols,
            config["target_column"],
            te_cfg,
            log_cfg,
            clip_cfg,
            seed,
        )
        val_path = logs_dir / f"{model_name}_validation_predictions.parquet"
        validation_df.to_parquet(val_path, index=False)
        validation_predictions[model_name] = validation_df

        summary_df = summarize_validation(validation_df)
        reporting.log_summary(summary_df)
        mean_score = float(summary_df["score"].mean()) if not summary_df.empty else 0.0
        model_summaries[model_name] = {
            "best_score": float(study.best_value) if study.best_trial else 0.0,
            "validation_score": mean_score,
        }

        history_df = build_trial_history(study)
        history_path = logs_dir / f"{model_name}_optuna_history.csv"
        history_df.to_csv(history_path, index=False)

        estimator, scaler = train_full_model(
            model_name,
            best_params,
            encoded_full_df,
            feature_cols,
            config["target_column"],
            log_cfg,
            seed,
            weight_cfg,
            train_end_month,
        )
        model_path = models_dir / f"{model_name}_final.pkl"
        joblib.dump(estimator, model_path)

        if scaler is not None:
            scaler_path = models_dir / f"{model_name}_scaler.pkl"
            joblib.dump(scaler, scaler_path)
            logging.info("Saved scaler for %s to %s", model_name, scaler_path)

        params_path = models_dir / f"{model_name}_best_params.json"
        with open(params_path, "w", encoding="utf-8") as fh:
            json.dump(best_params, fh, ensure_ascii=False, indent=2)

        metadata_path = models_dir / f"{model_name}_metadata.json"
        metadata = {
            "best_params_path": str(params_path),
            "model_path": str(model_path),
            "best_score": float(study.best_value) if study.best_trial else 0.0,
            "validation_score": mean_score,
            "train_samples": int(len(encoded_full_df)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, ensure_ascii=False, indent=2)
        logging.info("Finished training %s", model_name)

    progress.update(1)

    progress.set_description(STAGE_NAMES[4])
    ensemble_result = optimize_ensemble(
        validation_predictions,
        config.get("ensemble", {}),
        logs_dir,
        models_dir,
    )
    progress.update(1)

    progress.set_description(STAGE_NAMES[5])
    summary_payload = {
        "models": model_summaries,
        "ensemble_score": float(ensemble_result["score"]),
        "ensemble_weights": ensemble_result["weights"].tolist(),
        "config_snapshot": str(snapshot_path),
        "target_encoding": str(mapping_path) if mapping_path else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = logs_dir / "train_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, ensure_ascii=False, indent=2)
    logging.info("Training summary saved to %s", summary_path)
    progress.update(1)
    progress.close()

    outputs: Dict[str, Path] = {
        "config_snapshot": snapshot_path,
        "train_summary": summary_path,
        "ensemble_weights": ensemble_result["weights_path"],
        "ensemble_predictions": logs_dir / "ensemble_validation_predictions.parquet",
    }
    for model_name in validation_predictions.keys():
        outputs[f"{model_name}_model"] = models_dir / f"{model_name}_final.pkl"
        outputs[f"{model_name}_best_params"] = models_dir / f"{model_name}_best_params.json"
        outputs[f"{model_name}_metadata"] = models_dir / f"{model_name}_metadata.json"
    return outputs
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="China Real Estate Ensemble Training")
    parser.add_argument("--config", type=str, default=None, help="Path to training config")
    parser.add_argument("--base-dir", type=str, default=".", help="Project root directory")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated subset of models to train (e.g., lightgbm,xgboost)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    base_dir = Path(args.base_dir)
    model_filter = [m.strip() for m in args.models.split(",")] if args.models else None
    run_pipeline(base_dir, args.config, model_filter)


if __name__ == "__main__":
    main()
