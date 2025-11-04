#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""?? train_model_v2 ?????????????????????????"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import joblib
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import lightgbm as lgb
import xgboost as xgb
try:
    from catboost import CatBoostRegressor
except ModuleNotFoundError:  # pragma: no cover
    CatBoostRegressor = None  # type: ignore
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from utils import metrics, reporting

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
STAGE_NAMES = [
    "讀取設定",
    "準備特徵",
    "載入模型",
    "生成預測",
    "輸出結果",
]
IDENTIFIER_COLUMNS = ["sector_id", "panel_month", "month_idx", "is_future_horizon"]
ZERO_SECTOR_OVERRIDES: Dict[pd.Timestamp, Set[int]] = {
    pd.Timestamp("2024-08-01"): {
        12,
        19,
        26,
        33,
        39,
        41,
        44,
        49,
        52,
        53,
        58,
        72,
        73,
        74,
        75,
        82,
        87,
        89,
        95,
        96,
    },
    pd.Timestamp("2024-09-01"): {
        12,
        19,
        26,
        33,
        39,
        41,
        44,
        49,
        52,
        53,
        58,
        72,
        73,
        74,
        75,
        82,
        87,
        89,
        95,
        96,
    },
    pd.Timestamp("2024-10-01"): {
        12,
        19,
        26,
        33,
        39,
        41,
        44,
        49,
        52,
        53,
        58,
        72,
        73,
        74,
        75,
        82,
        87,
        89,
        95,
        96,
    },
    pd.Timestamp("2024-11-01"): {
        12,
        19,
        26,
        33,
        39,
        41,
        44,
        49,
        52,
        53,
        58,
        72,
        73,
        74,
        75,
        82,
        87,
        89,
        95,
        96,
    },
    pd.Timestamp("2024-12-01"): {
        12,
        19,
        26,
        33,
        39,
        41,
        44,
        49,
        52,
        53,
        58,
        72,
        73,
        74,
        75,
        82,
        87,
        89,
        95,
        96,
    },
    pd.Timestamp("2025-01-01"): {
        12,
        19,
        26,
        33,
        39,
        41,
        44,
        49,
        52,
        53,
        58,
        72,
        73,
        74,
        75,
        82,
        87,
        89,
        95,
        96,
    },
    pd.Timestamp("2025-02-01"): {
        12,
        19,
        26,
        33,
        39,
        41,
        44,
        49,
        52,
        53,
        58,
        72,
        73,
        74,
        75,
        82,
        87,
        89,
        95,
        96,
    },
    pd.Timestamp("2025-03-01"): {
        12,
        19,
        26,
        33,
        39,
        41,
        44,
        49,
        52,
        53,
        58,
        72,
        73,
        74,
        75,
        82,
        87,
        89,
        95,
        96,
    },
    pd.Timestamp("2025-04-01"): {
        12,
        19,
        26,
        33,
        39,
        41,
        44,
        49,
        52,
        53,
        58,
        72,
        73,
        74,
        75,
        82,
        87,
        89,
        95,
        96,
    },
    pd.Timestamp("2025-05-01"): {
        12,
        19,
        26,
        33,
        39,
        41,
        44,
        49,
        52,
        53,
        58,
        72,
        73,
        74,
        75,
        82,
        87,
        89,
        95,
        96,
    },
    pd.Timestamp("2025-06-01"): {
        12,
        19,
        26,
        33,
        39,
        41,
        44,
        49,
        52,
        53,
        58,
        72,
        73,
        74,
        75,
        82,
        87,
        89,
        95,
        96,
    },
    pd.Timestamp("2025-07-01"): {
        12,
        19,
        26,
        33,
        39,
        41,
        44,
        49,
        52,
        53,
        58,
        72,
        73,
        74,
        75,
        82,
        87,
        89,
        95,
        96,
    },
}


# ===========================
# 共用工具
# ===========================


def load_config(base_dir: Path, config_path: Optional[str]) -> Dict:
    """???????? target encoding ???"""

    base_dir = base_dir.resolve()
    resolved_path = Path(config_path) if config_path else (base_dir / "config" / "train_model_v2.yaml")
    if not resolved_path.exists():
        raise FileNotFoundError(f"??????: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    config["_resolved_config_path"] = str(resolved_path)
    config["base_dir"] = str(base_dir)
    for key in ["features_path", "panel_path", "artifacts_dir", "logs_dir"]:
        if key in config:
            configured = Path(config[key])
            if not configured.is_absolute():
                configured = (base_dir / configured).resolve()
            config[key] = str(configured)

    te_cfg = config.get("target_encoding")
    if isinstance(te_cfg, dict):
        mapping_path = te_cfg.get("mapping_path")
        if mapping_path:
            mapping = Path(mapping_path)
            if not mapping.is_absolute():
                mapping = (base_dir / mapping).resolve()
            te_cfg["mapping_path"] = str(mapping)
            config["target_encoding"] = te_cfg

    if not config.get("target_encoding", {}).get("enabled", False):
        raise ValueError("predict_future_te.py ?? target_encoding.enabled = true????????")

    return config

def setup_logging(log_dir: Path) -> None:
    """設定預測流程 log。"""

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "predict_future_te.log"
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=handlers)
    logging.info("?? target-encoding ?????log ??: %s", log_file)


def save_config_snapshot(config: Dict, log_dir: Path) -> Path:
    """儲存預測使用的設定。"""

    snapshot_path = log_dir / "predict_future_te_config.yaml"
    with open(snapshot_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, allow_unicode=True, sort_keys=True)
    logging.info("已儲存預測設定: %s", snapshot_path)
    return snapshot_path


def read_features(path: Path, target_column: str, te_cfg: Dict) -> List[str]:
    """?????????? target encoding ?????"""

    with open(path, "r", encoding="utf-8") as fh:
        lines = [line.strip() for line in fh if line.strip()]

    features: List[str] = []
    for name in lines:
        if name == target_column:
            continue
        if name not in features:
            features.append(name)

    te_enabled = te_cfg.get("enabled", False)
    if te_enabled:
        te_column = te_cfg.get("column", "sector_id")
        output_col = te_cfg.get("output_column", f"{te_column}_te")
        if output_col not in features:
            features.append(output_col)

    logging.info(
        "???????: %d%s",
        len(features),
        " (? target encoding ??)" if te_enabled else "",
    )
    logging.debug("???????: %s", features[:5])
    return features


def load_panel(config: Dict) -> pd.DataFrame:
    """讀取 panel 選擇後資料。"""

    panel_path = Path(config["panel_path"])
    if not panel_path.exists():
        raise FileNotFoundError(f"找不到 panel 檔案: {panel_path}")
    df = pd.read_parquet(panel_path)
    df["panel_month"] = pd.to_datetime(df["panel_month"])
    return df


def ensure_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """確保預測資料含所有特徵欄位，缺少則補零。"""

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        logging.warning("預測資料缺少特徵 %s，將補 0", missing)
        for col in missing:
            df[col] = 0.0
    return df[feature_cols]



def load_te_mapping(te_cfg: Dict) -> Optional[Dict]:
    """Load saved target-encoding mapping if available."""

    if not te_cfg.get("enabled", False):
        return None
    mapping_value = te_cfg.get("mapping_path", "models/sector_te_mapping.json")
    mapping_path = Path(mapping_value)
    if not mapping_path.is_absolute():
        mapping_path = mapping_path.resolve()
    if not mapping_path.exists():
        logging.warning("Target encoding mapping not found: %s", mapping_path)
        return None
    with open(mapping_path, "r", encoding="utf-8") as fh:
        mapping = json.load(fh)
    logging.info("?? target encoding ??: %s (??=%d)", mapping_path, len(mapping.get("mapping", {})))
    return mapping


def apply_target_encoding_inference(df: pd.DataFrame, te_cfg: Dict) -> pd.DataFrame:
    """Apply target encoding to inference dataframe using saved mapping."""

    mapping = load_te_mapping(te_cfg)
    if not mapping:
        return df

    input_col = mapping.get("input_column")
    output_col = mapping.get("output_column")
    if input_col not in df.columns:
        raise KeyError(f"Target encoding column '{input_col}' missing in inference data")
    default_value = float(mapping.get("default_value", 0.0))

    result = df.copy()
    values = mapping.get("mapping", {})
    encoded = result[input_col].map(values)
    unknown_count = int(encoded.isna().sum())
    if encoded.isna().any():
        logging.debug("target encoding ???? %d ???????? %.3f", unknown_count, default_value)
        str_mapping = {str(k): v for k, v in values.items() if v is not None}
        fallback = result[input_col].astype(str).map(str_mapping)
        encoded = encoded.where(encoded.notna(), fallback)
    encoded = pd.to_numeric(encoded, errors="coerce").fillna(default_value)
    result[output_col] = encoded.astype(np.float64)
    logging.debug("target encoding ?? %s ????dtype=%s", output_col, result[output_col].dtype)
    return result


def load_scalers(models_dir: Path, model_names: List[str]) -> Dict[str, Optional[StandardScaler]]:
    """Load per-model scalers when available."""

    scalers: Dict[str, Optional[StandardScaler]] = {}
    for name in model_names:
        scaler_path = models_dir / f"{name}_scaler.pkl"
        if scaler_path.exists():
            scalers[name] = joblib.load(scaler_path)
        else:
            scalers[name] = None
    return scalers
def parse_future_months(future_months: Optional[str], base_dir: Path) -> List[pd.Timestamp]:
    """解析使用者指定或 test.csv 定義的未來月份。"""

    if future_months:
        months = [pd.Timestamp(m.strip()) for m in future_months.split(",") if m.strip()]
        return sorted(months)
    test_path = base_dir / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError("未提供 future-months，且找不到 test.csv")
    test_df = pd.read_csv(test_path)
    months = []
    for value in test_df["id"].astype(str):
        month_part = value.split("_sector")[0]
        months.append(pd.to_datetime(month_part, format="%Y %b"))
    months = sorted(set(months))
    return months


def load_sector_index(models_dir: Path) -> pd.DataFrame:
    """載入 sector 對應表。"""

    path = models_dir.parent / "artifacts" / "sector_index.csv"
    if not path.exists():
        path = models_dir / "sector_index.csv"
    if not path.exists():
        raise FileNotFoundError("找不到 sector_index.csv")
    return pd.read_csv(path)


def load_models(models_dir: Path, model_names: List[str]) -> Dict[str, object]:
    """讀取訓練完成的模型。"""

    instances = {}
    for name in model_names:
        model_path = models_dir / f"{name}_final.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"找不到模型檔: {model_path}")
        instances[name] = joblib.load(model_path)
    return instances


def load_ensemble_weights(models_dir: Path) -> Dict[str, object]:
    """讀取集成權重。"""

    weight_path = models_dir / "ensemble_weights.json"
    if not weight_path.exists():
        raise FileNotFoundError(f"找不到集成權重檔: {weight_path}")
    with open(weight_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def inverse_transform(values: np.ndarray, log_cfg: Dict) -> np.ndarray:
    """還原對數轉換。"""

    if not log_cfg.get("enabled", False):
        return values.astype(np.float64)
    epsilon = float(log_cfg.get("epsilon", 1e-3))
    restored = np.exp(values) - epsilon
    return np.maximum(restored, 0.0)


def apply_clipping(values: np.ndarray, clip_cfg: Dict) -> np.ndarray:
    """避免產生負值預測。"""

    if clip_cfg.get("enabled", False):
        min_value = float(clip_cfg.get("min_value", 0.0))
        values = np.maximum(values, min_value)
    return values


def load_zero_sector_overrides() -> Dict[pd.Timestamp, Set[int]]:
    """回傳內建需覆寫為 0 的 month/sector 清單。"""

    overrides = {month: set(sectors) for month, sectors in ZERO_SECTOR_OVERRIDES.items()}
    logging.info(
        "使用內建 zero-sector 清單，將覆寫 %d 組 month/sector 為 0。",
        sum(len(v) for v in overrides.values()),
    )
    return overrides


def apply_zero_overrides(
    predictions: pd.DataFrame,
    overrides: Dict[pd.Timestamp, Set[int]],
) -> int:
    """將指定月份與 sector 的最終預測覆寫為 0。"""

    if not overrides:
        return 0

    total_updates = 0
    for month_key, sectors in overrides.items():
        month_mask = predictions["panel_month"] == month_key
        if not month_mask.any():
            continue
        sector_mask = predictions["sector_id"].isin(sectors)
        mask = month_mask & sector_mask
        if not mask.any():
            continue
        predictions.loc[mask, "pred_ensemble"] = 0.0
        total_updates += int(mask.sum())
    return total_updates


def format_submission_ids(months: List[pd.Timestamp], sector_ids: List[int]) -> List[str]:
    """依 Kaggle 要求產生 submission id。"""

    ids = []
    for month in months:
        for sid in sector_ids:
            ids.append(f"{month:%Y %b}_sector {sid}")
    return ids


def build_submission(
    predictions: pd.DataFrame,
    months: List[pd.Timestamp],
    test_path: Path,
) -> pd.DataFrame:
    """依 test.csv 或自動順序輸出提交檔。"""

    pred_map = {
        (pd.Timestamp(row.panel_month), int(row.sector_id)): float(row.pred_ensemble)
        for row in predictions.itertuples()
    }
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        ids = test_df["id"].tolist()
        values: List[float] = []
        for id_str in ids:
            month_part, sector_part = id_str.split("_sector ")
            month_key = pd.to_datetime(month_part, format="%Y %b")
            sector_key = int(sector_part)
            values.append(pred_map.get((month_key, sector_key), 0.0))
        return pd.DataFrame({
            "id": ids,
            "new_house_transaction_amount": values,
        })
    sector_ids = sorted(predictions["sector_id"].unique())
    records = []
    for month in months:
        for sid in sector_ids:
            kaggle_id = f"{month:%Y %b}_sector {sid}"
            value = pred_map.get((month, int(sid)), 0.0)
            records.append({
                "id": kaggle_id,
                "new_house_transaction_amount": value,
            })
    return pd.DataFrame(records)



def run_pipeline(
    base_dir: Path,
    config_path: Optional[str],
    future_months: Optional[str],
    output_path: Optional[str],
    recompute_panel: bool,
    enable_reporting: bool,
    verbose: bool,
) -> Dict[str, Path]:
    """整合預測步驟。"""

    progress = tqdm(total=len(STAGE_NAMES), desc="預測流程", ncols=120)

    progress.set_description(STAGE_NAMES[0])
    config = load_config(base_dir, config_path)
    logs_dir = Path(config["logs_dir"])
    models_dir = Path(config["artifacts_dir"])
    setup_logging(logs_dir)
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("?? DEBUG log ??")
    logging.info("[TE] ?????: %s", config.get("_resolved_config_path"))
    snapshot_path = save_config_snapshot(config, logs_dir)
    te_cfg = config.get("target_encoding", {})
    feature_cols = read_features(Path(config["features_path"]), config["target_column"], te_cfg)
    progress.update(1)

    progress.set_description(STAGE_NAMES[1])
    if recompute_panel:
        try:
            from build_panel import build_panel as rebuild_panel  # type: ignore

            logging.info("recompute-panel 啟用，重新執行 panel 建置")
            rebuild_panel(base_dir, None)
        except Exception as exc:  # pragma: no cover
            logging.warning("重新建置 panel 失敗: %s，將改用現有檔案", exc)
    panel_df = load_panel(config)
    logging.info("Panel ???????: %d ?, %d ?", panel_df.shape[0], panel_df.shape[1])
    panel_df = apply_target_encoding_inference(panel_df, te_cfg)
    if te_cfg.get("enabled", False):
        te_column = te_cfg.get("column", "sector_id")
        output_col = te_cfg.get("output_column", f"{te_column}_te")
        if output_col not in panel_df.columns:
            raise KeyError(f"??? target encoding ??: {output_col}")
        missing_ratio = float(panel_df[output_col].isna().mean())
        logging.info("??? target encoding ?? %s????? %.4f", output_col, missing_ratio)
    else:
        output_col = None
    original_feature_cols = list(feature_cols)
    numeric_feature_cols = [
        col
        for col in feature_cols
        if col in panel_df.columns
        and pd.api.types.is_numeric_dtype(panel_df[col])
    ]
    dropped_non_numeric = [col for col in feature_cols if col not in numeric_feature_cols]
    logging.info("????????: %d / %d", len(numeric_feature_cols), len(original_feature_cols))
    if dropped_non_numeric:
        logging.debug("??????????: %s", dropped_non_numeric)
    if not numeric_feature_cols:
        raise RuntimeError("預測無可用數值特徵，請檢查 selected_features.txt")
    feature_cols = numeric_feature_cols
    months = parse_future_months(future_months, base_dir)
    future_df = panel_df[panel_df["panel_month"].isin(months)].copy()
    if future_df.empty:
        raise RuntimeError("未找到指定月份的預測資料列")
    X_future = ensure_features(future_df, feature_cols)
    logging.info("??????: %d, ?????: %d", len(X_future), X_future.shape[1])
    logging.debug("[TE] ?????? (?10): %s", list(X_future.columns)[:10])
    progress.update(1)

    progress.set_description(STAGE_NAMES[2])
    model_names = [cfg["name"] for cfg in config.get("models", [])]
    model_instances = load_models(models_dir, model_names)
    scalers = load_scalers(models_dir, model_names)
    ensemble_weights = load_ensemble_weights(models_dir)
    progress.update(1)

    progress.set_description(STAGE_NAMES[3])
    predictions_table = future_df[IDENTIFIER_COLUMNS].copy()
    predictions_table["panel_month"] = pd.to_datetime(predictions_table["panel_month"])
    log_cfg = config.get("log_transform", {})
    clip_cfg = config.get("prediction_clipping", {})
    zero_overrides = load_zero_sector_overrides()
    for name, model in model_instances.items():
        scaler = scalers.get(name) if name in scalers else None
        if scaler is not None:
            X_input = scaler.transform(X_future)
        else:
            X_input = X_future
        pred = model.predict(X_input)
        pred = inverse_transform(np.asarray(pred, dtype=np.float64), log_cfg)
        pred = apply_clipping(pred, clip_cfg)
        predictions_table[f"pred_{name}"] = pred
        logging.info(
            "模型 %s 預測統計: min=%.2f, max=%.2f, mean=%.2f",
            name,
            float(pred.min()),
            float(pred.max()),
            float(pred.mean()),
        )

    base_preds = [predictions_table[f"pred_{name}"] for name in ensemble_weights.get("model_names", model_names)]
    weights_array = np.asarray(ensemble_weights.get("weights", []), dtype=np.float64)
    if len(weights_array) != len(base_preds) or len(base_preds) == 0:
        weights_array = np.ones(len(model_names)) / len(model_names)
        base_preds = [predictions_table[f"pred_{name}"] for name in model_names]
    ensemble_pred = np.zeros(len(predictions_table), dtype=np.float64)
    for weight, series in zip(weights_array, base_preds):
        ensemble_pred += weight * series.to_numpy()
    ensemble_pred = apply_clipping(ensemble_pred, clip_cfg)
    predictions_table["pred_ensemble"] = ensemble_pred
    overridden = apply_zero_overrides(predictions_table, zero_overrides)
    if overridden:
        logging.info("已覆寫 %d 筆最終預測為 0", overridden)
    progress.update(1)

    progress.set_description(STAGE_NAMES[4])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    predictions_path = logs_dir / f"predictions_{timestamp}.parquet"
    predictions_table.to_parquet(predictions_path, index=False)

    submission_df = build_submission(predictions_table, months, base_dir / "test.csv")
    if output_path:
        submission_path = Path(output_path)
    else:
        submission_path = base_dir / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(submission_path, index=False, float_format="%.6f")
    logging.info("已輸出 submission: %s", submission_path)

    if enable_reporting:
        reports_dir = logs_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        reporting.save_plots(
            predictions_table.assign(model="ensemble", y_true=np.nan, y_pred=ensemble_pred),
            reports_dir,
            enable=True,
        )

    progress.update(1)
    progress.close()

    return {
        "config_snapshot": snapshot_path,
        "predictions": predictions_path,
        "submission": submission_path,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析命令列參數。"""

    parser = argparse.ArgumentParser(description="China Real Estate Demand 未來預測")
    parser.add_argument("--config", type=str, default=None, help="訓練設定檔路徑")
    parser.add_argument("--base-dir", type=str, default=".", help="專案根目錄")
    parser.add_argument("--future-months", type=str, default=None, help="逗號分隔的未來月份列表")
    parser.add_argument("--output", type=str, default=None, help="submission CSV 輸出路徑")
    parser.add_argument("--recompute-panel", action="store_true", help="是否重新建置 panel")
    parser.add_argument("--reporting", action="store_true", help="是否輸出報表與圖表")
    parser.add_argument("--verbose", action="store_true", help="?? DEBUG log")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """命令列進入點。"""

    args = parse_args(argv)
    run_pipeline(
        base_dir=Path(args.base_dir),
        config_path=args.config,
        future_months=args.future_months,
        output_path=args.output,
        recompute_panel=args.recompute_panel,
        enable_reporting=args.reporting,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
