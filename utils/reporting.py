"""utils.reporting �W�����ϰ��X��T"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from . import metrics

LOGGER = logging.getLogger(__name__)
REPORTING_VERSION = "1.0"


def _ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """�ˬd DataFrame �O�_���w�˻ݪ�欰"""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"缺少必要欄位: {missing}")


def summarize_predictions(pred_df: pd.DataFrame, group_keys: List[str] | None = None) -> pd.DataFrame:
    """�ھ٧���飺�ƭȥ�ƭȫH�ۡA�n�g metrics.score_report"""
    if group_keys is None:
        group_keys = ["model"]
    _ensure_columns(pred_df, list({"y_true", "y_pred"} | set(group_keys)))
    has_weights = "weights" in pred_df.columns

    records = []
    grouped = pred_df.groupby(group_keys, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        metrics_dict = metrics.score_report(group["y_true"], group["y_pred"])
        record = {key: value for key, value in zip(group_keys, keys)}
        record.update(metrics_dict)
        if has_weights:
            weights = group["weights"].to_numpy(dtype=float)
            diff = group["y_pred"].to_numpy(dtype=float) - group["y_true"].to_numpy(dtype=float)
            mse = np.average(diff**2, weights=weights)
            mae_value = np.average(np.abs(diff), weights=weights)
            record["rmse_weighted"] = float(np.sqrt(mse))
            record["mae_weighted"] = float(mae_value)
        records.append(record)
    summary_df = pd.DataFrame(records)
    return summary_df


def sector_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
    """�ھڤ��a���e��ܥH�ӱ��"""
    _ensure_columns(pred_df, ["sector_id", "y_true", "y_pred"])
    group_keys = ["sector_id"]
    if "model" in pred_df.columns:
        group_keys.append("model")
    summary = summarize_predictions(pred_df, group_keys=group_keys)
    return summary.sort_values(by="score", ascending=True)


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """�w�M�ɮ׾��A�@�ҤF�T���ؿ�"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("已將報表存成 CSV: %s", output_path)
    return output_path


def save_plots(pred_df: pd.DataFrame, output_dir: str | Path, enable: bool = True) -> list[Path]:
    """�өԨ� matplotlib �b�����ɮתϥΤ��y���"""
    saved_paths: list[Path] = []
    if not enable:
        LOGGER.info("已停用繪圖，略過產出 PNG")
        return saved_paths
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:  # pragma: no cover
        LOGGER.warning("未安裝 matplotlib，略過繪圖程序")
        return saved_paths

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = pred_df.copy()
    if "y_true" not in plot_df.columns or plot_df["y_true"].dropna().empty:
        LOGGER.warning("缺少真值資料，略過誤差圖繪製")
        return saved_paths
    if "ape" not in plot_df.columns:
        plot_df["ape"] = metrics.ape_vector(plot_df["y_true"], plot_df["y_pred"])

    if "model" in plot_df.columns:
        model_groups = plot_df.groupby("model")
    else:
        model_groups = [("ensemble", plot_df)]

    for model_name, group in model_groups:
        plt.figure(figsize=(6, 4))
        plt.hist(group["ape"], bins=30, color="#1f77b4", alpha=0.8)
        plt.title(f"APE Histogram - {model_name}")
        plt.xlabel("APE")
        plt.ylabel("Frequency")
        hist_path = output_dir / f"ape_hist_{model_name}.png"
        plt.tight_layout()
        plt.savefig(hist_path, dpi=150)
        plt.close()
        saved_paths.append(hist_path)

    if "panel_month" in plot_df.columns:
        plt.figure(figsize=(8, 4))
        monthly = plot_df.groupby("panel_month").agg({"y_true": "mean", "y_pred": "mean"})
        plt.plot(monthly.index, monthly["y_true"], label="Actual")
        plt.plot(monthly.index, monthly["y_pred"], label="Prediction")
        plt.title("Monthly Mean Comparison")
        plt.xlabel("Panel Month")
        plt.ylabel("Value")
        plt.legend()
        monthly_path = output_dir / "monthly_trend.png"
        plt.tight_layout()
        plt.savefig(monthly_path, dpi=150)
        plt.close()
        saved_paths.append(monthly_path)

    sector_metrics = sector_summary(plot_df)
    worst = sector_metrics.nsmallest(10, "score")
    plt.figure(figsize=(8, 4))
    plt.barh(worst["sector_id"].astype(str), worst["score"], color="#d62728")
    plt.title("Worst Sectors by Score")
    plt.xlabel("Score")
    plt.ylabel("Sector")
    plt.tight_layout()
    sector_path = output_dir / "sector_worst.png"
    plt.savefig(sector_path, dpi=150)
    plt.close()
    saved_paths.append(sector_path)

    return saved_paths


def log_summary(summary_df: pd.DataFrame) -> None:
    """�Ϊť�h�����D���ȬݨD�p��"""
    columns = ["model", "score", "scaled_mape", "bad_ratio", "good_fraction", "rmse", "mae"]
    missing = [col for col in columns if col not in summary_df.columns]
    if missing:
        LOGGER.warning("log_summary 缺少欄位，無法完整輸出: %s", missing)
    for _, row in summary_df.iterrows():
        LOGGER.info(
            "模型=%s | score=%.4f | scaled_mape=%.4f | bad_ratio=%.4f | good_fraction=%.4f | rmse=%.4f | mae=%.4f",
            row.get("model", "N/A"),
            row.get("score", float("nan")),
            row.get("scaled_mape", float("nan")),
            row.get("bad_ratio", float("nan")),
            row.get("good_fraction", float("nan")),
            row.get("rmse", float("nan")),
            row.get("mae", float("nan")),
        )

