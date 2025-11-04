"""utils.metrics �۽s����}�C"""

from __future__ import annotations

import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None  # type: ignore

METRICS_VERSION = "1.0"


def _to_numpy(array_like: "np.ndarray | pd.Series | list | tuple") -> np.ndarray:
    """�N�J�ƶi�Φh��᭱ np.ndarray �H���B"""
    if pd is not None and isinstance(array_like, pd.Series):
        return array_like.to_numpy(dtype=np.float64, copy=False)
    return np.asarray(array_like, dtype=np.float64)


def validate_inputs(y_true: "np.ndarray | pd.Series | list | tuple", y_pred: "np.ndarray | pd.Series | list | tuple") -> tuple[np.ndarray, np.ndarray]:
    """�ˬd y_true �M y_pred �f�j�ε��e�ۦP�A�۩w np.ndarray."""
    true_array = _to_numpy(y_true)
    pred_array = _to_numpy(y_pred)
    if true_array.shape != pred_array.shape:
        raise ValueError("y_true 與 y_pred 長度或形狀不一致")
    if true_array.ndim != 1:
        true_array = true_array.reshape(-1)
    if pred_array.ndim != 1:
        pred_array = pred_array.reshape(-1)
    if true_array.size == 0:
        raise ValueError("輸入長度不可為零")
    return true_array, pred_array


def ape_vector(y_true: "np.ndarray | pd.Series | list | tuple", y_pred: "np.ndarray | pd.Series | list | tuple", epsilon: float = 1e-8) -> np.ndarray:
    """�b�@�Ӽƶq�p�� APE �Ӳv�ӰѪR epsilon �������"""
    true_array, pred_array = validate_inputs(y_true, y_pred)
    denominator = np.maximum(np.abs(true_array), epsilon)
    return np.abs(pred_array - true_array) / denominator


def mape(y_true: "np.ndarray | pd.Series | list | tuple", y_pred: "np.ndarray | pd.Series | list | tuple", epsilon: float = 1e-8) -> float:
    """�p�� MAPE �]�W�P�A�ϩ웻�����"""
    return float(np.mean(ape_vector(y_true, y_pred, epsilon=epsilon)))


def rmse(y_true: "np.ndarray | pd.Series | list | tuple", y_pred: "np.ndarray | pd.Series | list | tuple") -> float:
    """�p�� RMSE �q���"""
    true_array, pred_array = validate_inputs(y_true, y_pred)
    return float(np.sqrt(np.mean((pred_array - true_array) ** 2)))


def mae(y_true: "np.ndarray | pd.Series | list | tuple", y_pred: "np.ndarray | pd.Series | list | tuple") -> float:
    """�p�� MAE �q���"""
    true_array, pred_array = validate_inputs(y_true, y_pred)
    return float(np.mean(np.abs(pred_array - true_array)))


def competition_metric(y_true: "np.ndarray | pd.Series | list | tuple", y_pred: "np.ndarray | pd.Series | list | tuple") -> dict[str, float]:
    """�ӽÓ�����W��W�q spec �n�D�쥻�۰���bscore����凳"""
    true_array, pred_array = validate_inputs(y_true, y_pred)
    denom = np.where(true_array == 0.0, 1.0, np.abs(true_array))
    ape = np.abs(pred_array - true_array) / denom
    bad_mask = ape > 1.0
    bad_ratio = float(np.mean(bad_mask))
    if bad_ratio > 0.3 or bad_mask.size == 0:
        return {
            "score": 0.0,
            "scaled_mape": float("nan"),
            "bad_ratio": bad_ratio,
            "good_fraction": float(1.0 - bad_ratio),
        }
    good_mask = ~bad_mask
    good_count = np.sum(good_mask)
    if good_count == 0:
        return {
            "score": 0.0,
            "scaled_mape": float("nan"),
            "bad_ratio": bad_ratio,
            "good_fraction": 0.0,
        }
    scaled_mape = float(np.mean(ape[good_mask]) / np.mean(good_mask))
    score = max(0.0, 1.0 - scaled_mape)
    return {
        "score": float(score),
        "scaled_mape": scaled_mape,
        "bad_ratio": bad_ratio,
        "good_fraction": float(np.mean(good_mask)),
    }


def score_report(y_true: "np.ndarray | pd.Series | list | tuple", y_pred: "np.ndarray | pd.Series | list | tuple") -> dict[str, float]:
    """�۫H competition_metric �M�㯫 RMSE/MAE/MAPE �s���ܱ��"""
    true_array, pred_array = validate_inputs(y_true, y_pred)
    base = competition_metric(true_array, pred_array)
    report = {
        "score": base["score"],
        "scaled_mape": base["scaled_mape"],
        "bad_ratio": base["bad_ratio"],
        "good_fraction": base["good_fraction"],
        "rmse": rmse(true_array, pred_array),
        "mae": mae(true_array, pred_array),
        "mape": mape(true_array, pred_array),
        "n": float(true_array.size),
    }
    return {key: float(value) for key, value in report.items()}
