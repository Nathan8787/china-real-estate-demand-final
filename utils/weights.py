"""utils.weights �o�� sample weight ���ѪR."""

from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None  # type: ignore

WEIGHTS_VERSION = "1.0"


def validate_strategy(strategy_config: dict) -> None:
    """�ˬd�]�w�O�_���Ӥ� spec �n�D"""
    required_keys = {"enabled", "strategy", "half_life_months", "linear_min_ratio"}
    missing = required_keys - set(strategy_config.keys())
    if missing:
        raise ValueError(f"strategy_config 缺少必要鍵: {sorted(missing)}")
    if strategy_config["strategy"] not in {"none", "exponential", "linear"}:
        raise ValueError("strategy 必須為 none/exponential/linear 其中之一")
    half_life = strategy_config.get("half_life_months", 0)
    if half_life is not None and half_life <= 0:
        raise ValueError("half_life_months 必須大於 0")
    ratio = float(strategy_config.get("linear_min_ratio", 0))
    if ratio <= 0 or ratio > 1:
        raise ValueError("linear_min_ratio 必須介於 (0, 1]")


def _to_datetime_series(months: "Iterable") -> "np.ndarray":
    """�ғ���J���rxnp datetime64"""
    if pd is not None and isinstance(months, (pd.Series, pd.Index, pd.DatetimeIndex)):
        return months.to_numpy(dtype="datetime64[M]", copy=False)
    if isinstance(months, np.ndarray) and np.issubdtype(months.dtype, np.datetime64):
        return months.astype("datetime64[M]")
    if isinstance(months, Iterable):
        if pd is not None:
            return pd.to_datetime(list(months)).to_numpy(dtype="datetime64[M]")
        return np.array(list(months), dtype="datetime64[M]")
    raise TypeError("month_series 需為可轉換成 datetime64[M] 之型別")


def age_in_months(reference_month: "np.datetime64 | str", months: "Iterable") -> np.ndarray:
    """�p�� reference_month �P�C�Ӧb��᪺���᪺�Ȭq"""
    ref = np.datetime64(reference_month, "M")
    month_array = _to_datetime_series(months)
    ref_year = ref.astype("datetime64[Y]").astype(int)
    ref_month = ref.astype("datetime64[M]").astype(int) - ref_year * 12
    years = month_array.astype("datetime64[Y]").astype(int)
    months_in_year = month_array.astype("datetime64[M]").astype(int) - years * 12
    age = (ref_year - years) * 12 + (ref_month - months_in_year)
    age = age.astype(float)
    age = np.clip(age, 0.0, None)
    return age


def compute_sample_weights(month_series: "Iterable", strategy_config: dict, reference_month: "np.datetime64 | str") -> np.ndarray:
    """�ھڱ]�w�ˬd�ƭȡA�T����@�ӮױƧǪ��ũq"""
    validate_strategy(strategy_config)
    if not strategy_config.get("enabled", True) or strategy_config.get("strategy") == "none":
        months = _to_datetime_series(month_series)
        return np.ones(months.shape[0], dtype=np.float64)

    ages = age_in_months(reference_month, month_series)
    strategy = strategy_config.get("strategy")

    if strategy == "exponential":
        half_life = float(strategy_config["half_life_months"])
        weights = 0.5 ** (ages / half_life)
    elif strategy == "linear":
        max_age = np.max(ages)
        if max_age <= 0:
            weights = np.ones_like(ages)
        else:
            min_ratio = float(strategy_config["linear_min_ratio"])
            weights = min_ratio + (1 - min_ratio) * (1 - ages / max_age)
            weights = np.clip(weights, min_ratio, None)
    else:  # none already handled; keep for completeness
        weights = np.ones_like(ages)

    mean_weight = np.mean(weights)
    if mean_weight == 0:
        raise ValueError("權重平均值為 0，請確認設定")
    normalized = weights / mean_weight
    return normalized.astype(np.float64)
