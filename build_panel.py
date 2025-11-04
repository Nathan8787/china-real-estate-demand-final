#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""根據 panel_build.yaml 建立房市需求面板資料。
使用方式:
python build_panel.py --base-dir . --config config/panel_build.yaml
"""

import argparse
import json
import logging
import math
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from pypinyin import lazy_pinyin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ==========================
# 可快速調整的預設參數
# ==========================
DEFAULT_CONFIG: Dict = {
    # lags: sector 層級數值欄位要建立的滯後期數（月數）。
    "lags": [1, 2, 3, 6, 12],
    # diff_lags: 差分與增長率所依據的滯後期數。
    "diff_lags": [1, 3, 12],
    # pct_change_lags: 百分比變化（pct change）所採用的滯後期數。
    "pct_change_lags": [1, 3, 12],
    # rolling_windows: rolling 統計（mean/std/min/max）所使用的窗口大小（月數）。
    "rolling_windows": [3, 6],
    # ewm_alpha: 指數加權移動平均的平滑係數列表。
    "ewm_alpha": [0.3, 0.5],
    # poi_missing_threshold: POI 欄位允許缺失比例上限（超過即捨棄該欄）。
    "poi_missing_threshold": 0.5,
    # poi_pca_components: POI PCA 要保留的成分數量。
    "poi_pca_components": 5,
    # corr_threshold: 特徵間絕對相關係數超過此值就淘汰其中一個。
    "corr_threshold": 0.98,
    # importance_top_n: LightGBM 重要度排序若不足累積門檻時保底選取的特徵數量。
    "importance_top_n": 150,
    # cumulative_importance: 重要度累積比例門檻，達到後停止新增特徵。
    "cumulative_importance": 0.95,
    # lag_fill_zero: 是否將滯後/差分類欄位剩餘的 NaN 補為 0。
    "lag_fill_zero": False,
    # holiday_months: 各節慶旗標對應的月份列表。
    "holiday_months": {
        # is_spring_festival: 春節淡季月份。
        "is_spring_festival": [1, 2],
        # is_may_day: 五一假期月份。
        "is_may_day": [5],
        # is_golden_week: 國慶黃金周月份。
        "is_golden_week": [10],
        # is_double11: 雙十一檔期月份。
        "is_double11": [11],
    },
    # artifacts_dir: 輸出特徵字典、Scaler、選特徵清單等產物的目錄。
    "artifacts_dir": "artifacts",
    # data_dir: 各階段面板資料輸出的目錄。
    "data_dir": "data",
    # log_dir: log 檔案儲存目錄。
    "log_dir": "logs",
    # target_column: 主要預測目標欄位名稱。
    "target_column": "nh_amount_new_house_transactions",
    # train_end_month: 訓練資料截斷月份（含）用於定義 panel 訓練區間。
    "train_end_month": "2024-07-01",
    # poi_cluster_count: KMeans 分群時的群數，用於 POI 補值。
    "poi_cluster_count": 4,
}

STAGE_NAMES = [
    "載入設定與初始化",
    "讀取資料來源",
    "建立骨架並合併原始欄位",
    "進行特徵工程",
    "執行缺失值補齊",
    "執行特徵篩選",
    "輸出成果",
]

MANDATORY_COLUMNS = ["sector_id", "sector_label", "panel_month", "month_idx", "is_future_horizon"]


@dataclass
class PanelData:
    """封裝骨架與合併後資料。"""
    skeleton: pd.DataFrame
    merged: pd.DataFrame


# ==========================
# 共用工具函式
# ==========================

def load_config(base_dir: Path, config_path: Optional[str]) -> Dict:
    """Ū���~�� YAML �]�w�ɡA�ô� base_dir �ΨӰ��싣�ɮס���"""
    config = deepcopy(DEFAULT_CONFIG)
    if config_path:
        with open(config_path, "r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}
        config.update(user_cfg)
    config["base_dir"] = str(base_dir)
    for key in ["artifacts_dir", "data_dir", "log_dir"]:
        configured_path = Path(config[key])
        if not configured_path.is_absolute():
            configured_path = (base_dir / configured_path).resolve()
        config[key] = str(configured_path)
    return config


def ensure_directories(config: Dict) -> None:
    """依設定建立輸出目錄。"""
    for key in ["artifacts_dir", "data_dir", "log_dir"]:
        path = Path(config[key])
        path.mkdir(parents=True, exist_ok=True)


def setup_logging(log_dir: Path) -> None:
    """設定統一的 log 輸出格式。"""
    log_file = log_dir / "build_panel.log"
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def month_to_datetime(series: pd.Series) -> pd.Series:
    """將各種月份字串轉為當月第一天的 datetime。"""
    parsed = pd.to_datetime(series, format="%Y-%b", errors="coerce")
    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(series.loc[mask], format="%Y %b", errors="coerce")
    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(series.loc[mask], errors="coerce")
    return parsed.dt.to_period("M").dt.to_timestamp()


def slugify_keyword(text: str) -> str:
    """將關鍵字轉成蛇形命名（使用拼音避免中文無法使用）。"""
    if not isinstance(text, str):
        text = str(text)
    tokens = lazy_pinyin(text)
    candidate = "_".join(tokens).lower()
    candidate = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in candidate)
    candidate = "_".join(filter(None, candidate.split("_")))
    return candidate or "keyword"


def build_stage_progress() -> tqdm:
    """建立全流程進度條。"""
    return tqdm(total=len(STAGE_NAMES), desc="流程啟動", ncols=120, colour="green")

# ==========================
# 資料載入與骨架建立
# ==========================

def load_sources(base_dir: Path) -> Dict[str, pd.DataFrame]:
    """讀取所有必要的 CSV 檔案。"""
    logging.info("開始讀取所有資料來源…")
    file_map = {
        "new_house": base_dir / "train" / "new_house_transactions.csv",
        "new_house_nearby": base_dir / "train" / "new_house_transactions_nearby_sectors.csv",
        "pre_owned": base_dir / "train" / "pre_owned_house_transactions.csv",
        "pre_owned_nearby": base_dir / "train" / "pre_owned_house_transactions_nearby_sectors.csv",
        "land": base_dir / "train" / "land_transactions.csv",
        "land_nearby": base_dir / "train" / "land_transactions_nearby_sectors.csv",
        "poi": base_dir / "train" / "sector_POI.csv",
        "city_indexes": base_dir / "train" / "city_indexes.csv",
        "city_search": base_dir / "train" / "city_search_index.csv",
        "test": base_dir / "test.csv",
    }
    data = {}
    for name, path in tqdm(file_map.items(), desc="讀取檔案", ncols=120, colour="blue"):
        if not path.exists():
            raise FileNotFoundError(f"找不到必要資料檔案: {path}")
        data[name] = pd.read_csv(path)
    logging.info("資料來源讀取完成。")
    return data


def preprocess_city_search(df: pd.DataFrame) -> pd.DataFrame:
    """將城市搜尋指數轉為寬表格並產生蛇形欄名。"""
    temp = df.copy()
    temp["month"] = month_to_datetime(temp["month"])
    temp["keyword_slug"] = temp["keyword"].apply(slugify_keyword)
    temp["source_slug"] = temp["source"].apply(slugify_keyword)
    temp["col_name"] = "search_" + temp["keyword_slug"] + "_" + temp["source_slug"]
    pivot = temp.pivot_table(index="month", columns="col_name", values="search_volume", aggfunc="sum")
    pivot = pivot.sort_index()
    # 產生跨來源彙總欄位
    keyword_groups = {}
    for col in pivot.columns:
        base = col.rsplit("_", 1)[0]
        keyword_groups.setdefault(base, []).append(col)
    for base, cols in keyword_groups.items():
        pivot[f"{base}_all"] = pivot[cols].sum(axis=1)
    pivot = pivot.reset_index().rename(columns={"month": "panel_month"})
    return pivot


def preprocess_city_indexes(df: pd.DataFrame) -> pd.DataFrame:
    """將年度城巿指標擴展至月度並計算 YoY。"""
    temp = df.copy()
    temp = temp.sort_values("city_indicator_data_year")
    temp = temp.drop_duplicates(subset=["city_indicator_data_year"], keep="last")
    years = temp["city_indicator_data_year"].astype(int).tolist()
    if not years:
        raise ValueError("city_indexes.csv 無資料")
    min_year, max_year = min(years), max(years)
    # 擴展至 2024 年，若缺資料則使用最後一年數值
    records = []
    last_row = None
    for year in range(min_year, 2025):
        if year in temp["city_indicator_data_year"].values:
            last_row = temp[temp["city_indicator_data_year"] == year].iloc[0]
        if last_row is None:
            continue
        for month in range(1, 13):
            row = last_row.to_dict()
            row["panel_month"] = pd.Timestamp(year=year, month=month, day=1)
            row["macro_is_extrapolated"] = int(year >= 2023)
            records.append(row)
    macro_df = pd.DataFrame(records)
    rename_map = {
        col: col if col in {"panel_month", "macro_is_extrapolated", "city_indicator_data_year"} else f"macro_{col}"
        for col in macro_df.columns
    }
    macro_df = macro_df.rename(columns=rename_map)
    macro_df = macro_df.drop(columns=["city_indicator_data_year"], errors="ignore")
    macro_df = macro_df.sort_values("panel_month").reset_index(drop=True)

    macro_cols = [c for c in macro_df.columns if c.startswith("macro_") and not c.endswith("_interp") and c not in {"macro_is_extrapolated"}]
    for col in macro_cols:
        interp_col = f"{col}_interp"
        macro_df[interp_col] = macro_df[col].interpolate(method="linear")
        macro_df[f"{col}_yoy"] = macro_df[interp_col] / macro_df[interp_col].shift(12) - 1
    return macro_df


def build_skeleton(data: Dict[str, pd.DataFrame], config: Dict) -> PanelData:
    """建立 sector × month 的完整骨架。"""
    logging.info("建立面板骨架…")
    new_house = data["new_house"].copy()
    new_house["month"] = month_to_datetime(new_house["month"])

    sector_set = set(new_house["sector"].unique())
    for key in ["new_house_nearby", "pre_owned", "pre_owned_nearby", "land", "land_nearby"]:
        temp = data[key].copy()
        temp["month"] = month_to_datetime(temp["month"])
        sector_set.update(temp["sector"].unique())
    sector_set.update(data["poi"]["sector"].unique())

    test = data["test"].copy()
    test[["date_part", "sector_num"]] = test["id"].str.split("_sector ", expand=True)
    test["sector"] = test["sector_num"].astype(int).apply(lambda x: f"sector {x}")
    sector_set.update(test["sector"].unique())

    sector_list = sorted(sector_set, key=lambda s: int(s.split()[1]))
    sector_id_map = {label: idx for idx, label in enumerate(sector_list, start=1)}

    start_month = new_house["month"].min()
    end_month = pd.Timestamp(config["train_end_month"]) + pd.DateOffset(months=12)
    month_range = pd.date_range(start=start_month, end=end_month, freq="MS")

    skeleton = pd.MultiIndex.from_product([sector_list, month_range], names=["sector_label", "panel_month"]).to_frame(index=False)
    skeleton["sector_id"] = skeleton["sector_label"].map(sector_id_map)
    skeleton["month_idx"] = ((skeleton["panel_month"].dt.year - start_month.year) * 12
                              + (skeleton["panel_month"].dt.month - start_month.month))
    skeleton["is_future_horizon"] = (skeleton["panel_month"] >= pd.Timestamp("2024-08-01")).astype(int)

    logging.info("骨架建立完成，列數: %d", len(skeleton))
    return PanelData(skeleton=skeleton, merged=skeleton.copy())

def merge_sources(panel: PanelData, data: Dict[str, pd.DataFrame], artifacts_dir: Path) -> pd.DataFrame:
    """依照 spec 合併所有資料表與衍生旗標。"""
    df = panel.merged.copy()

    def rename_with_prefix(source_df: pd.DataFrame, prefix: str, exclude: Sequence[str]) -> pd.DataFrame:
        return source_df.rename(columns=lambda c: c if c in exclude else f"{prefix}{c}")

    def add_has_flag(target: pd.DataFrame, columns: List[str], flag_name: str) -> pd.DataFrame:
        if not columns:
            target[flag_name] = 0
            return target
        mask = target[columns].notna().all(axis=1)
        target[flag_name] = mask.astype(int)
        return target

    merge_plan = [
        ("new_house", "nh_", ["month", "sector"], "has_new_house"),
        ("new_house_nearby", "nhnbr_", ["month", "sector"], "has_new_house_nearby"),
        ("pre_owned", "po_", ["month", "sector"], "has_pre_owned"),
        ("pre_owned_nearby", "ponbr_", ["month", "sector"], "has_pre_owned_nearby"),
        ("land", "land_", ["month", "sector"], "has_land"),
        ("land_nearby", "landnbr_", ["month", "sector"], "has_land_nearby"),
    ]

    for name, prefix, exclude_cols, flag in merge_plan:
        src = data[name].copy()
        src["month"] = month_to_datetime(src["month"])
        src = rename_with_prefix(src, prefix, exclude_cols)
        df = df.merge(src, left_on=["sector_label", "panel_month"], right_on=["sector", "month"], how="left")
        df = df.drop(columns=[col for col in ["sector", "month"] if col in df.columns])
        numeric_cols = [c for c in df.columns if c.startswith(prefix)]
        df = add_has_flag(df, numeric_cols, flag)

    # POI 靜態資料
    poi = data["poi"].copy()
    poi = rename_with_prefix(poi, "poi_", ["sector"])
    df = df.merge(poi, left_on="sector_label", right_on="sector", how="left")
    df = df.drop(columns=["sector"], errors="ignore")
    df = add_has_flag(df, [c for c in df.columns if c.startswith("poi_")], "has_poi")

    # 城市搜尋指數 (城市層級)
    city_search = preprocess_city_search(data["city_search"])
    df = df.merge(city_search, on="panel_month", how="left")
    df = add_has_flag(df, [c for c in df.columns if c.startswith("search_")], "has_search")

    # 城市宏觀指標
    macro = preprocess_city_indexes(data["city_indexes"])
    df = df.merge(macro, on="panel_month", how="left")
    df = add_has_flag(df, [c for c in df.columns if c.startswith("macro_")], "has_macro")

    # 保存 sector index 對照
    sector_index_path = artifacts_dir / "sector_index.csv"
    panel.skeleton[["sector_label", "sector_id"]].drop_duplicates().sort_values("sector_id").to_csv(sector_index_path, index=False)

    return df

# ==========================
# 特徵工程相關函式
# ==========================

def create_missing_flags(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """針對指定欄位建立 missing_ 前綴旗標，方便模型感知缺失。"""
    for col in columns:
        flag = f"missing_{col}"
        if flag not in df:
            df[flag] = df[col].isna().astype(int)


def safe_divide(numerator: Optional[pd.Series], denominator: Optional[pd.Series]) -> pd.Series:
    """避免除以零並在任一輸入缺失時輸出 NaN 序列。"""
    if numerator is None and denominator is None:
        return pd.Series(dtype=float)
    if numerator is None:
        return pd.Series(np.nan, index=denominator.index)
    if denominator is None:
        return pd.Series(np.nan, index=numerator.index)
    denom = denominator.replace({0: np.nan})
    result = numerator / denom
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def generate_sector_lags(df: pd.DataFrame, cols: Sequence[str], lags: Sequence[int]) -> None:
    """依 sector 計算滯後欄位。"""
    df.sort_values(["sector_id", "panel_month"], inplace=True)
    group = df.groupby("sector_id", sort=False)
    for col in cols:
        for lag in lags:
            lag_col = f"{col}_lag{lag}"
            if lag_col in df:
                continue
            df[lag_col] = group[col].shift(lag)


def generate_city_lags(df: pd.DataFrame, cols: Sequence[str], lags: Sequence[int]) -> None:
    """針對城市層級欄位 (search/macro) 建立滯後，避免重複計算。"""
    base = df[["panel_month"] + list(cols)].drop_duplicates("panel_month").sort_values("panel_month").set_index("panel_month")
    for col in cols:
        for lag in lags:
            lag_series = base[col].shift(lag)
            lag_col = f"{col}_lag{lag}"
            if lag_col in df:
                continue
            df = df.merge(lag_series.rename(lag_col), left_on="panel_month", right_index=True, how="left")
    return df


def generate_differences(df: pd.DataFrame, cols: Sequence[str], lags: Sequence[int]) -> None:
    """依照滯後結果產生差分與成長率。"""
    for col in cols:
        for lag in lags:
            lag_col = f"{col}_lag{lag}"
            if lag_col not in df:
                continue
            diff_col = f"{col}_diff{lag}"
            pct_col = f"{col}_pctchg{lag}"
            df[diff_col] = df[col] - df[lag_col]
            denom = df[lag_col].replace({0: np.nan})
            df[pct_col] = (df[col] - df[lag_col]) / denom


def generate_rolling_stats(df: pd.DataFrame, cols: Sequence[str], windows: Sequence[int]) -> None:
    """計算 rolling mean/std/min/max。"""
    df.sort_values(["sector_id", "panel_month"], inplace=True)
    group = df.groupby("sector_id", sort=False)
    for col in cols:
        for window in windows:
            df[f"{col}_roll{window}_mean"] = group[col].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f"{col}_roll{window}_std"] = group[col].rolling(window=window, min_periods=1).std().reset_index(level=0, drop=True)
            df[f"{col}_roll{window}_min"] = group[col].rolling(window=window, min_periods=1).min().reset_index(level=0, drop=True)
            df[f"{col}_roll{window}_max"] = group[col].rolling(window=window, min_periods=1).max().reset_index(level=0, drop=True)


def generate_ewm(df: pd.DataFrame, cols: Sequence[str], alphas: Sequence[float]) -> None:
    """計算指數加權移動平均。"""
    df.sort_values(["sector_id", "panel_month"], inplace=True)
    for col in cols:
        grouped = df.groupby("sector_id", sort=False)[col]
        for alpha in alphas:
            df[f"{col}_ewm_alpha{alpha}"] = grouped.transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())


def add_supply_demand_features(df: pd.DataFrame) -> None:
    """加入供需相關衍生欄位。"""
    df["nh_po_amt_ratio"] = safe_divide(df.get("nh_amount_new_house_transactions"), df.get("po_amount_pre_owned_house_transactions"))
    df["nh_inventory_turnover"] = safe_divide(df.get("nh_num_new_house_available_for_sale"), df.get("nh_num_new_house_transactions"))
    df["nh_neighbor_amt_ratio"] = safe_divide(df.get("nh_amount_new_house_transactions"), df.get("nhnbr_amount_new_house_transactions"))
    df["nh_land_lead_ratio_lag3"] = safe_divide(df.get("land_transaction_amount_lag3"), df.get("nh_amount_new_house_transactions"))


def add_event_flags(df: pd.DataFrame) -> None:
    """建立事件旗標，例如土地成交與搜尋熱潮。"""
    land_col = "land_transaction_amount"
    if land_col in df:
        df["land_transacted_flag"] = (df[land_col] > 0).astype(int)
        for lag in [1, 3]:
            df[f"land_transacted_flag_lag{lag}"] = df.groupby("sector_id")["land_transacted_flag"].shift(lag)

    surge_keywords = ["search_maifang", "search_shoufu", "search_xiangou"]
    for base in surge_keywords:
        cols = [c for c in df.columns if c.startswith(base)]
        if not cols:
            continue
        monthly = df[["panel_month"] + cols].drop_duplicates("panel_month").sort_values("panel_month").set_index("panel_month")
        rolling_q3 = monthly[cols].rolling(window=12, min_periods=3).quantile(0.75)
        for col in cols:
            surge_col = f"{col}_surge"
            df = df.merge(rolling_q3[[col]].rename(columns={col: surge_col}), left_on="panel_month", right_index=True, how="left")
            df[surge_col] = ((df[col] > df[surge_col]) & df[col].notna()).astype(int)


def add_seasonality_features(df: pd.DataFrame, config: Dict) -> None:
    """加入月份、季節與節慶旗標。"""
    df["month"] = df["panel_month"].dt.month
    df["quarter"] = df["panel_month"].dt.quarter
    df["month_sin"] = np.sin(2 * math.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * math.pi * df["month"] / 12)
    for flag, months in config.get("holiday_months", {}).items():
        df[flag] = df["month"].isin(months).astype(int)


def prepare_poi_features(df: pd.DataFrame, config: Dict, artifacts_dir: Path) -> pd.DataFrame:
    """靜態 POI 特徵的整理、補值、標準化與 PCA。"""
    poi_cols = [c for c in df.columns if c.startswith("poi_")]
    if not poi_cols:
        logging.warning("未找到任何 POI 欄位，略過 POI 處理。")
        return df

    missing_ratio = df[poi_cols].isna().mean()
    keep_cols = [c for c in poi_cols if missing_ratio[c] <= config["poi_missing_threshold"]]
    drop_cols = [c for c in poi_cols if c not in keep_cols]
    if drop_cols:
        logging.info("以下 POI 欄位缺失過多而被排除: %s", drop_cols)
        df = df.drop(columns=drop_cols)
        poi_cols = keep_cols

    if not poi_cols:
        logging.warning("POI 欄位皆被濾除，略過後續 PCA。")
        return df

    # 計算缺失旗標
    create_missing_flags(df, poi_cols)

    poi_values = df[poi_cols].copy()
    global_medians = poi_values.median()
    filled_for_cluster = poi_values.fillna(global_medians)

    # 進行 KMeans 分群以協助補值
    k = min(config.get("poi_cluster_count", 4), len(filled_for_cluster))
    k = max(k, 1)
    scaler_for_cluster = StandardScaler()
    scaled = scaler_for_cluster.fit_transform(filled_for_cluster)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled)
    df["poi_cluster_id"] = clusters

    poi_imputed = poi_values.copy()
    for col in poi_cols:
        cluster_medians = poi_values.groupby(clusters)[col].median()
        for cluster_id, median_value in cluster_medians.items():
            mask = (clusters == cluster_id) & poi_imputed[col].isna()
            poi_imputed.loc[mask, col] = median_value
        poi_imputed[col].fillna(global_medians[col], inplace=True)

    imputed_flag = (poi_values.isna() & poi_imputed.notna()).any(axis=1).astype(int)
    df["imputed_poi_flag"] = imputed_flag
    df[poi_cols] = poi_imputed

    # 建立額外比率指標
    if "poi_number_of_shops" in df and "poi_resident_population" in df:
        df["poi_shop_per_capita"] = safe_divide(df["poi_number_of_shops"], df["poi_resident_population"])
    if "poi_catering" in df and "poi_number_of_shops" in df:
        df["poi_catering_share"] = safe_divide(df["poi_catering"], df["poi_number_of_shops"])

    scaler = StandardScaler()
    poi_scaled = scaler.fit_transform(df[poi_cols])
    joblib.dump(scaler, artifacts_dir / "poi_scaler.joblib")

    pca_components = min(config.get("poi_pca_components", 5), len(poi_cols))
    pca = PCA(n_components=pca_components, random_state=42)
    poi_pca = pca.fit_transform(poi_scaled)
    for idx in range(poi_pca.shape[1]):
        df[f"poi_pca_component_{idx + 1}"] = poi_pca[:, idx]

    meta = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "components": pca_components,
        "used_columns": poi_cols,
    }
    with open(artifacts_dir / "poi_pca_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    # PCA 完成後移除原始高維欄位
    df = df.drop(columns=poi_cols)
    return df


def feature_engineering(df: pd.DataFrame, config: Dict, artifacts_dir: Path) -> pd.DataFrame:
    """依 spec.md 產生所有衍生特徵。"""
    logging.info("開始進行特徵工程…")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    raw_cols = [c for c in numeric_cols if not any(tag in c for tag in ["_lag", "_diff", "_pctchg", "_roll", "_ewm", "_yoy"])]
    create_missing_flags(df, raw_cols)

    sector_prefixes = ("nh_", "nhnbr_", "po_", "ponbr_", "land_", "landnbr_")
    sector_cols = [c for c in df.columns if c.startswith(sector_prefixes)]
    sector_numeric = [c for c in sector_cols if df[c].dtype != "O"]

    generate_sector_lags(df, sector_numeric, config["lags"])
    generate_differences(df, sector_numeric, config["diff_lags"])
    generate_rolling_stats(df, sector_numeric, config["rolling_windows"])
    generate_ewm(df, sector_numeric, config["ewm_alpha"])

    city_cols = [c for c in df.columns if c.startswith("search_") or c.startswith("macro_")]
    df = generate_city_lags(df, city_cols, config["lags"])
    generate_differences(df, city_cols, config["diff_lags"])

    add_supply_demand_features(df)
    add_event_flags(df)
    add_seasonality_features(df, config)

    df = prepare_poi_features(df, config, artifacts_dir)

    logging.info("特徵工程完成，欄位總數: %d", len(df.columns))
    return df




# ==========================
# 缺失值補齊
# ==========================

def apply_imputation(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """依規格進行缺失值補齊。"""
    logging.info("開始缺失值處理…")
    df = df.sort_values(["sector_id", "panel_month"]).reset_index(drop=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    skip_patterns = ("_lag", "_diff", "_pctchg", "_roll", "_ewm")
    skip_cols = set()
    for col in numeric_cols:
        if any(pattern in col for pattern in skip_patterns):
            skip_cols.add(col)
    skip_cols.update(MANDATORY_COLUMNS)
    skip_cols.add(config["target_column"])

    for col in tqdm(numeric_cols, desc="補值進度", ncols=120, colour="cyan"):
        if col in skip_cols:
            continue
        original = df[col].copy()
        if col.startswith("search_") or col.startswith("macro_"):
            df[col] = df.groupby("panel_month")[col].transform("first")
            df[col] = df[col].fillna(df[col].median())
            changed = df[col].ne(original) & original.isna()
            if changed.any():
                df[f"imputed_{col}"] = changed.astype(int)
            continue

        group = df.groupby("sector_id", sort=False)[col]
        df[col] = group.ffill()
        df[col] = group.bfill()
        if df[col].isna().any():
            df[col] = df.groupby("panel_month")[col].transform(lambda s: s.fillna(s.median()))
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
        changed = df[col].ne(original)
        if changed.any():
            df[f"imputed_{col}"] = changed.astype(int)

    # 若設定要求，將滯後欄位 NaN 改為 0
    if config.get("lag_fill_zero", False):
        lag_cols = [c for c in df.columns if any(tag in c for tag in ["_lag", "_diff", "_pctchg"])]
        for col in lag_cols:
            df[col] = df[col].fillna(0)

    logging.info("缺失值處理完成。")
    return df

# ==========================
# 特徵篩選
# ==========================

def remove_low_variance(features: pd.DataFrame, threshold: float = 1e-6) -> Tuple[pd.DataFrame, List[str]]:
    """移除變異過低的欄位。"""
    var = features.var()
    drop_cols = var[var < threshold].index.tolist()
    features = features.drop(columns=drop_cols, errors="ignore")
    return features, drop_cols


def remove_high_correlation(features: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """移除高度共線的欄位。"""
    corr = features.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = []
    drop_pairs = []
    for col in upper.columns:
        high = upper.index[upper[col] > threshold].tolist()
        for high_col in high:
            if high_col not in to_drop:
                to_drop.append(high_col)
                drop_pairs.append((col, high_col))
    features = features.drop(columns=to_drop, errors="ignore")
    return features, drop_pairs


def time_based_folds(df: pd.DataFrame) -> List[Tuple[pd.Index, pd.Index]]:
    """依 spec 定義時間序列交叉驗證折。"""
    fold_definitions = [
        ("2022-06-01", "2022-07-01", "2022-12-01"),
        ("2022-12-01", "2023-01-01", "2023-06-01"),
        ("2023-06-01", "2023-07-01", "2023-12-01"),
        ("2023-12-01", "2024-01-01", "2024-06-01"),
    ]
    folds = []
    for train_end, valid_start, valid_end in fold_definitions:
        train_mask = df["panel_month"] <= pd.Timestamp(train_end)
        valid_mask = (df["panel_month"] >= pd.Timestamp(valid_start)) & (df["panel_month"] <= pd.Timestamp(valid_end))
        folds.append((df.loc[train_mask].index, df.loc[valid_mask].index))
    return folds


def compute_feature_importance(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> pd.Series:
    """使用 LightGBM 在多個時間折上計算特徵重要度。"""
    folds = time_based_folds(df)
    importance_accumulator = pd.Series(0.0, index=feature_cols)
    for idx, (train_idx, valid_idx) in enumerate(folds, start=1):
        X_train = df.loc[train_idx, feature_cols]
        y_train = df.loc[train_idx, target_col]
        X_valid = df.loc[valid_idx, feature_cols]
        y_valid = df.loc[valid_idx, target_col]
        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="rmse")
        importance_accumulator += pd.Series(model.booster_.feature_importance(importance_type="gain"), index=feature_cols)
    importance_accumulator /= len(folds)
    return importance_accumulator


def feature_selection(df: pd.DataFrame, config: Dict, artifacts_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    """執行多階段特徵篩選並輸出紀錄。"""
    logging.info("開始特徵篩選…")
    target_col = config["target_column"]
    train_cutoff = pd.Timestamp(config["train_end_month"])
    train_df = df[df["panel_month"] <= train_cutoff].copy()

    feature_cols = [c for c in df.columns if c not in MANDATORY_COLUMNS + [target_col]]
    X = train_df[feature_cols].select_dtypes(include=[np.number])

    X, low_var_drops = remove_low_variance(X)
    logging.info("低變異欄位移除數量: %d", len(low_var_drops))

    X, corr_pairs = remove_high_correlation(X, config["corr_threshold"])
    logging.info("高相關欄位移除數量: %d", len(corr_pairs))
    corr_log = pd.DataFrame(corr_pairs, columns=["kept", "dropped"])
    corr_log.to_csv(artifacts_dir / "correlation_drop.csv", index=False)

    feature_cols = X.columns.tolist()
    importance = compute_feature_importance(train_df, feature_cols, target_col).sort_values(ascending=False)
    importance.to_csv(artifacts_dir / "feature_importance.csv", header=["importance"])

    cumulative = importance.cumsum() / importance.sum()
    selected = importance[cumulative <= config["cumulative_importance"]].index.tolist()
    if len(selected) < config["importance_top_n"]:
        selected = importance.head(config["importance_top_n"]).index.tolist()

    final_features = sorted(set(selected) | set(MANDATORY_COLUMNS) | {target_col})
    with open(artifacts_dir / "selected_features.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(final_features))

    logging.info("特徵篩選後欄位數: %d", len(final_features))
    return df[final_features].copy(), final_features

# ==========================
# 主流程
# ==========================

def save_dataframe(df: pd.DataFrame, path: Path, fmt: str = "pkl") -> None:
    """依指定格式儲存資料。"""
    if fmt == "pkl":
        df.to_pickle(path)
    elif fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"不支援的格式: {fmt}")


def validate_panel(df: pd.DataFrame, panel: PanelData) -> None:
    """依據 spec 清單檢核。"""
    expected_rows = len(panel.skeleton)
    if len(df) != expected_rows:
        raise ValueError(f"面板列數不符，預期 {expected_rows}，實際 {len(df)}")
    if df.duplicated(subset=["sector_id", "panel_month"]).any():
        raise ValueError("(sector_id, panel_month) 組合出現重複。")


def run_pipeline(base_dir: Path, config_path: Optional[str]) -> Dict[str, Path]:
    """依規格執行 panel pipeline 並回傳輸出路徑對應。"""
    progress = build_stage_progress()

    progress.set_description(STAGE_NAMES[0])
    config = load_config(base_dir, config_path)
    ensure_directories(config)
    setup_logging(Path(config["log_dir"]))
    config_snapshot_path = Path(config["log_dir"]) / "build_panel_config.yaml"
    with open(config_snapshot_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, allow_unicode=True, sort_keys=True)
    logging.info("已載入並合併設定，寫入配置檔案: %s", config_snapshot_path)
    artifacts_dir = Path(config["artifacts_dir"])
    data_dir = Path(config["data_dir"])
    progress.update(1)

    progress.set_description(STAGE_NAMES[1])
    data = load_sources(base_dir)
    progress.update(1)

    progress.set_description(STAGE_NAMES[2])
    panel = build_skeleton(data, config)
    merged = merge_sources(panel, data, artifacts_dir)
    validate_panel(merged, panel)
    raw_panel_path = data_dir / "panel_with_raw.pkl"
    save_dataframe(merged, raw_panel_path)
    progress.update(1)

    progress.set_description(STAGE_NAMES[3])
    engineered = feature_engineering(merged.copy(), config, artifacts_dir)
    features_path = data_dir / "panel_with_features.pkl"
    save_dataframe(engineered, features_path)
    progress.update(1)

    progress.set_description(STAGE_NAMES[4])
    imputed = apply_imputation(engineered.copy(), config)
    imputed_path = data_dir / "panel_imputed.pkl"
    save_dataframe(imputed, imputed_path)
    progress.update(1)

    progress.set_description(STAGE_NAMES[5])
    final_df, selected_features = feature_selection(imputed.copy(), config, artifacts_dir)
    final_panel_path = data_dir / "panel_selected.parquet"
    save_dataframe(final_df, final_panel_path, fmt="parquet")
    progress.update(1)

    progress.set_description(STAGE_NAMES[6])
    progress.update(1)
    outputs = {
        "panel_with_raw": raw_panel_path,
        "panel_with_features": features_path,
        "panel_imputed": imputed_path,
        "panel_selected": final_panel_path,
        "selected_features": artifacts_dir / "selected_features.txt",
        "correlation_drop": artifacts_dir / "correlation_drop.csv",
        "feature_importance": artifacts_dir / "feature_importance.csv",
        "sector_index": artifacts_dir / "sector_index.csv",
        "poi_scaler": artifacts_dir / "poi_scaler.joblib",
        "poi_pca_meta": artifacts_dir / "poi_pca_meta.json",
        "config_snapshot": config_snapshot_path,
    }
    progress.close()
    logging.info("�y�{�����I�̲����� %d", len(selected_features))
    logging.info("�ػݤ覡�̭��X: %s", json.dumps({k: str(v) for k, v in outputs.items()}, ensure_ascii=False))
    return outputs



def build_panel(base_dir: str | Path = ".", config_path: Optional[str] = None) -> Dict[str, Path]:
    """�ѩϥΪ̱q�ʿ�J base_dir �M config �ɮסA�I�s panel pipeline"""
    resolved_base = Path(base_dir).resolve()
    outputs = run_pipeline(resolved_base, config_path)
    return outputs



def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="建置房市需求面板資料")
    parser.add_argument("--config", type=str, default=None, help="自訂 YAML 設定檔路徑")
    parser.add_argument("--base-dir", type=str, default=".", help="資料根目錄，預設為目前路徑")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    build_panel(args.base_dir, args.config)


if __name__ == "__main__":
    main()





