<div align="center">

# China Real Estate Demand Prediction - Final Submission

</div>

This document describes the final submission package, methodology, and execution workflow. No external public data were used; every script, feature engineering routine, and model was developed and trained by our team in compliance with Kaggle rules.

---

## Table of Contents

- [China Real Estate Demand Prediction - Final Submission](#china-real-estate-demand-prediction---final-submission)
  - [Table of Contents](#table-of-contents)
  - [Environment and Runtime Summary](#environment-and-runtime-summary)
  - [Methodology Overview](#methodology-overview)
  - [Repository Layout](#repository-layout)
  - [Data Acquisition](#data-acquisition)
  - [Panel Construction](#panel-construction)
    - [Prerequisites](#prerequisites)
    - [1. Command](#1-command)
    - [2. Parameters (`panel_build.yaml`)](#2-parameters-panel_buildyaml)
    - [3. Process Summary](#3-process-summary)
    - [4. Final Feature List](#4-final-feature-list)
  - [Model Training](#model-training)
    - [1. Command](#1-command-1)
    - [2. Configuration Highlights (`config/train_model.yaml`)](#2-configuration-highlights-configtrain_modelyaml)
    - [3. Workflow](#3-workflow)
    - [4. Models](#4-models)
  - [Inference and Submission Generation](#inference-and-submission-generation)
    - [1. Command](#1-command-2)
    - [2. Workflow Summary](#2-workflow-summary)
  - [Model and Artifact Inventory](#model-and-artifact-inventory)
  - [Reproducibility Notes](#reproducibility-notes)
  - [Compliance and Originality Statement](#compliance-and-originality-statement)

---

## Environment and Runtime Summary

| Item | Specification |
|------|---------------|
| Operating system | Windows 11 (build 10.0.26100) 64-bit |
| CPU | Intel Core i7-13620H, 10 cores / 16 threads |
| Memory | 64 GB |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (4 GB VRAM); Intel UHD Graphics available |
| Python | 3.13.2 (`py -3 --version`) |
| Dependencies | Install via `pip install -r requirements.txt`; GPU libraries are optional if training with CPU only |

| Pipeline stage | Approximate runtime (above hardware) |
|----------------|---------------------------------------|
| Panel construction (`build_panel.py`) | ~15 minutes |
| Model training (`train_model.py`) | ~2.5 hours |
| Inference and submission (`predict_future.py`) | ~2 to 3 minutes |

---

## Methodology Overview

1. **Panel-centric data processing**  
   - Combine housing transactions, land indicators, search trends, POI statistics, and neighbouring sector metrics into a monthly panel indexed by `(sector_id, panel_month)`.  
   - Engineer lag, difference, percentage-change, moving-window, and holiday features; derive POI PCA components and KMeans clusters to capture spatial structure.

2. **Target encoding and sample weighting**  
   - Apply log-transform and non-negative clipping to stabilise the heavy-tailed target distribution.  
   - Use target encoding on `sector_id` and exponential time decay (half-life 12 months) to emphasise recent observations.

3. **Multi-model ensemble**  
   - Train LightGBM, XGBoost, CatBoost, and Random Forest models with Optuna-based hyperparameter search.  
   - Solve a constrained optimisation problem (SciPy) to obtain ensemble weights; the submission uses manually tuned weights `[0.3025, 0.1168, 0.5807, 0.0]` (LightGBM, XGBoost, CatBoost, Random Forest) which performed best during experimentation.

4. **Conservative inference**  
   - After ensemble predictions, overwrite any sector that recorded zero transactions within the previous six months, ensuring forecasts remain conservative in stagnant markets.

---

## Repository Layout

| Path | Description |
|------|-------------|
| `build_panel.py` | Panel construction script |
| `train_model.py` | Hyperparameter search, model fitting, ensemble weighting |
| `predict_future.py` | Inference, ensemble aggregation, zero-sector override, CSV export |
| `download_kaggle_data.py` | Kaggle dataset download helper |
| `config/panel_build.yaml` | Panel construction parameters |
| `config/train_model.yaml` | Training and inference configuration |
| `artifacts/` | Feature list, POI PCA metadata, feature importance, correlation drops |
| `data/` | Panel artefacts (`panel_with_raw.pkl`, `panel_with_features.pkl`, `panel_imputed.pkl`, `panel_selected.parquet`) |
| `models/` | Final checkpoints, best-parameter JSONs, scaler, target-encoding map, ensemble weights |
| `utils/` | Shared modules (`metrics.py`, `reporting.py`, `weights.py`) |
| `requirements.txt` | Dependency list |
| `submission_te_final.csv` | Final Kaggle submission file |

---

## Data Acquisition

Raw Kaggle CSV files are not shipped with this package. Download them with:

```bash
py -3 download_kaggle_data.py --output-dir train --force
```

Notes:

- Prepare `kaggle.json` or set `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables in advance.  
- The script flattens the extracted archive and copies `test.csv` / `sample_submission.csv` into the repository root (disable with `--no-promote` if necessary).  
- If raw files already exist locally, place them in `train/` with matching filenames.

---

## Panel Construction

### Prerequisites

**Repositories do not carry the raw Kaggle CSVs nor the intermediate panel files (`data/panel_with_raw.pkl`, `panel_with_features.pkl`, `panel_imputed.pkl`) because they exceed GitHub's 100 MB limit.
Before training or inference, download the raw data (see [Data Acquisition](#data-acquisition)) and rebuild the panel using the command below. The process is deterministic; no random seeds are consumed in `build_panel.py`.**

### 1. Command

```bash
py -3 build_panel.py --base-dir . --config config/panel_build.yaml
```

### 2. Parameters (`panel_build.yaml`)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `lags` | `[1, 2, 3, 6, 12]` | Create multi-step lag features |
| `diff_lags` | `[1, 3, 12]` | Month, quarter, and year differences |
| `pct_change_lags` | `[1, 3, 12]` | Month, quarter, and year percentage change |
| `rolling_windows` | `[3, 6]` | Rolling statistics windows |
| `ewm_alpha` | `[0.3, 0.5]` | Exponential moving average smoothing |
| `lag_fill_zero` | `false` | Do not fill lagged NaNs with zero |
| `poi_missing_threshold` | `0.5` | Drop POI columns missing ≥ 50% values |
| `poi_pca_components` | `5` | Number of POI PCA components |
| `poi_cluster_count` | `4` | KMeans clusters for POI composition |
| `corr_threshold` | `0.9` | Pearson correlation threshold for dropping features |
| `importance_top_n` | `100` | Retain top 100 features by LightGBM importance |
| `cumulative_importance` | `0.95` | Cumulative importance must reach 95% |
| `holiday_months` | `[1, 2]`, `[5]`, `[10]`, `[11]` | Holiday dummies (Spring Festival, Labour Day, Golden Week, Double 11) |
| `target_column` | `nh_amount_new_house_transactions` | Target variable |
| `train_end_month` | `"2024-07-01"` | Aligns with training horizon |
| `artifacts_dir` | `artifacts` | Output folder for feature artefacts |
| `data_dir` | `data` | Output folder for panel data |
| `log_dir` | `logs` | Created at runtime (logs are not distributed) |

### 3. Process Summary

1. **Integration** - Normalise raw CSVs, build the `(sector_id, panel_month)` skeleton, and merge housing, land, search, POI, and neighbour datasets.  
2. **Feature engineering** - Generate all lag/diff/pct-change/rolling/EWM features, holiday dummies, POI PCA components, and KMeans clusters.  
3. **Imputation** - Fill gaps via sector-level temporal interpolation followed by global averages.  
4. **Feature pruning** - Drop highly collinear features and retain the most informative set via LightGBM importance.  
5. **Outputs** - Persist intermediate panels in `data/` and artefacts in `artifacts/`.

### 4. Final Feature List

`artifacts/selected_features.txt` contains the 106 features actually used in training. They are listed below with brief descriptions (order preserved):

| # | Feature | Description |
|---|---------|-------------|
| 1 | imputed_nh_inventory_turnover | Imputed new-home inventory turnover. |
| 2 | is_future_horizon | Indicator for forecast-horizon rows. |
| 3 | land_planned_building_area_ewm_alpha0.5 | EWM (alpha=0.5) of planned building area. |
| 4 | land_planned_building_area_roll3_max | Rolling 3-month max of planned building area. |
| 5 | land_transaction_amount_diff1 | Month-over-month change in land transaction amount. |
| 6 | land_transaction_amount_lag1 | Land transaction amount lagged 1 month. |
| 7 | land_transaction_amount_roll3_max | Rolling 3-month max of land transaction amount. |
| 8 | macro_over_60_years_percent_yoy_lag1 | Lagged YoY change of population over 60. |
| 9 | month_idx | Sequential month index. |
| 10 | nh_amount_new_house_transactions | Raw new-house transaction amount. |
| 11 | nh_amount_new_house_transactions_diff1 | Month-over-month change of new-house amount. |
| 12 | nh_amount_new_house_transactions_diff12 | Year-over-year change of new-house amount. |
| 13 | nh_amount_new_house_transactions_diff3 | Quarter-over-quarter change of new-house amount. |
| 14 | nh_amount_new_house_transactions_ewm_alpha0.5 | EWM (alpha=0.5) of new-house amount. |
| 15 | nh_amount_new_house_transactions_lag1 | Lag-1 new-house amount. |
| 16 | nh_amount_new_house_transactions_lag12 | Lag-12 new-house amount. |
| 17 | nh_amount_new_house_transactions_lag2 | Lag-2 new-house amount. |
| 18 | nh_amount_new_house_transactions_lag3 | Lag-3 new-house amount. |
| 19 | nh_amount_new_house_transactions_lag6 | Lag-6 new-house amount. |
| 20 | nh_amount_new_house_transactions_pctchg1 | Monthly percentage change of new-house amount. |
| 21 | nh_amount_new_house_transactions_pctchg3 | Quarterly percentage change of new-house amount. |
| 22 | nh_amount_new_house_transactions_roll6_min | Rolling 6-month minimum of new-house amount. |
| 23 | nh_area_new_house_available_for_sale_ewm_alpha0.5 | EWM of salable new-home area. |
| 24 | nh_area_new_house_available_for_sale_pctchg1 | Monthly change of salable area. |
| 25 | nh_area_new_house_available_for_sale_pctchg3 | Quarterly change of salable area. |
| 26 | nh_area_new_house_available_for_sale_roll3_std | 3-month standard deviation of salable area. |
| 27 | nh_area_new_house_available_for_sale_roll6_std | 6-month standard deviation of salable area. |
| 28 | nh_area_new_house_transactions_diff1 | Month-over-month change of new-house area sold. |
| 29 | nh_area_new_house_transactions_diff12 | Year-over-year change of new-house area sold. |
| 30 | nh_area_new_house_transactions_diff3 | Quarter-over-quarter change of new-house area sold. |
| 31 | nh_area_new_house_transactions_ewm_alpha0.5 | EWM of new-house area sold. |
| 32 | nh_area_new_house_transactions_pctchg1 | Monthly percentage change of new-house area sold. |
| 33 | nh_area_new_house_transactions_roll3_std | 3-month standard deviation of new-house area sold. |
| 34 | nh_area_per_unit_new_house_transactions_roll6_max | Rolling 6-month max of area per unit. |
| 35 | nh_inventory_turnover | New-home inventory turnover ratio. |
| 36 | nh_period_new_house_sell_through_diff1 | Month-over-month change in sell-through period. |
| 37 | nh_period_new_house_sell_through_ewm_alpha0.5 | EWM of sell-through period. |
| 38 | nh_period_new_house_sell_through_lag12 | Lag-12 sell-through period. |
| 39 | nh_period_new_house_sell_through_lag6 | Lag-6 sell-through period. |
| 40 | nh_period_new_house_sell_through_pctchg1 | Monthly percentage change in sell-through period. |
| 41 | nh_period_new_house_sell_through_pctchg3 | Quarterly percentage change in sell-through period. |
| 42 | nh_period_new_house_sell_through_roll3_std | 3-month standard deviation of sell-through period. |
| 43 | nh_period_new_house_sell_through_roll6_std | 6-month standard deviation of sell-through period. |
| 44 | nh_price_new_house_transactions_diff1 | Month-over-month change in new-house price. |
| 45 | nh_price_new_house_transactions_ewm_alpha0.5 | EWM of new-house price. |
| 46 | nh_price_new_house_transactions_pctchg1 | Monthly percentage change in new-house price. |
| 47 | nh_price_new_house_transactions_roll3_std | 3-month standard deviation of new-house price. |
| 48 | nh_price_new_house_transactions_roll6_std | 6-month standard deviation of new-house price. |
| 49 | nh_total_price_per_unit_new_house_transactions | Average total price per unit. |
| 50 | nh_total_price_per_unit_new_house_transactions_diff1 | Month-over-month change of average total price. |
| 51 | nh_total_price_per_unit_new_house_transactions_ewm_alpha0.5 | EWM of average total price. |
| 52 | nh_total_price_per_unit_new_house_transactions_lag6 | Lag-6 average total price. |
| 53 | nh_total_price_per_unit_new_house_transactions_pctchg1 | Monthly change of average total price. |
| 54 | nh_total_price_per_unit_new_house_transactions_roll3_std | 3-month standard deviation of average total price. |
| 55 | nh_total_price_per_unit_new_house_transactions_roll6_min | Rolling 6-month minimum of average total price. |
| 56 | nhnbr_amount_new_house_transactions_nearby_sectors_ewm_alpha0.5 | EWM of neighbouring sectors' transaction amounts. |
| 57 | nhnbr_amount_new_house_transactions_nearby_sectors_lag12 | Lag-12 neighbouring transaction amounts. |
| 58 | nhnbr_amount_new_house_transactions_nearby_sectors_pctchg1 | Monthly change of neighbouring transaction amounts. |
| 59 | nhnbr_amount_new_house_transactions_nearby_sectors_roll6_min | Rolling 6-month minimum of neighbouring transaction amounts. |
| 60 | nhnbr_area_per_unit_new_house_transactions_nearby_sectors_pctchg3 | Quarterly change of neighbouring area per unit. |
| 61 | nhnbr_period_new_house_sell_through_nearby_sectors_pctchg3 | Quarterly change of neighbouring sell-through period. |
| 62 | nhnbr_price_new_house_transactions_nearby_sectors_diff3 | Quarter-over-quarter change of neighbouring new-house price. |
| 63 | panel_month | Panel month timestamp. |
| 64 | po_amount_pre_owned_house_transactions_diff1 | Month-over-month change of pre-owned amount. |
| 65 | po_amount_pre_owned_house_transactions_diff12 | Year-over-year change of pre-owned amount. |
| 66 | po_num_pre_owned_house_transactions_diff12 | Year-over-year change of pre-owned transaction count. |
| 67 | po_num_pre_owned_house_transactions_roll3_std | 3-month standard deviation of pre-owned transaction count. |
| 68 | po_price_pre_owned_house_transactions_ewm_alpha0.5 | EWM of pre-owned price. |
| 69 | poi_pca_component_1 | First POI principal component. |
| 70 | poi_pca_component_5 | Fifth POI principal component. |
| 71 | ponbr_amount_pre_owned_house_transactions_nearby_sectors_pctchg3 | Quarterly change of neighbouring pre-owned amount. |
| 72 | ponbr_amount_pre_owned_house_transactions_nearby_sectors_roll3_std | 3-month standard deviation of neighbouring pre-owned amount. |
| 73 | ponbr_area_pre_owned_house_transactions_nearby_sectors_diff3 | Quarter-over-quarter change of neighbouring pre-owned area. |
| 74 | ponbr_area_pre_owned_house_transactions_nearby_sectors_lag12 | Lag-12 neighbouring pre-owned area. |
| 75 | ponbr_area_pre_owned_house_transactions_nearby_sectors_roll6_min | Rolling 6-month minimum of neighbouring pre-owned area. |
| 76 | ponbr_price_pre_owned_house_transactions_nearby_sectors_ewm_alpha0.5 | EWM of neighbouring pre-owned price. |
| 77 | ponbr_price_pre_owned_house_transactions_nearby_sectors_pctchg12 | Year-over-year change of neighbouring pre-owned price. |
| 78 | ponbr_price_pre_owned_house_transactions_nearby_sectors_roll3_std | 3-month standard deviation of neighbouring pre-owned price. |
| 79 | ponbr_price_pre_owned_house_transactions_nearby_sectors_roll6_std | 6-month standard deviation of neighbouring pre-owned price. |
| 80 | search_an_zhi_yi_dong_all_pctchg3 | Quarterly change in relocation search interest. |
| 81 | search_dai_kuan_li_lv_yi_dong_all_lag6 | Lag-6 loan-rate search interest. |
| 82 | search_er_shou_fang_shi_chang_pc_all_diff12 | Year-over-year change in second-hand market PC searches. |
| 83 | search_fang_di_chan_kai_fa_yi_dong_all_pctchg1 | Monthly change in development-related searches. |
| 84 | search_fang_di_chan_shui_pc_all_diff3 | Quarter-over-quarter change in property-tax PC searches. |
| 85 | search_fang_di_chan_shui_yi_dong_all_diff3 | Quarter-over-quarter change in property-tax mobile searches. |
| 86 | search_fang_jia_shang_zhang_yi_dong_all_diff1 | Month-over-month change in price-increase mobile searches. |
| 87 | search_fang_jia_tiao_kong_pc_all_diff1 | Month-over-month change in price-control PC searches. |
| 88 | search_fang_jia_tiao_kong_pc_all_pctchg12 | Year-over-year change in price-control PC searches. |
| 89 | search_fang_jia_yi_dong_all | Housing-price mobile search index. |
| 90 | search_fang_jia_yi_dong_all_lag1 | Lag-1 housing-price mobile search index. |
| 91 | search_fang_jia_yi_dong_all_lag3 | Lag-3 housing-price mobile search index. |
| 92 | search_fang_jia_zou_shi_pc_all_lag1 | Lag-1 housing-price-trend PC search index. |
| 93 | search_fang_wu_zhuang_xiu_pc_all_pctchg3 | Quarterly change in renovation PC searches. |
| 94 | search_fang_wu_zhuang_xiu_yi_dong_all_lag2 | Lag-2 renovation mobile search index. |
| 95 | search_peng_hu_qu_yi_dong_all | Penghu-related mobile search index. |
| 96 | search_qi_shui_pc_all | Deed-tax PC search index. |
| 97 | search_qu_ku_cun_yi_dong_all_pctchg1 | Monthly change in inventory mobile searches. |
| 98 | search_rong_zi_pc_all | Financing PC search index. |
| 99 | search_rong_zi_yi_dong_all_lag1 | Lag-1 financing mobile search index. |
| 100 | search_shui_fei_yi_dong_all_diff3 | Quarter-over-quarter change in tax/fee mobile searches. |
| 101 | search_xian_gou_pc_all_diff1 | Month-over-month change in purchase-restriction PC searches. |
| 102 | search_xian_shou_yi_dong_all_pctchg12 | Year-over-year change in sale-restriction mobile searches. |
| 103 | search_xue_qu_fang_yi_dong_all | School-district mobile search index. |
| 104 | search_xue_qu_fang_yi_dong_all_lag1 | Lag-1 school-district mobile search index. |
| 105 | sector_id | Numeric sector identifier. |
| 106 | sector_label | Sector label string (retained for completeness). |

---

## Model Training

### 1. Command

```bash
py -3 train_model.py --base-dir . --config config/train_model.yaml
```

The run takes roughly **2.5 hours**.

### 2. Configuration Highlights (`config/train_model.yaml`)

- Target column: `nh_amount_new_house_transactions`  
- Log-transform and prediction clipping enabled.  
- Target encoding on `sector_id` saved as `models/sector_te_mapping.json`.  
- Time-based 5-fold cross validation ending at 2024-08.  
- Exponential sample weighting (`half_life_months: 12`).  
- Optuna trials: 40 each for LightGBM/XGBoost/CatBoost; 1 for Random Forest.  
- Ensemble optimiser: constrained SciPy solver with uniform initial weights.

### 3. Workflow

1. Parse configuration, load features (`selected_features.txt`) and panel (`panel_selected.parquet`), compute sample weights.  
2. Fit target encoding within each fold (no leakage) and persist the mapping.  
3. Run Optuna to explore hyperparameters; store the best values in `*_best_params.json`. Save `xgboost_scaler.pkl` for XGBoost.  
4. Refit each model on all weighted data using the best hyperparameters; save `*_final.pkl` plus metadata JSON.  
5. Aggregate validation predictions and solve for non-negative ensemble weights (sum to one); save as `ensemble_weights.json`.

### 4. Models

| Model | Key search dimensions | Notes |
|-------|-----------------------|-------|
| LightGBM | `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `colsample_bytree`, `n_estimators`, `device_type="gpu"` | Major contributor |
| XGBoost | `learning_rate`, `max_depth`, `min_child_weight`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`, `tree_method="hist"`, `device="cuda"` | Uses `xgboost_scaler.pkl` |
| CatBoost | `depth`, `learning_rate`, `l2_leaf_reg`, `bagging_temperature`, `grow_policy`, `iterations`, `task_type="GPU"` | Receives highest ensemble weight |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `criterion` | Included for diversity (final weight 0) |

---

## Inference and Submission Generation

### 1. Command

```bash
# Requires rebuilt panel files (see [Panel Construction](#panel-construction))
py -3 predict_future.py ^
    --base-dir . ^
    --config config/train_model.yaml ^
    --output submission_te_final.csv
```

Optional flags:

- `--future-months YYYY-MM,YYYY-MM,...` to override the forecast horizon.  
- `--recompute-panel` to rebuild the panel before inference.  
- `--reporting` to generate diagnostic plots (not included here by default).

### 2. Workflow Summary

| Step | Description |
|------|-------------|
| Load config and artefacts | Read YAML, feature list, panel, target encoding map, scalers |
| Base model predictions | Run LightGBM, XGBoost, CatBoost, Random Forest models |
| Ensemble weighting | Apply manually tuned weights `[0.3025, 0.1168, 0.5807, 0.0]`; reviewers may edit `models/ensemble_weights.json` to test alternatives |
| Zero-sector override | Overwrite predictions to 0 for sectors with any zero transaction in the previous six months (short-term risk control) |
| Output | Save `submission_te_final.csv` and a timestamped parquet with detailed predictions |

---

## Model and Artifact Inventory

| File | Purpose |
|------|---------|
| `models/lightgbm_final.pkl` | Final LightGBM model |
| `models/xgboost_final.pkl` | Final XGBoost model |
| `models/catboost_final.pkl` | Final CatBoost model |
| `models/random_forest_final.pkl` | Final Random Forest model |
| `models/*_best_params.json` | Best hyperparameter sets |
| `models/*_metadata.json` | Training summaries (score, timestamp, sample count) |
| `models/xgboost_scaler.pkl` | Standard scaler for XGBoost inputs |
| `models/sector_te_mapping.json` | Target encoding mapping |
| `models/ensemble_weights.json` | Ensemble weights `[0.3025, 0.1168, 0.5807, 0.0]` |
| `submission_te_final.csv` | Final submission file |
| `artifacts/*` | Feature lists and engineering artefacts supporting reproducibility |

---

## Reproducibility Notes

- The configuration sets `seed: 8888` and propagates it to Python `random`, NumPy, and individual models.  
- **Nevertheless, retraining yields different results.**  
  1. GPU algorithms (LightGBM, XGBoost, CatBoost) rely on atomic operations whose accumulation order changes between runs.  
  2. Multi-threaded execution (GPU or CPU) alters floating-point summation order.  
  3. Optuna's exploration depends on the non-deterministic learners, so search trajectories diverge even with identical seeds.  
  4. Library or driver versions (BLAS, CUDA, GPU drivers) affect numerical rounding.  
- **Recommendation:** use the packaged checkpoints under `models/` for verification. Retrain only if required, and expect slight metric drift even with the same seed.

---

## Compliance and Originality Statement

- No external public datasets or third-party models were used; all raw data originate from the official Kaggle competition.  
- All code, feature engineering routines, and model training scripts were authored and executed by our team, fully compliant with Kaggle policies.  
- Dependencies follow their respective open-source licences (see `requirements.txt`).

For further clarification, please contact us; we are happy to assist with verification.
