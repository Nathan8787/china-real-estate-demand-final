# README.spec

此文件列出正式版 README.md 必須涵蓋的內容與對應細節，撰寫 README 時請逐項完成。

## 1. 專案概覽
- 簡短介紹競賽：China Real Estate Demand Prediction（Kaggle）與最終成果。
- 說明此套件的目的：重現 v4 最終提交（build panel → train → ensemble 推論）。
- 整體目錄結構概述：`build_panel.py`, `train_model.py`, `predict_future.py`, `download_kaggle_data.py`, `config/`, `data/`, `artifacts/`, `models/`, `utils/` 等。

## 2. 環境資訊
- Python 版本、作業系統、硬體（GPU/CPU 型號）以及是否使用 GPU。
- 安裝方式：使用 `requirements.txt`。提醒 CatBoost、LightGBM GPU 需額外依賴。
- 若使用 Conda/virtualenv，可提供範例指令。

## 3. 下載資料
- 說明 `train/` raw CSV 不包含在檔案中，需使用 `download_kaggle_data.py` 或 Kaggle CLI 手動下載。
- 提供使用範例：
  ```bash
  py -3 download_kaggle_data.py --output-dir train --force
  ```
- 提醒需要事先配置 `kaggle.json` 或環境變數。

## 4. 預處理與面板建構
- 描述 `build_panel.py` 的用途：轉換 raw CSV → `data/panel_with_raw.pkl` → `panel_with_features.pkl` → `panel_imputed.pkl` → `panel_selected.parquet`。
- 說明 `config/panel_build.yaml` 參數、產出的中間檔以及寫入 `artifacts/` 的內容（`selected_features.txt`, `correlation_drop.csv`, `feature_importance.csv`, `poi_*` 等）。
- 提供重新建 panel 的指令範例：
  ```bash
  py -3 build_panel.py --base-dir .
  ```

## 5. 模型訓練
- 說明此套件直接提供最終模型檔（位於 `models/`），可直接推論。
- 若要重新訓練：指令範例
  ```bash
  py -3 train_model.py --base-dir . --config config/train_model.yaml
  ```
- 解釋訓練流程：Optuna/自訂搜尋、LightGBM、XGBoost、CatBoost、RandomForest。列出 `models/*_best_params.json`, `*_final.pkl`, `ensemble_weights.json`, `sector_te_mapping.json`, `xgboost_scaler.pkl`。
- 說明 GPU/多執行緒導致的非 determinism：即便固定 seed（config `seed: 888888` 等），因為 GPU 累加順序、不同執行緒、不同行與 MLC/BLAS 版本，重新訓練結果可能與公開模型略有差異，因此建議使用提供的 checkpoint。
- 記錄實際訓練硬體、耗時（近似即可），或至少提供估計。

## 6. 推論流程
- 指出主程式為 `predict_future.py`（內建 zero-sector overrides）。
- 基本指令：
  ```bash
  py -3 predict_future.py --base-dir . --config config/train_model.yaml --output submission_te_final.csv --reporting
  ```
- 說明輸入：`data/panel_selected.parquet`、`artifacts/selected_features.txt`、`models/` 中模型與 `ensemble_weights.json`、`sector_te_mapping.json` 等。
- 說明輸出：`logs/` 下的預測資訊、`submission_te_final.csv`。
- 述及任意選項：
  - `--future-months` 可覆寫預測月份。
  - `--recompute-panel` 會呼叫 `build_panel.py` 重新建 panel。
  - `--reporting` 生成加值報表/圖表（使用 `utils.reporting`）。

### 6.1 Ensemble 權重調整（重點補充）
- 在推論前會載入 `models/ensemble_weights.json`。使用者可手動調整 `model_names` 與 `weights` 內容；目前設為：
  ```
  weights: [0.3025, 0.1168, 0.5807, 0.0]
  model_names: ["lightgbm", "xgboost", "catboost", "random_forest"]
  ```
- 說明不同權重組合會影響最終 submission 表現，建議在 README 中標示如何修改與重新推論。

### 6.2 Zero-Sector 覆寫策略（重點補充）
- 程式內建 `ZERO_SECTOR_OVERRIDES`，會將「近六個月內曾出現交易量為 0」的地區預測值強制設為 0。
- 說明判斷方式：從近一個月到近八個月測試後，選擇六個月效果最佳，因此 `predict_future.py` 會在 ensemble 完成後覆寫這些區域。
- 列出範例 log，例如 `載入內建 zero-sector 清單，將覆寫 240 組 month/sector 為 0`。

## 7. 模型檔案清單
- 逐檔說明 `models/` 與 `artifacts/` 內重要檔案作用、產出流程、檔案大小（可選）。
- 說明 `submission_te_final.csv` 是依上述流程生成，為最終獲獎提交。

## 8. 隨機種子與可重現性
- 明確列出 config 中的 `seed`、Optuna 相關設定、`PYTHONHASHSEED` 等建議環境變數。
- 說明仍會因 GPU / 多執行緒產生微差；官方審查應使用提供的模型檔驗證。

## 9. 外部資源與授權
- Kaggle 原始資料授權（連結至競賽主頁）。
- 使用的第三方套件、版本（可在 `requirements.txt` 提醒）。
- 若有其它公開資源（如人口資料、地圖等），需列出並附來源／授權說明（依實際情況補充）。

## 10. 原創性聲明
- 簡述：所有程式碼與模型均為團隊自行撰寫、符合 Kaggle 規範；若有競賽規則中特別要求的聲明亦一併加入。

## 11. 期望輸出與驗證
- 建議在 README 稍微提及如何檢查輸出：`submission_te_final.csv` 的 row 數、欄位名稱、基本統計等。
- 如果時間允許，可提供 `logs/reports/` 的檢查方式。

## 12. 常見問題與故障排除（選填）
- GPU 依賴不足、CatBoost 安裝錯誤可能遇到的訊息與解法。
- 若 `predict_future.py` 找不到 `config/train_model_v2.yaml`，提醒使用 `--config` 指定或改回舊檔名。

