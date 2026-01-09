# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path

# === 1. 讀取資料 =========================================================
input_csv = "駕駛室資料/cab_data.csv"
print(f"📖 正在讀取 {input_csv} ...")
df = pd.read_csv(input_csv)

# [保留功能] 如果欄位名稱是 DateTime，自動改為 datetime
if 'DateTime' in df.columns:
    df.rename(columns={'DateTime': 'datetime'}, inplace=True)

# === 2. 定義要刪除的欄位清單 ==============================================
cols_to_drop = [
    'mean_60s', 'root_amp_60s', 'rms_60s', 'std_60s', 'max_60s', 
    'skewness_60s', 'kurtosis_60s', 'crest_factor_60s', 'clearance_factor_60s', 
    'shape_factor_60s', 'impulse_factor_60s', 'peak_to_peak_60s', 'rss_60s'
]

# === 3. 執行刪除 =========================================================
# errors='ignore' 代表如果 CSV 裡本來就沒有這些欄位，程式不會報錯，直接繼續
df.drop(columns=cols_to_drop, errors='ignore', inplace=True)
print("🗑️ 已移除統計特徵欄位")

# === 4. 輸出 CSV =========================================================
out_dir = Path("dataset")
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "cab_data_123.csv"

# 儲存 (保留小數點後 2 位設定)
df.to_csv(out_path, index=False, encoding='utf-8', float_format='%.2f')

print("✅ 處理完成！已產生：", out_path.resolve())