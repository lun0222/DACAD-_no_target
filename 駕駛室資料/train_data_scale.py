# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path

# === 1. 讀完整資料 =========================================================
df_all = pd.read_csv("train_data.csv")
df_all['datetime'] = pd.to_datetime(df_all['datetime'])

# === 2. 來源時間參數設定 (要複製哪一段原始資料) ================================
# start_time = pd.Timestamp("2024-06-17 14:20:00")
# end_time   = pd.Timestamp("2024-06-17 16:20:00")

start_time = pd.Timestamp("2025-02-07 11:28:00")
end_time   = pd.Timestamp("2025-02-07 13:03:00")
duration   = end_time - start_time  # 計算區段長度

# === 3. 多欄位故障設定 ======================================================
# 注意：列表內的數值數量決定了會產生幾段資料
# 例如：[0.9, 0.8, 0.7] 會產生 3 段連續的時間資料
fault_specs = {
    # 壓縮機2025-03-01 00:00:14   2025-03-01 05:58:09
    # 'comp_current_1':  ("scale_step", [1.1, 1.2, 1.3]),
    # 'comp_current_2':  ("scale_step", [1.1, 1.2, 1.3]),
    # 'hp_comp_1':  ("scale_step", [1.02, 1.04, 1.06]),
    # 'hp_comp_2':  ("scale_step", [1.02, 1.04, 1.06]),
    # 'lp_comp_1':  ("scale_step", [1.01, 1.02, 1.03]),
    # 'lp_comp_2':  ("scale_step", [1.01, 1.02, 1.03]),

    # # 冷凝風扇2025-03-01 06:00:14   2025-03-01 11:58:09
    # 'cond_current_1':  ("scale_step", [1.1, 1.2, 1.3]),
    # 'cond_current_2':  ("scale_step", [1.1, 1.2, 1.3]),
    # 'hp_comp_1':  ("scale_step", [1.08, 1.15, 1.23]),
    # 'hp_comp_2':  ("scale_step", [1.08, 1.15, 1.23]),
    # 'comp_current_1':  ("scale_step", [1.05, 1.1, 1.15]),
    # 'comp_current_2':  ("scale_step", [1.05, 1.1, 1.15]),
    # 'lp_comp_1':  ("scale_step", [1.02, 1.04, 1.06]),
    # 'lp_comp_2':  ("scale_step", [1.02, 1.04, 1.06]),

    # # 蒸發風扇2025-03-01 12:00:14   2025-03-01 17:58:09
    # 'fan_current_1':  ("scale_step", [1.1, 1.2, 1.3]),
    # 'fan_current_2':  ("scale_step", [1.1, 1.2, 1.3]),
    # 'lp_comp_1':  ("shift_step", [0.98, 0.96, 0.94]),
    # 'lp_comp_2':  ("shift_step", [0.98, 0.96, 0.94]),
    # 'comp_current_1':  ("shift_step", [0.99, 0.98, 0.97]),
    # 'comp_current_2':  ("shift_step", [0.99, 0.98, 0.97]),
    # 'hp_comp_1':  ("shift_step", [0.99, 0.98, 0.97]),
    # 'hp_comp_2':  ("shift_step", [0.99, 0.98, 0.97]),
    
    # # 加熱器2025-03-01 18:00:53   2025-03-01 22:44:59
    'heater_temp':  ("scale_step", [0.9, 0.8, 0.7]),
    'return_air_temp':  ("scale_step", [0.98, 0.96, 0.94]),
}

# 取得要產生的段數 (以第一個欄位的設定長度為準)
num_scenarios = len(next(iter(fault_specs.values()))[1])
print(f"預計產生 {num_scenarios} 個連續區段...")

# === 4. 抽出基礎區段 ========================================================
base_segment = df_all[df_all['datetime'].between(start_time, end_time)].copy()
base_segment = base_segment.drop_duplicates('datetime').sort_values('datetime')

# 設定模擬起始時間點
sim_start_base = pd.Timestamp("2025-03-01 18:00:00")

# === 5. 迴圈生成多段資料 (取代原本的分段邏輯) =================================
generated_segments = []

for i in range(num_scenarios):
    # 複製一份基礎資料
    current_segment = base_segment.copy()
    
    # 針對每個設定欄位進行整段調整
    for col, (mode, step_values) in fault_specs.items():
        if col not in current_segment.columns:
            continue
            
        # 取得當前段數對應的數值 (例如第 i 個值)
        # 如果設定值不夠長，就取最後一個
        val = step_values[i] if i < len(step_values) else step_values[-1]
        
        if mode == "scale_step":
            current_segment[col] *= val
        elif mode == "shift_step":
            current_segment[col] += val
            
    # === 6. 時間平移計算 ===
    # 每一段的時間起點 = 基礎起點 + (第 i 段 * 單段長度)
    # 這樣會把資料變成：[第1段][第2段][第3段] 連續接在一起
    current_target_start = sim_start_base + (i * duration)
    
    # 計算該段的時間位移量
    time_shift = current_target_start - start_time
    current_segment['datetime'] += time_shift
    
    generated_segments.append(current_segment)

# 合併所有生成的片段
sim_df = pd.concat(generated_segments)

# === 7. 檢查並刪除目標區間的舊資料 ==========================================
# 取得新資料的時間範圍
sim_start = sim_df['datetime'].min()
sim_end = sim_df['datetime'].max()

print(f"-> 準備插入資料範圍: {sim_start} 至 {sim_end}")

# 建立遮罩：找出原始資料中，落在這個時間範圍內的資料
mask_exist = (df_all['datetime'] >= sim_start) & (df_all['datetime'] <= sim_end)
cnt_exist = mask_exist.sum()

if cnt_exist > 0:
    print(f"⚠️ 警告：目標時間段內已有 {cnt_exist} 筆資料，正在刪除舊資料以避免重複...")
    df_all = df_all[~mask_exist].copy()
else:
    print("✅ 目標時間段內無資料，直接拼接。")

# === 8. 合併並輸出 CSV =====================================================
final_df = pd.concat([df_all, sim_df]).sort_values('datetime').reset_index(drop=True)
final_df = final_df[df_all.columns]  # 保持欄位順序

out_dir = Path("dataset"); out_dir.mkdir(exist_ok=True)
out_path = out_dir/"train_data.csv"

# 保持兩位小數格式
final_df.to_csv(out_path, index=False, encoding='utf-8', float_format='%.2f')

print("✅ 已產生：", out_path.resolve())
print(f"📊 總共新增資料筆數: {len(sim_df)}")
print("🔍 重複時間戳 = ", final_df['datetime'].duplicated().sum())