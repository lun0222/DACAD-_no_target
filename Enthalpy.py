import pandas as pd
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
import numpy as np
import os

# ==========================================
# 1. 基礎設定
# ==========================================
# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# 檔案路徑設定
input_file_path = r'駕駛室資料\cab_data.csv'
output_csv_path = r'駕駛室資料\cab_data_final_train.csv'

# 圖片輸出資料夾 (建議改個名字區分，例如 _split)
image_output_folder = 'analysis_results_split'
if not os.path.exists(image_output_folder):
    os.makedirs(image_output_folder)

# 單位設定
P_UNIT_CONVERTER = 100000 # 1 bar = 100,000 Pa
ATM_PRESSURE = 101325     # 標準大氣壓 Pa

# ==========================================
# 2. 定義時段列表
# ==========================================
selected_periods = [
    ('2025-04-11 09:14:00', '2025-04-11 09:44:00', '冷凝盤管阻塞20%'),
    ('2025-04-11 09:46:00', '2025-04-11 10:16:00', '冷凝盤管阻塞30%'),
    ('2025-04-11 10:29:00', '2025-04-11 10:59:00', '蒸發盤管阻塞10%'),
    ('2025-04-11 11:01:00', '2025-04-11 11:31:00', '蒸發盤管阻塞20%'),
    ('2025-04-11 11:33:00', '2025-04-11 12:03:00', '蒸發盤管阻塞23%'),
    ('2025-04-11 12:05:00', '2025-04-11 13:35:00', '正常資料常溫34度'),
    ('2025-04-11 13:45:00', '2025-04-11 14:15:00', '蒸發風扇電流90%'),
    ('2025-04-11 14:19:00', '2025-04-11 14:49:00', '蒸發風扇電流80%'),
    ('2025-04-11 14:52:00', '2025-04-11 15:22:00', '蒸發風扇電流70%'),
    ('2025-04-11 15:46:00', '2025-04-11 16:16:00', '正常資料高溫42.7度'),
    ('2025-04-14 09:00:00', '2025-04-14 10:00:00', '正常資料低溫24度'),
    ('2025-04-14 13:25:00', '2025-04-14 14:25:00', '冷媒洩漏10%'),
    ('2025-04-14 14:50:00', '2025-04-14 15:50:00', '冷媒洩漏20%'),
    ('2025-12-23 10:23:00', '2025-12-23 11:23:00', '正常資料低溫27度'),
    ('2025-12-23 12:15:00', '2025-12-23 13:15:00', '正常資料常溫30度'),
    ('2025-12-23 13:19:00', '2025-12-23 13:57:00', '輕度(兩層洗衣袋)冷凝盤管阻塞'),
    ('2025-12-23 14:02:00', '2025-12-23 14:33:00', '重度(四層洗衣袋)冷凝盤管阻塞'),
    ('2025-12-23 15:00:00', '2025-12-23 16:00:00', '冷媒洩漏30%'),
    ('2026-01-01 00:00:00', '2026-01-01 01:30:00', '壓縮機故障10%'),
    ('2026-01-01 01:30:00', '2026-01-01 03:00:00', '壓縮機故障20%'),
    ('2026-01-01 03:00:00', '2026-01-01 04:30:00', '壓縮機故障30%'),
    ('2026-01-01 05:00:00', '2026-01-01 06:30:00', '冷凝風扇電流上升10%'),
    ('2026-01-01 06:30:00', '2026-01-01 08:00:00', '冷凝風扇電流上升20%'),
    ('2026-01-01 08:00:00', '2026-01-01 09:30:00', '冷凝風扇電流上升30%'),
    ('2026-01-01 10:00:00', '2026-01-01 11:30:00', '蒸發風扇電流上升10%'),
    ('2026-01-01 11:30:00', '2026-01-01 13:00:00', '蒸發風扇電流上升20%'),
    ('2026-01-01 13:00:00', '2026-01-01 14:30:00', '蒸發風扇電流上升30%'),
]

# ==========================================
# 3. 讀取資料
# ==========================================
print(f"正在讀取原始資料: {input_file_path} ...")
try:
    df = pd.read_csv(input_file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
except Exception as e:
    print(f"讀取失敗: {e}")
    exit()

# ==========================================
# 4. 全域計算比焓
# ==========================================
def calculate_h_suc_row(row):
    try:
        p_low_bar = row['lp_comp_1']
        superheat_c = row['superheat_1']
        p_low_pa = (p_low_bar * P_UNIT_CONVERTER) + ATM_PRESSURE
        t_dew_k = CP.PropsSI('T', 'P', p_low_pa, 'Q', 1, 'R407C')
        t_suction_k = t_dew_k + superheat_c
        h_val = CP.PropsSI('H', 'P', p_low_pa, 'T', t_suction_k, 'R407C')
        return round(h_val / 1000.0, 2)
    except:
        return np.nan

print("\n正在計算比焓特徵...")
df['h_suc_1'] = df.apply(calculate_h_suc_row, axis=1)

# ==========================================
# 5. 【修改處】上下分開繪圖
# ==========================================
print(f"\n開始繪製 {len(selected_periods)} 個時段的分析圖 (上下分割版)...")

for start_time, end_time, label in selected_periods:
    mask = (df['datetime'] >= start_time) & (df['datetime'] <= end_time)
    df_period = df.loc[mask].copy()
    
    if df_period.empty: continue
    
    # 篩選運轉中數據
    df_plot = df_period[(df_period['comp_current_1'] > 2) & (df_period['h_suc_1'].notna())].copy()
    if df_plot.empty: continue
        
    # --- 建立上下兩個子圖 ---
    # nrows=2, ncols=1, sharex=True 代表共用 X 軸(時間)
    # figsize 加高一點 (12, 10)，這樣上下兩張圖才不會太擠
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)
    
    # --- 上圖：低壓 (藍色) ---
    color_p = '#1f77b4' # 深藍
    ax1.plot(df_plot['datetime'], df_plot['lp_comp_1'], color=color_p, label='Low Pressure')
    ax1.set_ylabel('Low Pressure (bar)', color=color_p, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_p)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_title(f'{label}\n({start_time} ~ {end_time})', fontsize=14) # 標題放在最上面
    ax1.legend(loc='upper left')

    # --- 下圖：吸氣比焓 (紅色) ---
    color_h = '#d62728' # 深紅
    ax2.plot(df_plot['datetime'], df_plot['h_suc_1'], color=color_h, label='Suction Enthalpy')
    ax2.set_ylabel('Suction Enthalpy (kJ/kg)', color=color_h, fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=12) # 時間軸標籤只顯示在最下面
    ax2.tick_params(axis='y', labelcolor=color_h)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper left')
    
    # 調整佈局，避免標籤重疊
    plt.tight_layout()
    
    # 存檔
    time_str = str(start_time).replace(':', '').replace('-', '').replace(' ', '_')[:13] 
    filename = f"{image_output_folder}/{time_str}_{label}_split.png"
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"  -> 圖片已儲存: {filename}")

# ==========================================
# 6. 儲存 CSV
# ==========================================
print("\n正在儲存訓練資料...")
df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
print("全部完成！請查看 'analysis_results_split' 資料夾。")