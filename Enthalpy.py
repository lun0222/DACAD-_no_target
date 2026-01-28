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
output_csv_path = r'駕駛室資料\cab_data_final_train.csv' # 最終訓練用檔案

# 圖片輸出資料夾
image_output_folder = 'analysis_results_images'
if not os.path.exists(image_output_folder):
    os.makedirs(image_output_folder)

# 單位設定 (確認為 bar 與 °C)
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
    print(f"資料讀取成功，共 {len(df)} 筆。")
except Exception as e:
    print(f"讀取失敗: {e}")
    exit()

# ==========================================
# 4. 全域計算比焓並新增欄位 (含四捨五入)
# ==========================================
def calculate_h_suc_row(row):
    """計算單行吸氣比焓的函數"""
    try:
        # 取得原始數據 (bar 和 °C)
        p_low_bar = row['lp_comp_1']
        superheat_c = row['superheat_1']
        
        # 單位轉換
        p_low_pa = (p_low_bar * P_UNIT_CONVERTER) + ATM_PRESSURE
        
        # 計算 R407C 露點溫度 (Kelvin)
        t_dew_k = CP.PropsSI('T', 'P', p_low_pa, 'Q', 1, 'R407C')
        
        # 反推吸氣溫度 (Kelvin)
        t_suction_k = t_dew_k + superheat_c
        
        # 計算比焓 (J/kg -> kJ/kg)
        h_val = CP.PropsSI('H', 'P', p_low_pa, 'T', t_suction_k, 'R407C')
        
        # 【修改處】轉成 kJ/kg 並四捨五入到小數點後 2 位
        return round(h_val / 1000.0, 2) 
        
    except:
        return np.nan

print("\n正在進行全域物理特徵計算 (取小數點後兩位)...")
df['h_suc_1'] = df.apply(calculate_h_suc_row, axis=1)
print("計算完成！'h_suc_1' 已新增至資料表最後一欄。")

# ==========================================
# 5. 迴圈繪圖並存檔
# ==========================================
print(f"\n開始繪製 {len(selected_periods)} 個時段的分析圖...")

for start_time, end_time, label in selected_periods:
    mask = (df['datetime'] >= start_time) & (df['datetime'] <= end_time)
    df_period = df.loc[mask].copy()
    
    if df_period.empty: continue
    
    # 篩選運轉中且計算成功的數據
    df_plot = df_period[(df_period['comp_current_1'] > 2) & (df_period['h_suc_1'].notna())].copy()
    
    if df_plot.empty: continue
        
    # --- 繪圖 ---
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 左軸：低壓 (bar)
    color_p = '#1f77b4' # 深藍
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Low Pressure (bar)', color=color_p, fontweight='bold')
    ax1.plot(df_plot['datetime'], df_plot['lp_comp_1'], color=color_p, label='Low Pressure')
    ax1.tick_params(axis='y', labelcolor=color_p)
    ax1.grid(True, alpha=0.3)
    
    # 右軸：比焓 (kJ/kg)
    ax2 = ax1.twinx()
    color_h = '#d62728' # 深紅
    ax2.set_ylabel('Suction Enthalpy (kJ/kg)', color=color_h, fontweight='bold')
    ax2.plot(df_plot['datetime'], df_plot['h_suc_1'], color=color_h, label='Suction Enthalpy')
    ax2.tick_params(axis='y', labelcolor=color_h)
    
    plt.title(f'{label}\n({start_time} ~ {end_time})', fontsize=14)
    fig.tight_layout()
    
    # 存檔
    time_str = str(start_time).replace(':', '').replace('-', '').replace(' ', '_')[:13] 
    filename = f"{image_output_folder}/{time_str}_{label}.png"
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"  -> 圖片已儲存: {filename}")

# ==========================================
# 6. 儲存最終訓練資料集
# ==========================================
print("\n正在儲存最終資料集...")
df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

print(f"★ 任務完成！")
print(f"1. 圖片已儲存於 '{image_output_folder}'。")
print(f"2. 訓練用 CSV 已儲存為: '{output_csv_path}' (h_suc_1 已四捨五入)。")