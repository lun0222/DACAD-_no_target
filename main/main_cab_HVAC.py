import os
import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from matplotlib.patches import Patch
import time 
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import shutil 

if __name__ == '__main__':
    # 獲取 main_HVAC.py 所在的目錄
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 獲取專案根目錄
    project_root = os.path.dirname(current_dir)
    
    # 1. 設定您的 HVAC 數據集路徑
    dataset_path = os.path.join(project_root, 'datasets', 'HVAC')
    
    if not os.path.exists(dataset_path):
        print(f"錯誤：找不到數據集路徑 {dataset_path}")
        sys.exit()

    # 3. 獲取相關 script 的絕對路徑
    train_script = os.path.join(current_dir, 'train.py')
    eval_script = os.path.join(current_dir, 'eval.py')
    plot_script = os.path.join(current_dir, 'plot.py')
    
    python_executable = sys.executable
    
    # *** 設定特徵 ***
    target_features = [
    # 冷凝盤管阻塞 (Condenser Coil Fault)
    # 'hp_comp_1','lp_comp_1','cond_current_1','comp_current_1','return_air_temp' ,'outdoor_temp'

    # 蒸發盤管阻塞 (Evaporator Coil Fault)
    # 'fan_current_1','hp_comp_1','lp_comp_1','comp_current_1','return_air_temp','outdoor_temp'

    # 冷媒洩漏 (Refrigerant Leak Fault)
    # 'hp_comp_1','lp_comp_1','superheat_1','comp_current_1','return_air_temp','outdoor_temp'

    # 壓縮機故障 (Compressor Fault)
    # 'hp_comp_1','lp_comp_1','comp_current_1','cond_current_1','return_air_temp','outdoor_temp'

    # 冷凝風扇故障
    # 'hp_comp_1','cond_current_1','comp_current_1','return_air_temp','outdoor_temp'

    # 蒸發風扇故障
    'fan_current_1','hp_comp_1','lp_comp_1','comp_current_1','return_air_temp','outdoor_temp'

    #加熱器
    # 'heater_temp','return_air_temp','outdoor_temp'
    ]
    
    # 4. 明確指定 src 和 trg
    src = "source_data"
    trg = "target_data"
    

    # 2. 定義結果資料夾的變數
    experiments_main_folder = 'results'
    experiment_folder = 'HVAC'

    print(f'======= 正在執行: src: {src} / target: {trg} =======')

    # 5. 定義訓練命令
    command_train = [
        python_executable, train_script,
        '--algo_name', 'dacad',
        '--experiment_folder', experiment_folder,
        '--path_src', dataset_path,
        '--path_trg', dataset_path,
        '--id_src', src,
        '--id_trg', trg,
        '--num_epochs', '50',
        '--batch_size', '128',
        '--eval_batch_size', '256',
        '--learning_rate', '1e-4',
        '--dropout', '0.1',
        '--weight_decay', '1e-4',
        '--num_channels_TCN', '128-256-512',
        '--dilation_factor_TCN', '3',
        '--kernel_size_TCN', '7',
        '--hidden_dim_MLP', '1024',
        '--queue_size', '98304',
        '--momentum', '0.99',
        '--features', *target_features 
    ]
    
    # 執行訓練
    print("--- 1. 正在執行訓練 (train.py) ---")
    subprocess.run(command_train, cwd=project_root)

    # 6. 定義評估命令
    command_eval = [
        python_executable, eval_script,
        '--experiments_main_folder', experiments_main_folder,
        '--experiment_folder', experiment_folder,
        '--id_src', src,
        '--id_trg', trg
    ]
    
    # 執行評估
    print("--- 2. 正在執行評估 (eval.py) ---")
    subprocess.run(command_eval, cwd=project_root)

    # 7. 定義繪圖命令 (原有的 plot.py)
    command_plot = [
        python_executable, plot_script,
        '--experiments_main_folder', experiments_main_folder,
        '--experiment_folder', experiment_folder,
        '--id_src', src,
        '--id_trg', trg
    ]
    
    # 執行繪圖
    print("--- 3. 正在執行繪圖 (plot.py) ---")
    subprocess.run(command_plot, cwd=project_root)

    # 8. 複製 'Ours_msltest_' 檔案
    print(f"--- 4. 正在複製 'Ours_msltest_{src}.csv' 檔案 ---")
    
    results_dir = os.path.join(project_root, experiments_main_folder, experiment_folder, f'{src}-{trg}')
    
    try:
        source_filename = f'Ours_msltest_{src}.csv'
        source_file_path = os.path.join(project_root, source_filename)
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy2(source_file_path, results_dir)
        print(f"成功複製 '{source_filename}' 到 {results_dir}")
        
    except FileNotFoundError:
        print(f"錯誤：找不到來源檔案 {source_file_path}")
    except Exception as e:
        print(f"複製檔案時發生錯誤: {e}")

# ==========================================
    # 9. --- 5. 新增：繪製 y_score 背景標籤圖 (每個 Segment 獨立一張圖) ---
    # ==========================================
    print("--- 5. 正在繪製 y_score 背景標籤圖 (基於最佳閾值著色 - 獨立分圖版) ---")
    
    try:
        from sklearn.metrics import precision_recall_curve
        
        # 定義讀取的 csv 路徑
        predictions_csv = os.path.join(results_dir, 'predictions_test_source.csv')
        
        if os.path.exists(predictions_csv):
            df = pd.read_csv(predictions_csv)
            y_score = df['y_pred']
            y_true = df['y']

            # --- 1. 計算最佳閾值 (Best Threshold) ---
            precision, recall, thresholds = precision_recall_curve(y_true, y_score)
            numerator = 2 * recall * precision
            denominator = recall + precision
            f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
            
            best_idx = np.argmax(f1_scores)
            if best_idx < len(thresholds):
                best_thr = thresholds[best_idx]
            else:
                best_thr = thresholds[-1]
            
            print(f"計算出的最佳閾值 (Best Threshold): {best_thr:.4f}, 最高 F1: {f1_scores[best_idx]:.4f}")

            # --- 2. 定義 Segments (請確認與您的資料順序一致) ---
            segments = [
                {'name': '冷凝盤管20%', 'len': 1801},
                {'name': '蒸發盤管10%', 'len': 1801},
                {'name': '冷媒洩漏20%', 'len': 3601},
                {'name': '壓縮機10%', 'len': 1801},
                {'name': '冷凝風扇10%', 'len': 1801},
                {'name': '蒸發風扇10%', 'len': 1801},
                # 若有加熱器相關，請自行加入
                # {'name': '加熱器運轉', 'len': 1801},
            ]

            # 設定中文字型
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False 

            current_idx = 0 # 用來追蹤全域的索引位置

            # --- 3. 迴圈遍歷每個 Segment 並獨立繪圖 ---
            for i, seg in enumerate(segments):
                seg_name = seg['name']
                seg_len = seg['len']
                
                # 計算該 Segment 在原始資料中的起訖點
                start = current_idx
                end = current_idx + seg_len
                
                # 確保不超出資料範圍
                if start >= len(y_score):
                    print(f"警告：Segment '{seg_name}' 超出資料範圍，停止繪圖。")
                    break
                
                real_end = min(end, len(y_score))
                
                # 切片 (Slice) 取得該區段的數據
                # 注意：reset_index(drop=True) 讓 x 軸從 0 開始，或是保留 index 看全域時間
                # 這裡我們使用全域 index (start ~ real_end) 來畫 x 軸，方便對照原始資料
                seg_y_score = y_score[start:real_end]
                seg_indices = np.arange(start, real_end)
                
                # 開始繪圖
                plt.figure(figsize=(12, 6))
                plt.plot(seg_indices, seg_y_score, label='y_score', color='blue', linewidth=1.5)
                
                # 畫閾值線
                plt.axhline(y=best_thr, color='black', linestyle=':', alpha=0.8, label=f'Threshold: {best_thr:.2f}')
                
                # 背景顏色填充 (針對這個片段)
                pred_labels_seg = (seg_y_score > best_thr).astype(int).values
                n_seg = len(pred_labels_seg)
                
                if n_seg > 0:
                    local_start_idx = 0
                    current_val = pred_labels_seg[0]
                    
                    for k in range(1, n_seg):
                        if pred_labels_seg[k] != current_val:
                            color = 'red' if current_val == 1 else 'green'
                            # 注意 x 座標要加上全域的 start
                            plt.axvspan(start + local_start_idx, start + k, facecolor=color, alpha=0.3)
                            
                            local_start_idx = k
                            current_val = pred_labels_seg[k]
                    
                    # 畫最後一段
                    color = 'red' if current_val == 1 else 'green'
                    plt.axvspan(start + local_start_idx, start + n_seg - 1, facecolor=color, alpha=0.3)

                plt.title(f"Segment {i+1}: {seg_name} (Thr={best_thr:.3f})")
                plt.xlabel('Global Index (Time)')
                plt.ylabel('Score')
                plt.legend(loc='upper right')
                plt.tight_layout()
                
                # 處理檔名 (去除特殊符號避免錯誤)
                safe_name = seg_name.replace('%', 'pct').replace(' ', '_').replace('/', '_')
                output_filename = f'segment_{i+1:02d}_{safe_name}.png'
                output_path = os.path.join(results_dir, output_filename)
                
                plt.savefig(output_path)
                plt.close() # 關閉畫布以釋放記憶體
                
                print(f"  -> 已儲存: {output_filename}")
                
                # 更新下一個區段的起始點
                current_idx += seg_len

        else:
            print(f"找不到檔案: {predictions_csv}，無法繪製圖表。")

    except Exception as e:
        print(f"繪製圖表時發生錯誤: {e}")
        import traceback
        traceback.print_exc()

    print(f"======= HVAC 實驗全部完成 (src: {src}, trg: {trg}) =======")