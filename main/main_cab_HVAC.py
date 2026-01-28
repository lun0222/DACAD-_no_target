import os
import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
from matplotlib.patches import Patch
import time 
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import shutil 

if __name__ == '__main__':
    # ===================================================================
    # 第一部分：資料分割 (原 cab_data_select.py 的功能)
    # ===================================================================
    print("=" * 60)
    print("步驟 1: 資料分割與前處理")
    print("=" * 60)
    
    # 獲取目錄
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # CSV 檔案路徑
    csv_file_name = 'D:\\DACAD-_no_target\\駕駛室資料\\cab_data_final_train.csv'
    time_column_name = 'datetime'
    
    # 輸出資料夾
    output_directory = os.path.join(project_root, 'datasets', 'HVAC')
    
    # 定義訓練資料時間段
    source_time_periods = [
        ('2025-04-11 09:14:01', '2025-04-11 09:34:00', 0), #冷凝盤管阻塞20% 訓練
        ('2025-04-11 09:46:01', '2025-04-11 10:06:00', 0), #冷凝盤管阻塞30% 訓練
        ('2025-04-11 10:29:01', '2025-04-11 10:49:00', 0), #蒸發盤管阻塞10% 訓練
        ('2025-04-11 11:01:01', '2025-04-11 11:21:00', 0), #蒸發盤管阻塞20% 訓練
        ('2025-04-11 11:33:01', '2025-04-11 11:53:00', 0), #蒸發盤管阻塞23% 訓練
        ('2025-04-11 12:05:01', '2025-04-11 13:05:00', 0), #正常資料常溫34度 訓練
        ('2025-04-11 13:45:01', '2025-04-11 14:05:00', 0), #蒸發風扇電流90% 訓練
        ('2025-04-11 14:19:01', '2025-04-11 14:39:00', 0), #蒸發風扇電流80% 訓練
        ('2025-04-11 14:52:01', '2025-04-11 15:12:00', 0), #蒸發風扇電流70% 訓練
        ('2025-04-11 15:46:01', '2025-04-11 16:06:00', 0), #正常資料高溫42.7度 訓練
        ('2025-04-14 09:00:01', '2025-04-14 09:40:00', 0), #正常資料低溫24度 訓練
        ('2025-04-14 13:25:01', '2025-04-14 14:05:00', 1), #冷媒洩漏10% 訓練 
        ('2025-04-14 14:50:01', '2025-04-14 15:30:00', 1), #冷媒洩漏20% 訓練
        ('2025-12-23 10:23:01', '2025-12-23 11:03:00', 0), #正常資料低溫27度 訓練
        ('2025-12-23 12:15:01', '2025-12-23 12:55:00', 0), #正常資料常溫30度 訓練
        ('2025-12-23 13:19:01', '2025-12-23 13:40:00', 0), #輕度(兩層洗衣袋)冷凝盤管阻塞 訓練
        ('2025-12-23 14:03:01', '2025-12-23 14:23:00', 0), #重度(四層洗衣袋)冷凝盤管阻塞 訓練
        ('2025-12-23 15:00:01', '2025-12-23 15:40:00', 1), #冷媒洩漏30% 訓練
        ('2026-01-01 00:00:01', '2026-01-01 01:00:00', 0), #壓縮機故障10% 訓練
        ('2026-01-01 01:30:01', '2026-01-01 02:30:00', 0), #壓縮機故障20% 訓練
        ('2026-01-01 03:00:01', '2026-01-01 04:00:00', 0), #壓縮機故障30% 訓練
        ('2026-01-01 05:00:01', '2026-01-01 06:00:00', 0), #冷凝風扇電流上升10% 訓練
        ('2026-01-01 06:30:01', '2026-01-01 07:30:00', 0), #冷凝風扇電流上升20% 訓練
        ('2026-01-01 08:00:01', '2026-01-01 09:00:00', 0), #冷凝風扇電流上升30% 訓練
        ('2026-01-01 10:00:01', '2026-01-01 11:00:00', 0), #蒸發風扇電流上升10% 訓練
        ('2026-01-01 11:30:01', '2026-01-01 12:30:00', 0), #蒸發風扇電流上升20% 訓練 
        ('2026-01-01 13:00:01', '2026-01-01 14:00:00', 0), #蒸發風扇電流上升30% 訓練
    ]

    # 定義測試資料時間段 (每個獨立測試)
    test_time_periods = [
        ('2025-04-11 09:34:01', '2025-04-11 09:44:00', 0, '冷凝盤管阻塞20%'),
        ('2025-04-11 10:06:01', '2025-04-11 10:16:00', 0, '冷凝盤管阻塞30%'),
        ('2025-04-11 10:49:01', '2025-04-11 10:59:00', 0, '蒸發盤管阻塞10%'),
        ('2025-04-11 11:21:01', '2025-04-11 11:31:00', 0, '蒸發盤管阻塞20%'),
        ('2025-04-11 11:53:01', '2025-04-11 12:03:00', 0, '蒸發盤管阻塞23%'),
        ('2025-04-11 13:05:01', '2025-04-11 13:35:00', 0, '正常資料常溫34度'),
        ('2025-04-11 14:05:01', '2025-04-11 14:15:00', 0, '蒸發風扇電流90%'),
        ('2025-04-11 14:39:01', '2025-04-11 14:49:00', 0, '蒸發風扇電流80%'),
        ('2025-04-11 15:12:01', '2025-04-11 15:22:00', 0, '蒸發風扇電流70%'),
        ('2025-04-11 16:06:01', '2025-04-11 16:16:00', 0, '正常資料高溫42.7度'),
        ('2025-04-14 09:40:01', '2025-04-14 10:00:00', 0, '正常資料低溫24度'),
        ('2025-04-14 14:05:01', '2025-04-14 14:25:00', 1, '冷媒洩漏10%'),
        ('2025-04-14 15:30:01', '2025-04-14 15:50:00', 1, '冷媒洩漏20%'),
        ('2025-12-23 11:03:01', '2025-12-23 11:23:00', 0, '正常資料低溫27度'),
        ('2025-12-23 12:55:01', '2025-12-23 13:15:00', 0, '正常資料常溫30度'),
        ('2025-12-23 13:40:01', '2025-12-23 13:57:00', 0, '輕度冷凝盤管阻塞'),
        ('2025-12-23 14:23:01', '2025-12-23 14:33:00', 0, '重度冷凝盤管阻塞'),
        ('2025-12-23 15:40:01', '2025-12-23 16:00:00', 1, '冷媒洩漏30%'),
        ('2026-01-01 01:00:01', '2026-01-01 01:30:00', 0, '壓縮機故障10%'),
        ('2026-01-01 02:30:01', '2026-01-01 03:00:00', 0, '壓縮機故障20%'),
        ('2026-01-01 04:00:01', '2026-01-01 04:30:00', 0, '壓縮機故障30%'),
        ('2026-01-01 06:00:01', '2026-01-01 06:30:00', 0, '冷凝風扇電流上升10%'),
        ('2026-01-01 07:30:01', '2026-01-01 08:00:00', 0, '冷凝風扇電流上升20%'),
        ('2026-01-01 09:00:01', '2026-01-01 09:30:00', 0, '冷凝風扇電流上升30%'),
        ('2026-01-01 11:00:01', '2026-01-01 11:30:00', 0, '蒸發風扇電流上升10%'),
        ('2026-01-01 12:30:01', '2026-01-01 13:00:00', 0, '蒸發風扇電流上升20%'),
        ('2026-01-01 14:00:01', '2026-01-01 14:30:00', 0, '蒸發風扇電流上升30%'),
    ]
    
    # 讀取並處理 CSV
    print(f"\n正在讀取 CSV 檔案: {csv_file_name}")
    try:
        df = pd.read_csv(csv_file_name)
        df[time_column_name] = pd.to_datetime(df[time_column_name], errors='coerce')
        df.dropna(subset=[time_column_name], inplace=True)
        df.sort_values(by=time_column_name, inplace=True)
        print(f"成功讀取 {len(df)} 筆資料")
    except Exception as e:
        print(f"讀取 CSV 檔案時發生錯誤: {e}")
        sys.exit(1)
    
    # 建立輸出資料夾
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"已建立資料夾：{output_directory}")
    
    # 處理訓練資料
    def process_and_save_periods(dataframe, periods_list, filename, output_dir):
        if not periods_list:
            print(f"\n列表 {filename} 為空，跳過儲存。")
            return
        
        labeled_dfs = []
        for period in periods_list:
            start_str, end_str, label = period[0], period[1], period[2]
            start_time = pd.to_datetime(start_str)
            end_time = pd.to_datetime(end_str)
            
            mask = (dataframe[time_column_name] >= start_time) & (dataframe[time_column_name] <= end_time)
            period_df = dataframe[mask].copy()
            
            if not period_df.empty:
                period_df['label'] = label
                labeled_dfs.append(period_df)
        
        if not labeled_dfs:
            print(f"\n在 {filename} 的時間段中沒有找到任何資料。")
            return
        
        final_df = pd.concat(labeled_dfs).sort_values(by=time_column_name)
        output_path = os.path.join(output_dir, filename)
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"檔案已成功儲存至 {output_path} (共 {len(final_df)} 筆資料)")
    
    print("\n正在處理並儲存 Source Training Data...")
    process_and_save_periods(df, source_time_periods, 'source_data_train.csv', output_directory)
    
    # 為每個測試時間段建立獨立的 CSV 檔案
    print("\n正在為每個測試時間段建立獨立的測試檔案...")
    test_files_info = []
    for idx, (start_str, end_str, label, name) in enumerate(test_time_periods):
        start_time = pd.to_datetime(start_str)
        end_time = pd.to_datetime(end_str)
        
        mask = (df[time_column_name] >= start_time) & (df[time_column_name] <= end_time)
        period_df = df[mask].copy()
        
        if not period_df.empty:
            period_df['label'] = label
            
            # 檔名使用索引編號
            test_filename = f'test_data_{idx:02d}.csv'
            output_path = os.path.join(output_directory, test_filename)
            period_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            test_files_info.append({
                'idx': idx,
                'filename': test_filename,
                'name': name,
                'label': label,
                'count': len(period_df)
            })
            print(f"  [{idx:02d}] {name}: {len(period_df)} 筆資料 -> {test_filename}")
    
    print(f"\n資料分割完成！共建立 {len(test_files_info)} 個測試檔案")
    
    # ===================================================================
    # 第二部分：模型訓練
    # ===================================================================
    print("\n" + "=" * 60)
    print("步驟 2: 模型訓練")
    print("=" * 60)
    
    dataset_path = output_directory
    train_script = os.path.join(current_dir, 'train.py')
    python_executable = sys.executable
    
    # 設定特徵
    target_features = [
    # 冷凝盤管阻塞 (Condenser Coil Fault)
    # 'hp_comp_1','lp_comp_1','cond_current_1','comp_current_1','return_air_temp' ,'outdoor_temp'

    # 蒸發盤管阻塞 (Evaporator Coil Fault)
    # 'fan_current_1','hp_comp_1','lp_comp_1','comp_current_1','return_air_temp','outdoor_temp','EF1speed'

    # 冷媒洩漏 (Refrigerant Leak Fault)
    'hp_comp_1','lp_comp_1','superheat_1','lp_plate_temp_1','comp_current_1','h_suc_1','return_air_temp','outdoor_temp'

    # 壓縮機故障 (Compressor Fault)
    # 'hp_comp_1','lp_comp_1','comp_current_1','cond_current_1','return_air_temp','outdoor_temp'

    # 冷凝風扇故障
    # 'hp_comp_1','cond_current_1','comp_current_1','return_air_temp','outdoor_temp'

    # 蒸發風扇故障
    # 'fan_current_1','hp_comp_1','lp_comp_1','comp_current_1','return_air_temp','outdoor_temp'

    #加熱器
    # 'heater_temp','return_air_temp','outdoor_temp'
    ]
    
    src = "source_data"
    trg = "target_data"
    
    experiments_main_folder = 'results'
    experiment_folder = 'HVAC'
    
    # 訓練命令
    command_train = [
        python_executable, train_script,
        '--algo_name', 'dacad',
        '--experiment_folder', experiment_folder,
        '--path_src', dataset_path,
        '--path_trg', dataset_path,
        '--id_src', src,
        '--id_trg', trg,
        '--num_epochs', '10',
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
    
    print("\n正在執行訓練...")
    subprocess.run(command_train, cwd=project_root)
    
    # ===================================================================
    # 第三部分：獨立測試每個時間段
    # ===================================================================
    print("\n" + "=" * 60)
    print("步驟 3: 獨立測試每個時間段")
    print("=" * 60)
    
    results_dir = os.path.join(project_root, experiments_main_folder, experiment_folder, f'{src}-{trg}')
    
    # 匯入必要的模組
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, project_root)
    from utils.dataset import get_dataset
    from utils.util_progress_log import get_logger, get_dataset_type
    from algorithms import get_algorithm
    import json
    from collections import namedtuple
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 讀取訓練時的參數
    with open(os.path.join(results_dir, 'commandline_args.txt'), 'r') as f:
        saved_args_dict = json.load(f)
    saved_args = namedtuple("SavedArgs", saved_args_dict.keys())(*saved_args_dict.values())
    
    # 載入 Source Train 的標準化參數
    dataset_src_train = get_dataset(saved_args, domain_type="source", split_type="train")
    d_mean, d_std = dataset_src_train.get_statistic()
    print(f"\n使用 Source Train 的 Mean/Std (mean[0]={d_mean[0]:.4f}, std[0]={d_std[0]:.4f})")
    
    input_channels_dim = dataset_src_train[0]['sequence'].shape[1]
    input_static_dim = dataset_src_train[0]['static'].shape[0] if 'static' in dataset_src_train[0] else 0
    del dataset_src_train
    
    # 初始化模型
    algorithm = get_algorithm(saved_args, input_channels_dim=input_channels_dim, 
                             input_static_dim=input_static_dim, device=DEVICE)
    algorithm.load_state(results_dir)
    algorithm.eval()
    print("模型載入完成\n")
    
    # 儲存所有測試結果
    all_test_results = []
    
    # 儲存所有預測結果（用於計算整體分數）
    all_predictions = []
    all_true_labels = []
    
    # 對每個測試檔案進行獨立測試
    for test_info in test_files_info:
        idx = test_info['idx']
        filename = test_info['filename']
        name = test_info['name']
        true_label = test_info['label']
        
        print(f"\n{'='*60}")
        print(f"測試 [{idx:02d}] {name}")
        print(f"{'='*60}")
        
        # 建立臨時測試資料集
        temp_test_path = os.path.join(output_directory, 'test_data.csv')
        source_test_path = os.path.join(output_directory, filename)
        
        # 複製測試檔案
        shutil.copy2(source_test_path, temp_test_path)
        
        try:
            # 載入測試資料
            dataset_test = get_dataset(saved_args, domain_type="source", split_type="test",
                                      d_mean=d_mean, d_std=d_std)
            
            if len(dataset_test) == 0:
                print(f"  警告：{name} 沒有產生有效的測試視窗，跳過")
                os.remove(temp_test_path)
                continue
            
            # 直接從 dataset 取得原始資料進行預測（避免 __getitem__ 的 positive/negative 問題）
            all_sequences = []
            all_labels = []
            
            for i in range(len(dataset_test)):
                # 直接讀取序列和標籤
                seq = dataset_test.sequence[i]
                lbl = dataset_test.label[i]
                all_sequences.append(seq)
                all_labels.append(lbl)
            
            # 轉換為 tensor
            all_sequences = torch.FloatTensor(np.array(all_sequences)).to(DEVICE)
            all_labels = torch.LongTensor(np.array(all_labels)).to(DEVICE)
            
            # 批次預測
            algorithm.pred_meter_val_src.target_list = []
            algorithm.pred_meter_val_src.output_list = []
            
            batch_size = saved_args.eval_batch_size
            num_batches = (len(all_sequences) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(all_sequences))
                
                batch_seq = all_sequences[start_idx:end_idx]
                batch_lbl = all_labels[start_idx:end_idx]
                
                # 建立 sequence_mask (全 1，表示所有資料都有效)
                batch_mask = torch.ones_like(batch_seq).long().to(DEVICE)
                
                # 建立 sample_batched 字典
                sample_batched = {
                    'sequence': batch_seq,
                    'sequence_mask': batch_mask,
                    'label': batch_lbl.unsqueeze(1),
                    'static': torch.zeros(len(batch_seq), input_static_dim).to(DEVICE)  # 假設沒有靜態特徵
                }
                
                with torch.no_grad():
                    algorithm.predict_src(sample_batched)
            
            # 取得預測結果
            y_pred = np.array(algorithm.pred_meter_val_src.output_list)
            y_true = np.array(algorithm.pred_meter_val_src.target_list)
            
            # 儲存到全域列表（用於計算整體分數）
            all_predictions.extend(y_pred)
            all_true_labels.extend(y_true)
            
            # 顯示結果
            print(f"  資料筆數: {len(y_true)}")
            print(f"  真實標籤: {true_label} ({'異常' if true_label == 1 else '正常'})")
            
            # 儲存當前測試段的結果（不計算獨立指標）
            all_test_results.append({
                'idx': idx,
                'name': name,
                'true_label': true_label,
                'sample_count': len(y_true),
                'y_pred': y_pred,
                'y_true': y_true
            })
            
        finally:
            # 清理臨時檔案
            if os.path.exists(temp_test_path):
                os.remove(temp_test_path)
    
    # ===================================================================
    # 計算整體評估指標（所有測試段合併）
    # ===================================================================
    print("\n" + "=" * 60)
    print("整體評估結果（所有測試段合併）")
    print("=" * 60)
    
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    best_thr = 2.0
    print(f"使用固定閾值 (Fixed Threshold): {best_thr:.4f}")
    
    # 計算整體指標
    try:
        overall_auroc = roc_auc_score(all_true_labels, all_predictions)
        overall_auprc = average_precision_score(all_true_labels, all_predictions)
    except:
        overall_auroc = 0.0
        overall_auprc = 0.0
    
    # 基於最佳閾值計算混淆矩陣
    all_pred_binary = (all_predictions >= best_thr).astype(int)
    overall_TP = np.sum((all_true_labels == 1) & (all_pred_binary == 1))
    overall_TN = np.sum((all_true_labels == 0) & (all_pred_binary == 0))
    overall_FP = np.sum((all_true_labels == 0) & (all_pred_binary == 1))
    overall_FN = np.sum((all_true_labels == 1) & (all_pred_binary == 0))
    
    overall_precision = overall_TP / (overall_TP + overall_FP) if (overall_TP + overall_FP) > 0 else 0
    overall_recall = overall_TP / (overall_TP + overall_FN) if (overall_TP + overall_FN) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    overall_accuracy = (overall_TP + overall_TN) / len(all_true_labels) if len(all_true_labels) > 0 else 0
    
    # 顯示整體結果
    print(f"\n總樣本數: {len(all_true_labels)}")
    print(f"最佳閾值: {best_thr:.4f}")
    print(f"AUROC: {overall_auroc:.4f}")
    print(f"AUPRC: {overall_auprc:.4f}")
    print(f"Precision: {overall_precision:.4f}")
    print(f"Recall: {overall_recall:.4f}")
    print(f"F1 Score: {overall_f1:.4f}")
    print(f"Accuracy: {overall_accuracy:.4f}")
    print(f"TP={overall_TP}, TN={overall_TN}, FP={overall_FP}, FN={overall_FN}")
    
    # 儲存整體結果
    overall_results = pd.DataFrame([{
        'src_id': src,
        'trg_id': trg,
        'total_samples': len(all_true_labels),
        'best_threshold': best_thr,
        'auroc': overall_auroc,
        'auprc': overall_auprc,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'accuracy': overall_accuracy,
        'TP': overall_TP,
        'TN': overall_TN,
        'FP': overall_FP,
        'FN': overall_FN
    }])
    
    overall_csv_path = os.path.join(results_dir, f'Ours_msltest_{src}.csv')
    overall_results.to_csv(overall_csv_path, index=False)
    print(f"\n整體結果已儲存至: {overall_csv_path}")
    
    # ===================================================================
    # 繪製每個測試段的圖表
    # ===================================================================
    print("\n" + "=" * 60)
    print("繪製各測試段的預測圖表")
    print("=" * 60)
    
    # 設定中文字型
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    current_idx = 0
    
    for result in all_test_results:
        idx = result['idx']
        name = result['name']
        y_pred = result['y_pred']
        y_true = result['y_true']
        
        # 建立索引
        indices = np.arange(current_idx, current_idx + len(y_pred))
        
        # 繪圖
        plt.figure(figsize=(12, 6))
        plt.plot(indices, y_pred, label='Prediction Score', color='blue', linewidth=1.5)
        
        # 畫閾值線
        plt.axhline(y=best_thr, color='black', linestyle=':', alpha=0.8, 
                   label=f'Threshold: {best_thr:.3f}')
        
        # 背景顏色填充（基於預測）
        pred_labels = (y_pred > best_thr).astype(int)
        
        if len(pred_labels) > 0:
            local_start_idx = 0
            current_val = pred_labels[0]
            
            for k in range(1, len(pred_labels)):
                if pred_labels[k] != current_val:
                    color = 'red' if current_val == 1 else 'green'
                    plt.axvspan(current_idx + local_start_idx, current_idx + k, 
                               facecolor=color, alpha=0.3)
                    local_start_idx = k
                    current_val = pred_labels[k]
            
            # 畫最後一段
            color = 'red' if current_val == 1 else 'green'
            plt.axvspan(current_idx + local_start_idx, current_idx + len(pred_labels) - 1, 
                       facecolor=color, alpha=0.3)
        
        plt.title(f"Segment {idx+1:02d}: {name} (Threshold={best_thr:.3f})")
        plt.xlabel('Global Index (Time)')
        plt.ylabel('Score')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 儲存圖片
        safe_name = name.replace('%', 'pct').replace(' ', '_').replace('/', '_')
        output_filename = f'segment_{idx+1:02d}_{safe_name}.png'
        output_path = os.path.join(results_dir, output_filename)
        
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"  [{idx+1:02d}] {name} -> {output_filename}")
        
        # 更新索引
        current_idx += len(y_pred)
    
    print(f"\n所有圖表已儲存至: {results_dir}")
    
    # 儲存各測試段的詳細資訊（不含評估指標，因為是整體計算）
    segments_info = []
    for result in all_test_results:
        segments_info.append({
            'idx': result['idx'],
            'name': result['name'],
            'true_label': result['true_label'],
            'sample_count': result['sample_count']
        })
    
    segments_df = pd.DataFrame(segments_info)
    results_csv_path = os.path.join(results_dir, 'test_segments_info.csv')
    segments_df.to_csv(results_csv_path, index=False)
    print(f"測試段資訊已儲存至: {results_csv_path}")
    
    # ===================================================================
    # 第四部分：繪製損失曲線
    # ===================================================================
    print("\n" + "=" * 60)
    print("步驟 4: 繪製訓練損失曲線")
    print("=" * 60)
    
    plot_script = os.path.join(current_dir, 'plot.py')
    command_plot = [
        python_executable, plot_script,
        '--experiments_main_folder', experiments_main_folder,
        '--experiment_folder', experiment_folder,
        '--id_src', src,
        '--id_trg', trg
    ]
    subprocess.run(command_plot, cwd=project_root)
    
    print("\n" + "=" * 60)
    print("所有步驟完成！")
    print("=" * 60)
    print(f"\n結果儲存位置: {results_dir}")
    print(f"- 整體評估結果: Ours_msltest_{src}.csv")
    print(f"- 測試段資訊: test_segments_info.csv")
    print(f"- 各段預測圖表: segment_XX_*.png")