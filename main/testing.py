import sys
sys.path.insert(0, 'D:\\DACAD')
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils.dataset import get_dataset
from utils.util_progress_log import get_logger, get_dataset_type
from algorithms import get_algorithm
import json
from argparse import ArgumentParser
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import shutil


def calculate_metrics_with_threshold(y_true, y_score, threshold):
    """
    根據給定的閾值計算評估指標
    """
    y_pred = (y_score >= threshold).astype(int)
    
    # 計算 TP, TN, FP, FN
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    # 計算指標
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / len(y_true) if len(y_true) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }


def plot_segments(y_score, y_true, threshold, segments, results_dir, log):
    """
    繪製每個 Segment 的預測結果圖表
    """
    log("--- 正在繪製 Segment 圖表 ---")
    
    # 設定中文字型
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    current_idx = 0
    
    for i, seg in enumerate(segments):
        seg_name = seg['name']
        seg_len = seg['len']
        
        # 計算該 Segment 的起訖點
        start = current_idx
        end = current_idx + seg_len
        
        # 確保不超出資料範圍
        if start >= len(y_score):
            log(f"警告：Segment '{seg_name}' 超出資料範圍，停止繪圖。")
            break
        
        real_end = min(end, len(y_score))
        
        # 切片取得該區段的數據
        seg_y_score = y_score[start:real_end]
        seg_y_true = y_true[start:real_end]
        seg_indices = np.arange(start, real_end)
        
        # 開始繪圖
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # 上圖：預測分數與閾值
        ax1.plot(seg_indices, seg_y_score, label='Prediction Score', color='blue', linewidth=1.5)
        ax1.axhline(y=threshold, color='black', linestyle=':', alpha=0.8, 
                   label=f'Threshold: {threshold:.3f}')
        
        # 背景顏色填充（基於預測）
        pred_labels_seg = (seg_y_score > threshold).astype(int)
        n_seg = len(pred_labels_seg)
        
        if n_seg > 0:
            local_start_idx = 0
            current_val = pred_labels_seg[0]
            
            for k in range(1, n_seg):
                if pred_labels_seg[k] != current_val:
                    color = 'red' if current_val == 1 else 'green'
                    ax1.axvspan(start + local_start_idx, start + k, 
                               facecolor=color, alpha=0.2)
                    local_start_idx = k
                    current_val = pred_labels_seg[k]
            
            # 畫最後一段
            color = 'red' if current_val == 1 else 'green'
            ax1.axvspan(start + local_start_idx, start + n_seg - 1, 
                       facecolor=color, alpha=0.2)
        
        ax1.set_title(f"Segment {i+1}: {seg_name} - Prediction (Threshold={threshold:.3f})")
        ax1.set_xlabel('Time Index')
        ax1.set_ylabel('Score')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 下圖：真實標籤對比
        ax2.plot(seg_indices, seg_y_true, label='True Label', color='orange', 
                linewidth=1.5, marker='o', markersize=2)
        
        # 背景顏色填充（基於真實標籤）
        if n_seg > 0:
            local_start_idx = 0
            current_val = seg_y_true[0]
            
            for k in range(1, n_seg):
                if seg_y_true[k] != current_val:
                    color = 'red' if current_val == 1 else 'green'
                    ax2.axvspan(start + local_start_idx, start + k, 
                               facecolor=color, alpha=0.2)
                    local_start_idx = k
                    current_val = seg_y_true[k]
            
            # 畫最後一段
            color = 'red' if current_val == 1 else 'green'
            ax2.axvspan(start + local_start_idx, start + n_seg - 1, 
                       facecolor=color, alpha=0.2)
        
        ax2.set_title(f"Segment {i+1}: {seg_name} - Ground Truth")
        ax2.set_xlabel('Time Index')
        ax2.set_ylabel('Label (0=Normal, 1=Anomaly)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([-0.1, 1.1])
        
        plt.tight_layout()
        
        # 儲存圖片
        safe_name = seg_name.replace('%', 'pct').replace(' ', '_').replace('/', '_')
        output_filename = f'segment_{i+1:02d}_{safe_name}.png'
        output_path = os.path.join(results_dir, output_filename)
        
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        log(f"  -> 已儲存: {output_filename}")
        
        # 更新下一個區段的起始點
        current_idx += seg_len


def main(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 設定結果資料夾路徑
    experiment_folder_path = os.path.join(
        args.experiments_main_folder, 
        args.experiment_folder,
        f"{args.id_src}-{args.id_trg}"
    )
    
    # 確保結果資料夾存在
    if not os.path.exists(experiment_folder_path):
        print(f"錯誤：找不到實驗資料夾 {experiment_folder_path}")
        print("請確認模型已經訓練完成")
        return
    
    # 配置 logger
    log_filename = f"test_threshold_{args.threshold:.3f}.log"
    log = get_logger(os.path.join(experiment_folder_path, log_filename))
    
    log(f"使用設備: {DEVICE}")
    log(f"載入模型: {experiment_folder_path}")
    log(f"使用閾值: {args.threshold}")
    
    # 載入訓練時的參數
    with open(os.path.join(experiment_folder_path, 'commandline_args.txt'), 'r') as f:
        saved_args_dict = json.load(f)
    
    saved_args = type('SavedArgs', (), saved_args_dict)()
    dataset_type = get_dataset_type(saved_args)
    
    # 1. 載入 Source Train 以取得標準化參數
    log("載入 Source Train 資料集以取得標準化參數...")
    dataset_src_train = get_dataset(saved_args, domain_type="source", split_type="train")
    d_mean, d_std = dataset_src_train.get_statistic()
    log(f"使用的 Mean/Std: mean[0]={d_mean[0]:.4f}, std[0]={d_std[0]:.4f}")
    del dataset_src_train
    
    # 2. 載入測試資料集
    log("載入 Source Test 資料集...")
    dataset_test_src = get_dataset(saved_args, domain_type="source", split_type="test",
                                    d_mean=d_mean, d_std=d_std)
    
    # 3. 取得模型輸入維度
    if len(dataset_test_src) == 0:
        log("錯誤：測試資料集為空")
        return
    
    input_channels_dim = dataset_test_src[0]['sequence'].shape[1]
    input_static_dim = dataset_test_src[0]['static'].shape[0] if 'static' in dataset_test_src[0] else 0
    
    log(f"輸入維度: channels={input_channels_dim}, static={input_static_dim}")
    
    # 4. 建立並載入模型
    log("建立模型...")
    algorithm = get_algorithm(saved_args, 
                             input_channels_dim=input_channels_dim,
                             input_static_dim=input_static_dim, 
                             device=DEVICE)
    
    log("載入模型權重...")
    algorithm.load_state(experiment_folder_path)
    algorithm.eval()
    
    # 5. 建立 DataLoader
    eval_batch_size = saved_args.eval_batch_size
    dataloader_test_src = DataLoader(dataset_test_src, 
                                     batch_size=eval_batch_size,
                                     shuffle=False, 
                                     num_workers=0, 
                                     drop_last=False)
    
    # 6. 進行預測
    log("開始預測...")
    y_pred_list = []
    y_true_list = []
    
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader_test_src):
            # 將數據移到設備
            for key, value in sample_batched.items():
                sample_batched[key] = sample_batched[key].to(device=DEVICE, non_blocking=True)
            
            # 預測
            algorithm.predict_src(sample_batched)
            
            if (i_batch + 1) % 10 == 0:
                log(f"  處理進度: {i_batch + 1}/{len(dataloader_test_src)}")
    
    # 7. 取得預測結果
    y_pred = np.array(algorithm.pred_meter_val_src.output_list)
    y_true = np.array(algorithm.pred_meter_val_src.target_list)
    
    log(f"預測完成，共 {len(y_pred)} 筆資料")
    
    # 8. 使用自定義閾值計算指標
    log(f"\n使用閾值 {args.threshold} 計算指標...")
    metrics = calculate_metrics_with_threshold(y_true, y_pred, args.threshold)
    
    log(f"Precision: {metrics['precision']:.4f}")
    log(f"Recall: {metrics['recall']:.4f}")
    log(f"F1 Score: {metrics['f1']:.4f}")
    log(f"Accuracy: {metrics['accuracy']:.4f}")
    log(f"TP: {metrics['TP']}, TN: {metrics['TN']}, FP: {metrics['FP']}, FN: {metrics['FN']}")
    
    # 9. 計算 AUROC 和 AUPRC
    try:
        auroc = roc_auc_score(y_true, y_pred)
        auprc = average_precision_score(y_true, y_pred)
        log(f"AUROC: {auroc:.4f}")
        log(f"AUPRC: {auprc:.4f}")
    except Exception as e:
        log(f"計算 AUROC/AUPRC 時發生錯誤: {e}")
        auroc = 0.0
        auprc = 0.0
    
    # 10. 儲存 predictions_test_source.csv
    log("\n儲存預測結果...")
    id_test_src = algorithm.pred_meter_val_src.id_patient_list
    stay_hour_src = algorithm.pred_meter_val_src.stay_hours_list
    
    if len(id_test_src) == 0:
        id_test_src = [-1] * len(y_true)
        stay_hour_src = [-1] * len(y_true)
    
    pred_src_df = pd.DataFrame({
        "patient_id": id_test_src,
        "stay_hour": stay_hour_src,
        "y": y_true,
        "y_pred": y_pred
    })
    
    predictions_path = os.path.join(experiment_folder_path, 
                                   f"predictions_test_source_thr_{args.threshold:.3f}.csv")
    pred_src_df.to_csv(predictions_path, index=False)
    log(f"預測結果已儲存至: {predictions_path}")
    
    # 11. 儲存 Ours_msltest_source_data.csv
    metrics_df = pd.DataFrame({
        'src_id': [args.id_src],
        'trg_id': [args.id_trg],
        'threshold': [args.threshold],
        'precision': [metrics['precision']],
        'recall': [metrics['recall']],
        'f1': [metrics['f1']],
        'accuracy': [metrics['accuracy']],
        'auroc': [auroc],
        'auprc': [auprc],
        'TP': [metrics['TP']],
        'TN': [metrics['TN']],
        'FP': [metrics['FP']],
        'FN': [metrics['FN']]
    })
    
    metrics_filename = f'Ours_msltest_{args.id_src}_thr_{args.threshold:.3f}.csv'
    metrics_path = os.path.join(experiment_folder_path, metrics_filename)
    metrics_df.to_csv(metrics_path, index=False)
    log(f"評估指標已儲存至: {metrics_path}")
    
    # 12. 繪製 Segment 圖表
    if args.plot_segments:
        log("\n開始繪製 Segment 圖表...")
        
        # 定義 Segments（根據你的資料調整）
        segments = [
                {'name': '冷凝盤管阻塞30%', 'len': 1800},
                {'name': '蒸發盤管阻塞20%', 'len': 1800},
                {'name': '蒸發風扇電流80%', 'len': 1800},
                {'name': '正常資料高溫42.7度', 'len': 1800},
                {'name': '正常資料低溫24度', 'len': 3600},
                {'name': '冷媒洩漏20%', 'len': 3600},
                {'name': '正常資料常溫27度', 'len': 3600},
                {'name': '正常資料常溫30度', 'len': 3600},
                {'name': '冷凝盤管阻塞重度', 'len': 1800},
                {'name': '壓縮機故障20%', 'len': 5400},
                {'name': '冷凝風扇電流上升20%', 'len': 5400},
                {'name': '蒸發風扇電流上升20%', 'len': 5400},
        ]
        
        try:
            plot_segments(y_pred, y_true, args.threshold, segments, 
                         experiment_folder_path, log)
            log("所有 Segment 圖表繪製完成")
        except Exception as e:
            log(f"繪製 Segment 圖表時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
    
    log(f"\n測試完成！所有結果已儲存至: {experiment_folder_path}")


if __name__ == '__main__':
    parser = ArgumentParser(description="model_best.pth.tar")
    
    # 基本參數
    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='results',
                       help='實驗主資料夾路徑')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='HVAC',
                       help='實驗子資料夾名稱')
    parser.add_argument('--id_src', type=str, default='source_data',
                       help='Source 資料集 ID')
    parser.add_argument('--id_trg', type=str, default='target_data',
                       help='Target 資料集 ID')
    
    # 閾值參數
    parser.add_argument('-t', '--threshold', type=float, default=1.9,
                       help='預測閾值 (0.0 - 1.0)')
    
    # 繪圖參數
    parser.add_argument('--plot_segments', action='store_true', default=True,
                       help='是否繪製 Segment 圖表')
    parser.add_argument('--no_plot_segments', dest='plot_segments', action='store_false',
                       help='不繪製 Segment 圖表')
    
    args = parser.parse_args()
    
    main(args)