import sys
sys.path.insert(0, 'D:\\DACAD-_no_target')
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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


def main(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 設定結果資料夾路徑
    experiment_folder_path = os.path.join(
        args.experiments_main_folder, 
        args.experiment_folder,
        f"{args.id_src}-{args.id_trg}"
    )
    
    # 配置 logger
    log_filename = "test_segments.log"
    log = get_logger(os.path.join(experiment_folder_path, log_filename))
    
    log("=" * 60)
    log("步驟 1: 載入模型和訓練資料統計")
    log("=" * 60)
    
    # 載入訓練時的參數
    with open(os.path.join(experiment_folder_path, 'commandline_args.txt'), 'r') as f:
        saved_args_dict = json.load(f)
    
    saved_args = type('SavedArgs', (), saved_args_dict)()
    dataset_type = get_dataset_type(saved_args)
    
    # 載入 Source Train 以取得標準化參數
    log("載入 Source Train 資料集以取得標準化參數...")
    dataset_src_train = get_dataset(saved_args, domain_type="source", split_type="train")
    d_mean, d_std = dataset_src_train.get_statistic()
    log(f"使用 Source Train 的 Mean/Std (mean[0]={d_mean[0]:.4f}, std[0]={d_std[0]:.4f})")
    del dataset_src_train
    
    # 載入測試資料集
    log("載入 Source Test 資料集...")
    dataset_test_src = get_dataset(saved_args, domain_type="source", split_type="test",
                                    d_mean=d_mean, d_std=d_std)
    
    # 取得模型輸入維度
    input_channels_dim = dataset_test_src[0]['sequence'].shape[1]
    input_static_dim = dataset_test_src[0]['static'].shape[0] if 'static' in dataset_test_src[0] else 0
    
    # 建立並載入模型
    log("建立模型...")
    algorithm = get_algorithm(saved_args, 
                             input_channels_dim=input_channels_dim,
                             input_static_dim=input_static_dim, 
                             device=DEVICE)
    
    log("載入模型權重...")
    algorithm.load_state(experiment_folder_path)
    algorithm.eval()
    log("模型載入完成")
    log("")
    
    # ============================================================
    # 步驟 2: 收集 t-SNE 所需的 embeddings
    # ============================================================
    log("=" * 60)
    log("步驟 2: 收集 embeddings 用於 t-SNE 視覺化")
    log("=" * 60)
    
    eval_batch_size = saved_args.eval_batch_size
    dataloader_test = DataLoader(dataset_test_src, batch_size=eval_batch_size,
                                 shuffle=False, num_workers=0, drop_last=False)
    
    all_embeddings = []
    all_labels = []
    all_predictions = []
    
    log(f"開始收集 embeddings (共 {len(dataloader_test)} 個批次)...")
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader_test):
            # 將數據移到設備
            for key, value in sample_batched.items():
                sample_batched[key] = sample_batched[key].to(device=DEVICE, non_blocking=True)
            
            # 取得 embeddings
            embeddings = algorithm.get_embedding(sample_batched)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(sample_batched['label'].cpu().numpy())
            
            # 同時進行預測
            algorithm.predict_src(sample_batched)
            
            if (i_batch + 1) % 10 == 0:
                log(f"  已處理 {i_batch + 1}/{len(dataloader_test)} 批次")
    
    # 合併所有結果
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    
    log(f"收集完成!")
    log(f"  Embeddings 形狀: {final_embeddings.shape}")
    log(f"  標籤形狀: {final_labels.shape}")
    log(f"  正常樣本: {np.sum(final_labels==0)}")
    log(f"  異常樣本: {np.sum(final_labels==1)}")
    log("")
    
    # ============================================================
    # 步驟 3: 生成 t-SNE 視覺化
    # ============================================================
    log("=" * 60)
    log("步驟 3: 執行 t-SNE 降維並繪圖")
    log("=" * 60)
    
    try:
        # 設定中文字型
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 採樣策略選擇
        USE_FULL_DATA = True  # 設為 True 使用全部資料，False 進行採樣
        
        if USE_FULL_DATA:
            log(f"使用全部 {len(final_embeddings)} 個樣本（不採樣）")
            tsne_embeddings = final_embeddings
            tsne_labels = final_labels
        else:
            # 分層採樣（保持正常/異常比例）
            max_samples = 5000
            if len(final_embeddings) > max_samples:
                log(f"使用分層採樣：從 {len(final_embeddings)} 個樣本中選取 {max_samples} 個...")
                
                from sklearn.model_selection import train_test_split
                indices = np.arange(len(final_embeddings))
                sampled_indices, _ = train_test_split(
                    indices,
                    train_size=max_samples,
                    stratify=final_labels.flatten(),
                    random_state=42  # 固定種子確保可重現
                )
                
                tsne_embeddings = final_embeddings[sampled_indices]
                tsne_labels = final_labels[sampled_indices]
                log(f"  採樣後: {len(tsne_embeddings)} 個樣本")
                log(f"  正常: {np.sum(tsne_labels.flatten()==0)}, 異常: {np.sum(tsne_labels.flatten()==1)}")
            else:
                tsne_embeddings = final_embeddings
                tsne_labels = final_labels
        
        # 執行 t-SNE
        log("正在執行 t-SNE 降維 (這可能需要幾分鐘)...")
        
        # 根據資料量調整參數
        n_samples = len(tsne_embeddings)
        perplexity_value = min(50, max(5, n_samples // 100))  # 動態調整 perplexity
        
        log(f"  樣本數: {n_samples}")
        log(f"  Perplexity: {perplexity_value}")
        log(f"  Max iterations: 500")
        
        tsne = TSNE(
            n_components=2, 
            verbose=1, 
            perplexity=perplexity_value,
            max_iter=500,  # 增加迭代次數以獲得更好的結果
            random_state=42,
            n_jobs=-1  # 使用所有 CPU 核心加速
        )
        embeddings_2d = tsne.fit_transform(tsne_embeddings)
        
        log(f"t-SNE 完成! 2D embeddings 形狀: {embeddings_2d.shape}")
        
        # 建立 DataFrame 用於繪圖
        # 確保 tsne_labels 是一維陣列
        tsne_labels_1d = tsne_labels.flatten()
        
        df_tsne = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'label': tsne_labels_1d
        })
        df_tsne['label_str'] = df_tsne['label'].map({0: '正常 (Normal)', 1: '異常 (Abnormal)'})
        
        # 繪製 t-SNE 圖
        log("繪製 t-SNE 視覺化圖表...")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 使用 seaborn 繪製散點圖
        sns.scatterplot(
            data=df_tsne,
            x='x',
            y='y',
            hue='label_str',
            palette={'正常 (Normal)': '#2ecc71', '異常 (Abnormal)': '#e74c3c'},
            s=60,
            alpha=0.6,
            edgecolor='black',
            linewidth=0.3,
            ax=ax
        )
        
        # 設定標題和標籤
        ax.set_title('t-SNE 視覺化: 模型嵌入空間 (Source Test Data)', 
                     fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('t-SNE 維度 1', fontsize=14)
        ax.set_ylabel('t-SNE 維度 2', fontsize=14)
        
        # 改善圖例
        ax.legend(title='標籤類別', title_fontsize=12, fontsize=11, 
                 loc='best', framealpha=0.9)
        
        # 加入網格
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # 加入統計資訊
        normal_count = np.sum(tsne_labels_1d == 0)
        abnormal_count = np.sum(tsne_labels_1d == 1)
        info_text = f'樣本數: {len(tsne_labels_1d)}\n正常: {normal_count}\n異常: {abnormal_count}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 儲存圖片
        tsne_save_path = os.path.join(experiment_folder_path, "tsne_visualization.png")
        plt.savefig(tsne_save_path, dpi=300, bbox_inches='tight')
        log(f"t-SNE 圖表已儲存至: {tsne_save_path}")
        plt.close()
        
        # 額外：繪製不同視角的 t-SNE
        log("繪製額外的 t-SNE 分析圖...")
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # 確保 tsne_labels_1d 是一維的
        tsne_labels_1d = tsne_labels.flatten()
        
        # 左圖：按標籤著色
        for label, color, name in [(0, '#2ecc71', '正常'), (1, '#e74c3c', '異常')]:
            mask = tsne_labels_1d == label
            axes[0].scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1],
                c=color, 
                label=name, 
                s=50, 
                alpha=0.6, 
                edgecolor='black', 
                linewidth=0.2
            )
        axes[0].set_title('按標籤分類', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('t-SNE 維度 1', fontsize=12)
        axes[0].set_ylabel('t-SNE 維度 2', fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # 右圖：密度圖
        from scipy.stats import gaussian_kde
        
        # 計算密度
        xy = embeddings_2d.T
        z = gaussian_kde(xy)(xy)
        
        scatter = axes[1].scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1],
            c=z, 
            s=50, 
            cmap='viridis', 
            alpha=0.6, 
            edgecolor='black', 
            linewidth=0.2
        )
        axes[1].set_title('密度熱圖', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('t-SNE 維度 1', fontsize=12)
        axes[1].set_ylabel('t-SNE 維度 2', fontsize=12)
        plt.colorbar(scatter, ax=axes[1], label='密度')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 儲存多視角圖
        tsne_multi_path = os.path.join(experiment_folder_path, "tsne_multi_view.png")
        plt.savefig(tsne_multi_path, dpi=300, bbox_inches='tight')
        log(f"多視角 t-SNE 圖表已儲存至: {tsne_multi_path}")
        plt.close()
        
        # 儲存 t-SNE 座標到 CSV
        tsne_csv_path = os.path.join(experiment_folder_path, "tsne_coordinates.csv")
        df_tsne.to_csv(tsne_csv_path, index=False)
        log(f"t-SNE 座標已儲存至: {tsne_csv_path}")
        
        log("t-SNE 視覺化完成!")
        
    except Exception as e:
        log(f"錯誤：t-SNE 視覺化失敗: {e}")
        import traceback
        log(traceback.format_exc())
    
    log("")
    
    # ============================================================
    # 步驟 4: 定義測試段並進行測試
    # ============================================================
    log("=" * 60)
    log("步驟 4: 獨立測試每個時間段")
    log("=" * 60)
    
    # 定義 segments (根據您的資料調整)
    segments = [
        {'name': '冷凝盤管阻塞20%', 'len': 600, 'expected_label': 0},
        {'name': '冷凝盤管阻塞30%', 'len': 600, 'expected_label': 0},
        {'name': '蒸發盤管阻塞10%', 'len': 600, 'expected_label': 0},
        {'name': '蒸發盤管阻塞20%', 'len': 600, 'expected_label': 0},
        {'name': '蒸發盤管阻塞23%', 'len': 600, 'expected_label': 0},
        {'name': '正常資料常溫34度', 'len': 1800, 'expected_label': 0},
        {'name': '蒸發風扇電流90%', 'len': 600, 'expected_label': 0},
        {'name': '蒸發風扇電流80%', 'len': 600, 'expected_label': 0},
        {'name': '蒸發風扇電流70%', 'len': 600, 'expected_label': 0},
        {'name': '正常資料高溫42.7度', 'len': 600, 'expected_label': 0},
        {'name': '正常資料低溫24度', 'len': 1200, 'expected_label': 0},
        {'name': '冷媒洩漏10%', 'len': 1200, 'expected_label': 0},
        {'name': '冷媒洩漏20%', 'len': 1200, 'expected_label': 0},
        {'name': '正常資料低溫27度', 'len': 1200, 'expected_label': 0},
        {'name': '正常資料常溫30度', 'len': 1200, 'expected_label': 0},
        {'name': '輕度冷凝盤管阻塞', 'len': 1020, 'expected_label': 0},
        {'name': '重度冷凝盤管阻塞', 'len': 600, 'expected_label': 0},
        {'name': '冷媒洩漏30%', 'len': 1200, 'expected_label': 0},
        {'name': '壓縮機故障10%', 'len': 1800, 'expected_label': 0},
        {'name': '壓縮機故障20%', 'len': 1800, 'expected_label': 0},
        {'name': '壓縮機故障30%', 'len': 1800, 'expected_label': 0},
        {'name': '冷凝風扇電流上升10%', 'len': 1800, 'expected_label': 0},
        {'name': '冷凝風扇電流上升20%', 'len': 1800, 'expected_label': 0},
        {'name': '冷凝風扇電流上升30%', 'len': 1800, 'expected_label': 0},
        {'name': '蒸發風扇電流上升10%', 'len': 1800, 'expected_label': 1},
        {'name': '蒸發風扇電流上升20%', 'len': 1800, 'expected_label': 1},
        {'name': '蒸發風扇電流上升30%', 'len': 1800, 'expected_label': 1},
    ]
    
    # 取得預測結果
    y_pred = np.array(algorithm.pred_meter_val_src.output_list)
    y_true = np.array(algorithm.pred_meter_val_src.target_list)
    
    # 固定閾值
    FIXED_THRESHOLD = 2.0
    
    # 測試每個 segment
    segment_results = []
    current_idx = 0
    
    for i, seg in enumerate(segments):
        log("")
        log("=" * 60)
        log(f"測試 [{i:02d}] {seg['name']}")
        log("=" * 60)
        
        # 計算該 segment 的範圍
        start = current_idx
        end = min(current_idx + seg['len'], len(y_pred))
        
        if start >= len(y_pred):
            log(f"警告：超出資料範圍，跳過")
            break
        
        # 取得該 segment 的預測和標籤
        seg_pred = y_pred[start:end]
        seg_true = y_true[start:end]
        
        # 使用固定閾值計算
        seg_pred_binary = (seg_pred >= FIXED_THRESHOLD).astype(int)
        
        # 計算指標
        seg_precision = precision_score(seg_true, seg_pred_binary, zero_division=0)
        seg_recall = recall_score(seg_true, seg_pred_binary, zero_division=0)
        seg_f1 = f1_score(seg_true, seg_pred_binary, zero_division=0)
        seg_accuracy = np.mean(seg_true == seg_pred_binary)
        
        log(f"  資料筆數: {len(seg_pred)}")
        log(f"  真實標籤: {seg['expected_label']} ({'正常' if seg['expected_label']==0 else '異常'})")
        log(f"  Precision: {seg_precision:.4f}")
        log(f"  Recall: {seg_recall:.4f}")
        log(f"  F1: {seg_f1:.4f}")
        log(f"  Accuracy: {seg_accuracy:.4f}")
        
        # 儲存結果
        segment_results.append({
            'segment_id': i,
            'name': seg['name'],
            'start_idx': start,
            'end_idx': end,
            'length': len(seg_pred),
            'expected_label': seg['expected_label'],
            'precision': seg_precision,
            'recall': seg_recall,
            'f1': seg_f1,
            'accuracy': seg_accuracy,
            'avg_score': np.mean(seg_pred)
        })
        
        current_idx = end
    
    # 儲存 segment 結果
    segment_df = pd.DataFrame(segment_results)
    segment_csv_path = os.path.join(experiment_folder_path, "test_segments_info.csv")
    segment_df.to_csv(segment_csv_path, index=False)
    log(f"\n測試段資訊已儲存至: {segment_csv_path}")
    
    # ============================================================
    # 步驟 5: 整體評估
    # ============================================================
    log("")
    log("=" * 60)
    log("步驟 5: 整體評估結果（所有測試段合併）")
    log("=" * 60)
    log(f"使用固定閾值 (Fixed Threshold): {FIXED_THRESHOLD:.4f}")
    log("")
    
    # 整體評估
    y_pred_binary = (y_pred >= FIXED_THRESHOLD).astype(int)
    
    overall_precision = precision_score(y_true, y_pred_binary, zero_division=0)
    overall_recall = recall_score(y_true, y_pred_binary, zero_division=0)
    overall_f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    overall_accuracy = np.mean(y_true == y_pred_binary)
    overall_auroc = roc_auc_score(y_true, y_pred)
    overall_auprc = average_precision_score(y_true, y_pred)
    
    # 計算混淆矩陣
    TP = np.sum((y_true == 1) & (y_pred_binary == 1))
    TN = np.sum((y_true == 0) & (y_pred_binary == 0))
    FP = np.sum((y_true == 0) & (y_pred_binary == 1))
    FN = np.sum((y_true == 1) & (y_pred_binary == 0))
    
    log(f"總樣本數: {len(y_true)}")
    log(f"最佳閾值: {FIXED_THRESHOLD:.4f}")
    log(f"AUROC: {overall_auroc:.4f}")
    log(f"AUPRC: {overall_auprc:.4f}")
    log(f"Precision: {overall_precision:.4f}")
    log(f"Recall: {overall_recall:.4f}")
    log(f"F1 Score: {overall_f1:.4f}")
    log(f"Accuracy: {overall_accuracy:.4f}")
    log(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}")
    
    # 儲存整體結果
    overall_results = pd.DataFrame({
        'src_id': [args.id_src],
        'trg_id': [args.id_trg],
        'threshold': [FIXED_THRESHOLD],
        'auroc': [overall_auroc],
        'auprc': [overall_auprc],
        'precision': [overall_precision],
        'recall': [overall_recall],
        'f1': [overall_f1],
        'accuracy': [overall_accuracy],
        'TP': [TP],
        'TN': [TN],
        'FP': [FP],
        'FN': [FN]
    })
    
    overall_csv_path = os.path.join(experiment_folder_path, f'Ours_msltest_{args.id_src}.csv')
    overall_results.to_csv(overall_csv_path, index=False)
    log(f"\n整體結果已儲存至: {overall_csv_path}")
    
    log("")
    log("=" * 60)
    log("所有測試完成！")
    log("=" * 60)
    log(f"\n結果儲存位置: {experiment_folder_path}")
    log(f"- t-SNE 視覺化: tsne_visualization.png")
    log(f"- t-SNE 多視角: tsne_multi_view.png")
    log(f"- t-SNE 座標: tsne_coordinates.csv")
    log(f"- 整體評估結果: Ours_msltest_{args.id_src}.csv")
    log(f"- 測試段資訊: test_segments_info.csv")


if __name__ == '__main__':
    parser = ArgumentParser(description="HVAC 測試腳本 (含 t-SNE)")
    
    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='results',
                       help='實驗主資料夾路徑')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='HVAC',
                       help='實驗子資料夾名稱')
    parser.add_argument('--id_src', type=str, default='source_data',
                       help='Source 資料集 ID')
    parser.add_argument('--id_trg', type=str, default='target_data',
                       help='Target 資料集 ID')
    
    args = parser.parse_args()
    main(args)