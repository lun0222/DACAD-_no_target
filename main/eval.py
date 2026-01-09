import sys
# 使用絕對路徑強制 E:\DACAD 進入搜尋路徑
# 確保 E:\\DACAD 是您專案的正確路徑
sys.path.insert(0, 'D:\\DACAD')
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd

from utils.dataset import get_dataset
from utils.augmentations import Augmenter

from torch.utils.data import DataLoader
from utils.util_progress_log import get_logger,  get_dataset_type

import json

from argparse import ArgumentParser
from collections import namedtuple

from algorithms import get_algorithm
import torch

from sklearn.manifold import TSNE  
import matplotlib.pyplot as plt    
import seaborn as sns              

def main(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(args.experiments_main_folder, args.experiment_folder,
                           str(args.id_src) + "-" + str(args.id_trg), 'commandline_args.txt'), 'r') as f:
        saved_args_dict_ = json.load(f)

    saved_args = namedtuple("SavedArgs", saved_args_dict_.keys())(*saved_args_dict_.values())

    # configure our logger
    log = get_logger(os.path.join(saved_args.experiments_main_folder, args.experiment_folder,
                                  str(args.id_src) + "-" + str(saved_args.id_trg), "eval_" + saved_args.log))
    log(f"Using device: {DEVICE}") # <-- 建議新增

    # Some functions and variables for logging
    dataset_type = get_dataset_type(saved_args)

    def log_scores(args, dataset_type, metrics_pred):
        if dataset_type == "smd":
            log("AUPRC score is : %.4f " % (metrics_pred["avg_prc"]))
            log("Best F1 score is : %.4f " % (metrics_pred["best_f1"]))
            log("Best Prec score is : %.4f " % (metrics_pred["best_prec"]))
            log("Best Rec score is : %.4f " % (metrics_pred["best_rec"]))
        elif dataset_type == "msl":
            log("AUPRC score is : %.4f " % (metrics_pred["avg_prc"]))
            log("Best F1 score is : %.4f " % (metrics_pred["best_f1"]))
            log("Best Prec score is : %.4f " % (metrics_pred["best_prec"]))
            log("Best Rec score is : %.4f " % (metrics_pred["best_rec"]))
        elif dataset_type == "boiler":
            log("AUPRC score is : %.4f " % (metrics_pred["avg_prc"]))
            log("Best F1 score is : %.4f " % (metrics_pred["best_f1"]))
            log("Best Prec score is : %.4f " % (metrics_pred["best_prec"]))
            log("Best Rec score is : %.4f " % (metrics_pred["best_rec"]))
        elif dataset_type == "hvac": # <-- 新增 HVAC
            log("AUPRC score is : %.4f " % (metrics_pred["avg_prc"]))
            log("Best F1 score is : %.4f " % (metrics_pred["best_f1"]))
            log("Best Prec score is : %.4f " % (metrics_pred["best_prec"]))
            log("Best Rec score is : %.4f " % (metrics_pred["best_rec"]))
        else:
            log("Accuracy score is : %.4f " % (metrics_pred["acc"]))
            log("Macro F1 score is : %.4f " % (metrics_pred["mac_f1"]))
            log("Weighted F1 score is : %.4f " % (metrics_pred["w_f1"]))

    batch_size = saved_args.batch_size
    eval_batch_size = saved_args.eval_batch_size

    # 1. 為了取得正確的 mean/std, 我們【必須】先載入 "source" 的 "train" split
    log("Loading source train dataset to get normalization stats...")
    # 假設你 `utils/dataset.py` 也改好了
    dataset_src_train_for_stats = get_dataset(saved_args, domain_type="source", split_type="train")
    
    # 2. 獲取 mean/std
    d_mean, d_std = dataset_src_train_for_stats.get_statistic()
    log(f"Using Mean/Std from Source Train (mean[0]={d_mean[0]}, std[0]={d_std[0]})")
    
    # 刪除這個僅用於獲取 stats 的 dataset
    del dataset_src_train_for_stats 
    
    # 3. 現在, 載入 "test" splits 並【傳入】 mean/std
    log("Loading Source Test...")
    dataset_test_src = get_dataset(saved_args, domain_type="source", split_type="test",
                                    d_mean=d_mean, d_std=d_std)
    
    log("Loading Target Test...")
    dataset_test_trg = get_dataset(saved_args, domain_type="target", split_type="test",
                                    d_mean=d_mean, d_std=d_std)

    # =============================================================================
    # START: 【重要修復】初始化 Algorithm
    # =============================================================================
    
    # 4. 從測試集中獲取模型需要的輸入維度
    try:
        # 檢查 dataset_test_src 是否為空
        if len(dataset_test_src) == 0:
            log("錯誤：Source 測試資料集為空 (長度為 0)。")
            log("請檢查 'test_data.csv' 檔案是否包含數據。")
            sys.exit(1)
            
        input_channels_dim = dataset_test_src[0]['sequence'].shape[1]
        input_static_dim = dataset_test_src[0]['static'].shape[0] if 'static' in dataset_test_src[0] else 0
    
    except IndexError as e:
        log(f"錯誤：無法從測試資料集獲取維度: {e}")
        log("請檢查 'test_data.csv' 檔案是否正確。")
        sys.exit(1) # 終止程式
    except Exception as e:
        log(f"載入資料集維度時發生未知錯誤: {e}")
        sys.exit(1)

    # 5. 建立 algorithm 物件
    #    (saved_args 包含了所有 train.py 的設定)
    algorithm = get_algorithm(saved_args, input_channels_dim=input_channels_dim, input_static_dim=input_static_dim, device=DEVICE)
    
    # =============================================================================
    # END: 【重要修復】
    # =============================================================================

    experiment_folder_path = os.path.join(saved_args.experiments_main_folder, args.experiment_folder,
                                          str(args.id_src) + "-" + str(saved_args.id_trg))

    algorithm.load_state(experiment_folder_path) # <-- 現在 'algorithm' 已經被定義

    # turn algorithm into eval mode
    algorithm.eval() # <-- 這裡也不會再報錯
    
    log("Starting to collect embeddings for t-SNE plot...")
    all_embeddings = []
    all_labels = []
    all_domains = []

    # 建立 Dataloaders (這也漏掉了)
    dataloader_test_src = DataLoader(dataset_test_src, batch_size=eval_batch_size,
                                     shuffle=False, num_workers=0, drop_last=False)
    dataloader_test_trg = DataLoader(dataset_test_trg, batch_size=eval_batch_size,
                                     shuffle=False, num_workers=0, drop_last=False)


    for i_batch, sample_batched in enumerate(dataloader_test_trg):
        # 將數據張量傳送到正確的設備 (這也漏掉了)
        for key, value in sample_batched.items():
            sample_batched[key] = sample_batched[key].to(device=DEVICE, non_blocking=True)
        with torch.no_grad():
            # 呼叫 get_embedding (此方法定義在 algorithms.py)
            embeddings = algorithm.get_embedding(sample_batched) 
            
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(sample_batched['label'].cpu().numpy())
        # 為 Target 數據標記 "Target"
        all_domains.extend(['Target'] * len(sample_batched['label']))
        algorithm.predict_trg(sample_batched)

    # even though the name is "pred_meter_val_trg", in this script it saves test results
    y_test_trg = np.array(algorithm.pred_meter_val_trg.target_list)
    y_pred_trg = np.array(algorithm.pred_meter_val_trg.output_list)
    id_test_trg = np.array(algorithm.pred_meter_val_trg.id_patient_list)
    stay_hour_trg = np.array(algorithm.pred_meter_val_trg.stay_hours_list)

    if len(id_test_trg) == 0 and len(stay_hour_trg) == 0:
        id_test_trg = [-1] * len(y_test_trg)
        stay_hour_trg = [-1] * len(y_test_trg)

    pred_trg_df = pd.DataFrame(
        {"patient_id": id_test_trg, "stay_hour": stay_hour_trg, "y": y_test_trg, "y_pred": y_pred_trg})
    df_save_path_trg = os.path.join(saved_args.experiments_main_folder, args.experiment_folder,
                                    str(args.id_src) + "-" + str(args.id_trg), "predictions_test_target.csv")
    pred_trg_df.to_csv(df_save_path_trg, index=False)

    log("Target results saved to " + df_save_path_trg)

    log("TARGET RESULTS")
    log("loaded from " + saved_args.path_trg)
    log("")

    metrics_pred_test_trg = algorithm.pred_meter_val_trg.get_metrics()

    log_scores(saved_args, dataset_type, metrics_pred_test_trg)
    df_trg = pd.DataFrame.from_dict(metrics_pred_test_trg, orient='index')
    df_trg = df_trg.T
    df_trg.insert(0, 'src_id', args.id_src)
    df_trg.insert(1, 'trg_id', args.id_trg)

    fname = 'Ours_msltest_' + args.id_src + ".csv"
    if os.path.isfile(fname):
        df_trg.to_csv(fname, mode='a', header=False, index=False)
    else:
        df_trg.to_csv(fname, mode='a', header=True, index=False)

    for i_batch, sample_batched in enumerate(dataloader_test_src):
        for key, value in sample_batched.items():
            sample_batched[key] = sample_batched[key].to(device=DEVICE, non_blocking=True)
        with torch.no_grad():
            embeddings = algorithm.get_embedding(sample_batched)
            
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(sample_batched['label'].cpu().numpy())
        # 為 Source 數據標記 "Source"
        all_domains.extend(['Source'] * len(sample_batched['label']))
        algorithm.predict_src(sample_batched)

    # even though the name is "pred_meter_val_src", in this script it saves test results
    y_test_src = np.array(algorithm.pred_meter_val_src.target_list)
    y_pred_src = np.array(algorithm.pred_meter_val_src.output_list)
    id_test_src = np.array(algorithm.pred_meter_val_src.id_patient_list)
    stay_hour_src = np.array(algorithm.pred_meter_val_src.stay_hours_list)

    if len(id_test_src) == 0 and len(stay_hour_src) == 0:
        id_test_src = [-1] * len(y_test_src)
        stay_hour_src = [-1] * len(y_test_src)

    pred_src_df = pd.DataFrame(
        {"patient_id": id_test_src, "stay_hour": stay_hour_src, "y": y_test_src, "y_pred": y_pred_src})
    df_save_path_src = os.path.join(saved_args.experiments_main_folder, saved_args.experiment_folder,
                                    str(saved_args.id_src) + "-" + str(saved_args.id_trg),
                                    "predictions_test_source.csv")
    pred_src_df.to_csv(df_save_path_src, index=False)

    log("Source results saved to " + df_save_path_src)

    log("SOURCE RESULTS")
    log("loaded from " + saved_args.path_src)
    log("")

    metrics_pred_test_src = algorithm.pred_meter_val_src.get_metrics()

    log_scores(saved_args, dataset_type, metrics_pred_test_src)

    log("Generating t-SNE plot...")
    try:
        # 1. 彙整所有數據
        final_embeddings = np.concatenate(all_embeddings, axis=0)
        final_labels = np.concatenate(all_labels, axis=0)
        final_domains = np.array(all_domains) # 已經是一維列表，轉 array 即可

        # 2. 執行 t-SNE
        #    n_jobs=-1 使用所有 CPU 核心
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, max_iter=300, n_jobs=-1, random_state=42)
        embeddings_2d = tsne.fit_transform(final_embeddings)

        # 3. 建立 DataFrame 以便繪圖
        #    (記住： 1 是正常, 0 是異常)
        df_tsne = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'domain': final_domains
        })
        # 建立一個組合標籤，就像論文中那樣
        df_tsne['label_str'] = np.where(final_labels == 0, 'Normal (0)','Abnormal (1)')
        df_tsne['plot_category'] = df_tsne['domain'] + ' - ' + df_tsne['label_str']

        # 4. 使用 Seaborn 繪圖
        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            data=df_tsne,
            x='x',
            y='y',
            hue='plot_category', # 顏色
            style='domain',      # 形狀
            s=50,                # 點的大小
            alpha=0.7
        )
        
        plt.title('t-SNE Visualization of Model Embeddings')
        plt.legend(loc='best')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        
        # 5. 儲存圖片
        plot_save_path = os.path.join(experiment_folder_path, "tsne_embeddings_plot.png")
        plt.savefig(plot_save_path, dpi=300)
        log(f"t-SNE plot saved to {plot_save_path}")

    except Exception as e:
        log(f"Error generating t-SNE plot: {e}")

# parse command-line arguments and execute the main method
if __name__ == '__main__':
    parser = ArgumentParser(description="parse args")

    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='results')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='default')
    parser.add_argument('--id_src', type=str, default='1-1')
    parser.add_argument('--id_trg', type=str, default='1-5')

    args = parser.parse_args()

    main(args)