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

    log = get_logger(os.path.join(saved_args.experiments_main_folder, args.experiment_folder,
                                  str(args.id_src) + "-" + str(saved_args.id_trg), "eval_" + saved_args.log))
    log(f"Using device: {DEVICE}")

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
        elif dataset_type == "hvac":
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

    # 1. 載入 Source Train 以取得 mean/std
    log("Loading source train dataset to get normalization stats...")
    dataset_src_train_for_stats = get_dataset(saved_args, domain_type="source", split_type="train")
    d_mean, d_std = dataset_src_train_for_stats.get_statistic()
    log(f"Using Mean/Std from Source Train (mean[0]={d_mean[0]}, std[0]={d_std[0]})")
    del dataset_src_train_for_stats 
    
    # 2. 載入 Source Test
    log("Loading Source Test...")
    dataset_test_src = get_dataset(saved_args, domain_type="source", split_type="test",
                                    d_mean=d_mean, d_std=d_std)
    
    # --- 修改：檢查是否需要載入 Target Test ---
    USE_TARGET = True  # 設定為 False 如果不想評估 Target
    
    if USE_TARGET:
        log("Loading Target Test...")
        try:
            dataset_test_trg = get_dataset(saved_args, domain_type="target", split_type="test",
                                        d_mean=d_mean, d_std=d_std)
        except Exception as e:
            log(f"Warning: Could not load Target Test dataset: {e}")
            log("Continuing with Source-only evaluation...")
            USE_TARGET = False
    else:
        log("Skipping Target Test (Source-only mode)")

    # 3. 初始化 Algorithm
    try:
        if len(dataset_test_src) == 0:
            log("錯誤：Source 測試資料集為空 (長度為 0)。")
            log("請檢查 'test_data.csv' 檔案是否包含數據。")
            sys.exit(1)
            
        input_channels_dim = dataset_test_src[0]['sequence'].shape[1]
        input_static_dim = dataset_test_src[0]['static'].shape[0] if 'static' in dataset_test_src[0] else 0
    
    except IndexError as e:
        log(f"錯誤：無法從測試資料集獲取維度: {e}")
        sys.exit(1)
    except Exception as e:
        log(f"載入資料集維度時發生未知錯誤: {e}")
        sys.exit(1)

    algorithm = get_algorithm(saved_args, input_channels_dim=input_channels_dim, 
                             input_static_dim=input_static_dim, device=DEVICE)
    
    experiment_folder_path = os.path.join(saved_args.experiments_main_folder, args.experiment_folder,
                                          str(args.id_src) + "-" + str(saved_args.id_trg))

    algorithm.load_state(experiment_folder_path)
    algorithm.eval()
    
    log("Starting to collect embeddings for t-SNE plot...")
    all_embeddings = []
    all_labels = []
    all_domains = []

    dataloader_test_src = DataLoader(dataset_test_src, batch_size=eval_batch_size,
                                     shuffle=False, num_workers=0, drop_last=False)
    
    if USE_TARGET:
        dataloader_test_trg = DataLoader(dataset_test_trg, batch_size=eval_batch_size,
                                         shuffle=False, num_workers=0, drop_last=False)

    # --- 修改：只在有 Target 時才處理 Target 資料 ---
    if USE_TARGET:
        for i_batch, sample_batched in enumerate(dataloader_test_trg):
            for key, value in sample_batched.items():
                sample_batched[key] = sample_batched[key].to(device=DEVICE, non_blocking=True)
            with torch.no_grad():
                embeddings = algorithm.get_embedding(sample_batched) 
                
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(sample_batched['label'].cpu().numpy())
            all_domains.extend(['Target'] * len(sample_batched['label']))
            algorithm.predict_trg(sample_batched)

        # 儲存 Target 預測結果
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
    else:
        log("Skipping Target evaluation (Source-only mode)")

    # 處理 Source 資料
    for i_batch, sample_batched in enumerate(dataloader_test_src):
        for key, value in sample_batched.items():
            sample_batched[key] = sample_batched[key].to(device=DEVICE, non_blocking=True)
        with torch.no_grad():
            embeddings = algorithm.get_embedding(sample_batched)
            
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(sample_batched['label'].cpu().numpy())
        all_domains.extend(['Source'] * len(sample_batched['label']))
        algorithm.predict_src(sample_batched)

    # 儲存 Source 預測結果
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

    # t-SNE 繪圖（只在有足夠資料時才繪製）
    if len(all_embeddings) > 0:
        log("Generating t-SNE plot...")
        try:
            final_embeddings = np.concatenate(all_embeddings, axis=0)
            final_labels = np.concatenate(all_labels, axis=0)
            final_domains = np.array(all_domains)

            tsne = TSNE(n_components=2, verbose=1, perplexity=40, max_iter=300, n_jobs=-1, random_state=42)
            embeddings_2d = tsne.fit_transform(final_embeddings)

            df_tsne = pd.DataFrame({
                'x': embeddings_2d[:, 0],
                'y': embeddings_2d[:, 1],
                'domain': final_domains
            })
            df_tsne['label_str'] = np.where(final_labels == 0, 'Normal (0)','Abnormal (1)')
            df_tsne['plot_category'] = df_tsne['domain'] + ' - ' + df_tsne['label_str']

            plt.figure(figsize=(12, 10))
            sns.scatterplot(
                data=df_tsne,
                x='x',
                y='y',
                hue='plot_category',
                style='domain',
                s=50,
                alpha=0.7
            )
            
            plt.title('t-SNE Visualization of Model Embeddings (Source Only)')
            plt.legend(loc='best')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            
            plot_save_path = os.path.join(experiment_folder_path, "tsne_embeddings_plot.png")
            plt.savefig(plot_save_path, dpi=300)
            log(f"t-SNE plot saved to {plot_save_path}")

        except Exception as e:
            log(f"Error generating t-SNE plot: {e}")

if __name__ == '__main__':
    parser = ArgumentParser(description="parse args")

    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='results')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='default')
    parser.add_argument('--id_src', type=str, default='1-1')
    parser.add_argument('--id_trg', type=str, default='1-5')

    args = parser.parse_args()
    main(args)