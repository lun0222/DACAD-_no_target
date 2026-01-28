import sys
sys.path.insert(0, 'D:\\DACAD-_no_target')
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
from utils.dataset import get_dataset
from torch.utils.data import DataLoader
from utils.util_progress_log import get_logger, get_dataset_type
import json
from argparse import ArgumentParser
from collections import namedtuple
from algorithms import get_algorithm
import torch
from sklearn.manifold import TSNE  
import matplotlib.pyplot as plt    
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score

def main(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(os.path.join(args.experiments_main_folder, args.experiment_folder,
                           str(args.id_src) + "-" + str(args.id_trg), 
                           'commandline_args.txt'), 'r') as f:
        saved_args_dict_ = json.load(f)

    saved_args = namedtuple("SavedArgs", saved_args_dict_.keys())(*saved_args_dict_.values())

    log = get_logger(os.path.join(saved_args.experiments_main_folder, args.experiment_folder,
                                  str(args.id_src) + "-" + str(saved_args.id_trg), 
                                  "eval_" + saved_args.log))
    log(f"Using device: {DEVICE}")
    log("=== Source-Only Evaluation Mode ===")

    dataset_type = get_dataset_type(saved_args)

    def log_scores(args, dataset_type, metrics_pred):
        if dataset_type in ["smd", "msl", "boiler", "hvac"]:
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

    # 1. Load Source Train to get normalization stats
    log("Loading source train dataset to get normalization stats...")
    dataset_src_train_for_stats = get_dataset(saved_args, domain_type="source", split_type="train")
    d_mean, d_std = dataset_src_train_for_stats.get_statistic()
    log(f"Using Mean/Std from Source Train (mean[0]={d_mean[0]}, std[0]={d_std[0]})")
    del dataset_src_train_for_stats 
    
    # 2. Load Source Test
    log("Loading Source Test...")
    dataset_test_src = get_dataset(saved_args, domain_type="source", split_type="test",
                                    d_mean=d_mean, d_std=d_std)
    
    # 3. Get model input dimensions
    try:
        if len(dataset_test_src) == 0:
            log("ERROR: Source test dataset is empty (length 0).")
            log("Please check if 'test_data.csv' contains data.")
            sys.exit(1)
            
        input_channels_dim = dataset_test_src[0]['sequence'].shape[1]
        input_static_dim = dataset_test_src[0]['static'].shape[0] if 'static' in dataset_test_src[0] else 0
    
    except IndexError as e:
        log(f"ERROR: Cannot get dimensions from test dataset: {e}")
        sys.exit(1)
    except Exception as e:
        log(f"Unknown error while loading dataset dimensions: {e}")
        sys.exit(1)

    # 4. Initialize algorithm
    algorithm = get_algorithm(saved_args, input_channels_dim=input_channels_dim, 
                             input_static_dim=input_static_dim, device=DEVICE)
    
    experiment_folder_path = os.path.join(saved_args.experiments_main_folder, args.experiment_folder,
                                          str(args.id_src) + "-" + str(saved_args.id_trg))

    # 5. Load model weights
    algorithm.load_state(experiment_folder_path)
    algorithm.eval()
    
    log("Starting to collect embeddings for t-SNE plot...")
    all_embeddings = []
    all_labels = []
    all_domains = []

    # 6. Create DataLoaders
    dataloader_test_src = DataLoader(dataset_test_src, batch_size=eval_batch_size,
                                     shuffle=False, num_workers=0, drop_last=False)

    # 7. Process Source data
    log(f"Processing {len(dataloader_test_src)} batches...")
    for i_batch, sample_batched in enumerate(dataloader_test_src):
        for key, value in sample_batched.items():
            sample_batched[key] = sample_batched[key].to(device=DEVICE, non_blocking=True)
        
        with torch.no_grad():
            embeddings = algorithm.get_embedding(sample_batched)
            
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(sample_batched['label'].cpu().numpy())
        all_domains.extend(['Source'] * len(sample_batched['label']))
        algorithm.predict_src(sample_batched)
        
        if (i_batch + 1) % 10 == 0:
            log(f"  Processed batch {i_batch + 1}/{len(dataloader_test_src)}")

    log(f"Collected {len(all_embeddings)} embedding batches")

    # 8. Save Source predictions
    y_test_src = np.array(algorithm.pred_meter_val_src.target_list)
    y_pred_src = np.array(algorithm.pred_meter_val_src.output_list)
    id_test_src = np.array(algorithm.pred_meter_val_src.id_patient_list)
    stay_hour_src = np.array(algorithm.pred_meter_val_src.stay_hours_list)

    if len(id_test_src) == 0 and len(stay_hour_src) == 0:
        id_test_src = [-1] * len(y_test_src)
        stay_hour_src = [-1] * len(y_test_src)

    pred_src_df = pd.DataFrame({
        "patient_id": id_test_src, 
        "stay_hour": stay_hour_src, 
        "y": y_test_src, 
        "y_pred": y_pred_src
    })
    
    df_save_path_src = os.path.join(saved_args.experiments_main_folder, saved_args.experiment_folder,
                                    str(saved_args.id_src) + "-" + str(saved_args.id_trg),
                                    "predictions_test_source.csv")
    pred_src_df.to_csv(df_save_path_src, index=False)
    log("Source results saved to " + df_save_path_src)

    log("")
    log("=" * 60)
    log("SOURCE RESULTS")
    log("loaded from " + saved_args.path_src)
    log("=" * 60)

    # 使用固定閾值
    FIXED_THRESHOLD = 2.0
    metrics_pred_test_src = algorithm.pred_meter_val_src.get_metrics(fixed_threshold=FIXED_THRESHOLD)
    log_scores(saved_args, dataset_type, metrics_pred_test_src)

    # 9. Save Source metrics
    log("")
    log("Saving Source evaluation metrics...")
    df_src = pd.DataFrame.from_dict(metrics_pred_test_src, orient='index')
    df_src = df_src.T
    df_src.insert(0, 'src_id', args.id_src)
    df_src.insert(1, 'trg_id', args.id_trg)

    fname_in_folder = os.path.join(experiment_folder_path, f'Ours_msltest_{args.id_src}.csv')
    df_src.to_csv(fname_in_folder, mode='w', header=True, index=False)
    log(f"Saved metrics to: {fname_in_folder}")

    fname_root = f'Ours_msltest_{args.id_src}.csv'
    if os.path.isfile(fname_root):
        df_src.to_csv(fname_root, mode='a', header=False, index=False)
        log(f"Appended metrics to: {fname_root}")
    else:
        df_src.to_csv(fname_root, mode='w', header=True, index=False)
        log(f"Created metrics file: {fname_root}")

    # 10. Generate t-SNE plot (Source only)
    log("")
    log("=" * 60)
    log("Generating t-SNE plot (Source only)...")
    log("=" * 60)
    
    if len(all_embeddings) > 0:
        try:
            # 合併所有 embeddings
            final_embeddings = np.concatenate(all_embeddings, axis=0)
            final_labels = np.concatenate(all_labels, axis=0)
            final_domains = np.array(all_domains)
            
            log(f"Total embeddings shape: {final_embeddings.shape}")
            log(f"Total labels shape: {final_labels.shape}")
            log(f"Unique labels: {np.unique(final_labels)}")
            log(f"Label distribution: Normal={np.sum(final_labels==0)}, Abnormal={np.sum(final_labels==1)}")

            # 設定中文字型
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False

            # 執行 t-SNE
            log("Running t-SNE (this may take a few minutes)...")
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, max_iter=300, 
                       n_jobs=-1, random_state=42)
            embeddings_2d = tsne.fit_transform(final_embeddings)
            log(f"t-SNE completed. 2D embeddings shape: {embeddings_2d.shape}")

            # 建立 DataFrame
            df_tsne = pd.DataFrame({
                'x': embeddings_2d[:, 0],
                'y': embeddings_2d[:, 1],
                'domain': final_domains
            })
            df_tsne['label_str'] = np.where(final_labels == 0, 'Normal (0)', 'Abnormal (1)')
            df_tsne['plot_category'] = df_tsne['domain'] + ' - ' + df_tsne['label_str']

            # 繪圖
            log("Creating plot...")
            plt.figure(figsize=(12, 10))
            sns.scatterplot(
                data=df_tsne,
                x='x',
                y='y',
                hue='plot_category',
                style='domain',
                s=50,
                alpha=0.7,
                palette='Set2'
            )
            
            plt.title('t-SNE Visualization of Model Embeddings (Source Only)', fontsize=16)
            plt.legend(loc='best', fontsize=12)
            plt.xlabel('t-SNE Component 1', fontsize=14)
            plt.ylabel('t-SNE Component 2', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 儲存圖片
            plot_save_path = os.path.join(experiment_folder_path, "tsne_embeddings_plot_source_only.png")
            plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
            log(f"t-SNE plot saved to {plot_save_path}")
            plt.close()
            
            log("t-SNE plot generation completed successfully!")

        except Exception as e:
            log(f"Error generating t-SNE plot: {e}")
            import traceback
            log(traceback.format_exc())
    else:
        log("Warning: No embeddings collected, skipping t-SNE plot")

    log("")
    log("=" * 60)
    log("Evaluation Complete!")
    log("=" * 60)

if __name__ == '__main__':
    parser = ArgumentParser(description="Source-Only Model Evaluation")

    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='results')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='default')
    parser.add_argument('--id_src', type=str, default='1-1')
    parser.add_argument('--id_trg', type=str, default='1-5')

    args = parser.parse_args()
    main(args)