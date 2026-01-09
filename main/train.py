import sys
# 使用絕對路徑強制 E:\DACAD 進入搜尋路徑
# 確保 E:\\DACAD 是您專案的正確路徑
sys.path.insert(0, 'D:\\DACAD') 
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import random
import torch
from utils.dataset import get_dataset
from torch.utils.data import DataLoader
from utils.util_progress_log import ProgressMeter, get_logger, get_dataset_type
import json
from argparse import ArgumentParser
from algorithms import get_algorithm


def main(args):
    # Torch RNG
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Python RNG
    np.random.seed(args.seed)
    random.seed(args.seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    # First configure our logger
    log = get_logger(
        os.path.join(args.experiments_main_folder, args.experiment_folder, str(args.id_src) + "-" + str(args.id_trg),
                     args.log))
    log(f"Using device: {DEVICE}") # <-- 建議新增，方便除錯

    # Some functions and variables for logging
    dataset_type = get_dataset_type(args)

    # --- 驗證相關函式，可以保留或刪除 (因為不會再被呼叫) ---
    def log_scores(args, dataset_type, metrics_pred):
        if dataset_type == "smd":
            log("AUPRC score is : %.4f " % (metrics_pred["avg_prc"]))
            log("Best F1 score is : %.4f " % (metrics_pred["best_f1"]))
        elif dataset_type == "msl":
            log("AUPRC score is : %.4f " % (metrics_pred["avg_prc"]))
            log("Best F1 score is : %.4f " % (metrics_pred["best_f1"]))
        elif dataset_type == "boiler":
            log("AUPRC score is : %.4f " % (metrics_pred["avg_prc"]))
            log("Best F1 score is : %.4f " % (metrics_pred["best_f1"]))
        elif dataset_type == "hvac": # <-- 新增 HVAC
            log("AUPRC score is : %.4f " % (metrics_pred["avg_prc"]))
            log("Best F1 score is : %.4f " % (metrics_pred["best_f1"]))
        else:
            log("Accuracy score is : %.4f " % (metrics_pred["acc"]))
            log("Macro F1 score is : %.4f " % (metrics_pred["mac_f1"]))
            log("Weighted F1 score is : %.4f " % (metrics_pred["w_f1"]))
    # --- END ---

    batch_size = args.batch_size
    # --- REMOVE START ---
    # eval_batch_size = args.eval_batch_size
    # num_val_iteration = args.num_val_iteration
    # --- REMOVE END ---


# 1. 建立 Source Train Dataset (唯一不傳 d_mean/d_std 的)
    #    它將根據 "source_data_train.csv" (假設檔名) 計算自己的 mean/std
    log("Loading Source Train...")
    dataset_src = get_dataset(args, domain_type="source", split_type="train")
    
    # 2. 【關鍵】從 Source Train 取出 mean 和 std
    #    *** 這裡定義了 d_mean 和 d_std ***
    d_mean, d_std = dataset_src.get_statistic()
    log(f"Using Mean/Std calculated from Source Train (mean[0]={d_mean[0]}, std[0]={d_std[0]})")

    # --- REMOVE START ---
    # 3. 建立 Source Val, 並【傳入】 mean/std
    # log("Loading Source Val...")
    # dataset_val_src = get_dataset(args, domain_type="source", split_type="val", 
    #                               d_mean=d_mean, d_std=d_std)
    # --- REMOVE END ---

    # 4. 建立 Target Train, 並【傳入】 mean/std
    # log("Loading Target Train...")
    # dataset_trg = get_dataset(args, domain_type="target", split_type="train", 
    #                           d_mean=d_mean, d_std=d_std)
    
    # --- REMOVE START ---
    # 5. 建立 Target Val, 並【傳入】 mean/std
    # log("Loading Target Val...")
    # dataset_val_trg = get_dataset(args, domain_type="target", split_type="val", 
    #                                 d_mean=d_mean, d_std=d_std)
    # --- REMOVE END ---


    # 6. 建立 DataLoaders (這部分不變)
    dataloader_src = DataLoader(dataset_src, batch_size=batch_size,
                                shuffle=True, num_workers=0, drop_last=True)
    
    # --- REMOVE START ---
    # dataloader_val_src = DataLoader(dataset_val_src, batch_size=eval_batch_size,
    #                                 shuffle=True, num_workers=0, drop_last=True)
    # dataloader_val_trg = DataLoader(dataset_val_trg, batch_size=eval_batch_size,
    #                                 shuffle=True, num_workers=0, drop_last=True)

    # max_num_val_iteration = min(len(dataloader_val_src), len(dataloader_val_trg))
    # if max_num_val_iteration < num_val_iteration:
    #     num_val_iteration = max_num_val_iteration
    # --- REMOVE END ---


    # 7. 保持不變 (lines 85-86)
    input_channels_dim = dataset_src[0]['sequence'].shape[1]
    input_static_dim = dataset_src[0]['static'].shape[0] if 'static' in dataset_src[0] else 0

    # 8. Get our algorithm (保持不變)
    algorithm = get_algorithm(args, input_channels_dim=input_channels_dim, input_static_dim=input_static_dim, device=DEVICE)

    experiment_folder_path = os.path.join(args.experiments_main_folder, args.experiment_folder,
                                        str(args.id_src) + "-" + str(args.id_trg))

    # Initialize progress metrics before training
    count_step = 0
    # best_val_score = -100 # <-- REMOVE (不再需要)

    # 9. 【關鍵】修改這裡，使用上面定義的 d_mean/d_std (lines 94-95)
    src_mean, src_std = d_mean, d_std 
    # trg_mean, trg_std = d_mean, d_std
    for i in range(args.num_epochs):
        # dataloader_trg = DataLoader(dataset_trg, batch_size=batch_size,
        #                             shuffle=True, num_workers=0, drop_last=True)
        # dataloader_iterator = iter(dataloader_trg)

        # <-- 新增開始 -->
        # 在 epoch 開始時初始化 ProgressMeter
        progress = ProgressMeter(
            len(dataloader_src),
            algorithm.return_metrics(),
            prefix="Epoch: [{}]".format(i))
        
        # 設定進度顯示的頻率 (例如每 50 個 batch 顯示一次)
        progress_freq = 10 
        # <-- 新增結束 -->

        for i_batch, sample_batched_src in enumerate(dataloader_src):
                sample_batched_src = sample_batched_src
                for key, value in sample_batched_src.items():
                    # 將數據張量傳送到正確的設備
                    sample_batched_src[key] = sample_batched_src[key].to(device=DEVICE, non_blocking=True)
                # Current model does not support smaller batches than batch_size (due to queue ptr)
                if len(sample_batched_src['sequence']) != batch_size:
                    continue

                # try:
                #     sample_batched_trg = next(dataloader_iterator)
                # except StopIteration:
                #     dataloader_trg = DataLoader(dataset_trg, batch_size=batch_size,
                #                                 shuffle=True, num_workers=0, drop_last=True)
                #     dataloader_iterator = iter(dataloader_trg)
                #     sample_batched_trg = next(dataloader_iterator)

                # for key, value in sample_batched_trg.items():
                #     # 將數據張量傳送到正確的設備
                #     sample_batched_trg[key] = sample_batched_trg[key].to(device=DEVICE, non_blocking=True)

                # # Current model does not support smaller batches than batch_size (due to queue ptr)
                # if len(sample_batched_trg['sequence']) != batch_size:
                #     continue

                # Training step of algorithm
                algorithm.step(sample_batched_src, sample_batched_trg=None, count_step=count_step, epoch=i,
                               src_mean=src_mean, src_std=src_std, trg_mean=None, trg_std=None)

                # <-- 新增開始 -->
                # 定期顯示進度
                if (i_batch + 1) % progress_freq == 0:
                    log(progress.display(i_batch + 1, is_logged=True))
                # <-- 新增結束 -->

                count_step += 1
                if count_step % len(dataloader_src) == 0:
                    
                    # <-- 修改：只記錄最終進度 -->
                    log(progress.display(i_batch + 1, is_logged=True))

                    # Refresh the saved metrics for algorithm (for next epoch's progress meter)
                    algorithm.init_metrics()

                    # --- 驗證邏輯已移除 ---
                    # --- START: 新增的模型儲存邏輯 ---
                    log(f"Epoch {i} complete. Saving model state...")
                    algorithm.save_state(experiment_folder_path)
                    # --- END: 新增的模型儲存邏輯 ---

                else:
                    continue
                break


# parse command-line arguments and execute the main method
if __name__ == '__main__':

    parser = ArgumentParser(description="parse args")

    parser.add_argument('--algo_name', type=str, default='dacad')

    parser.add_argument('-dr', '--dropout', type=float, default=0.1)
    parser.add_argument('-mo', '--momentum', type=float, default=0.99)  # DACAD
    parser.add_argument('-qs', '--queue_size', type=int, default=98304)  # DACAD
    parser.add_argument('--use_batch_norm', action='store_true')
    parser.add_argument('--use_mask', action='store_true')  # DACAD
    parser.add_argument('-wr', '--weight_ratio', type=float, default=10.0)
    parser.add_argument('-bs', '--batch_size', type=int, default=200)  # 2048)
    parser.add_argument('-ebs', '--eval_batch_size', type=int, default=200)  # 2048)
    parser.add_argument('-nvi', '--num_val_iteration', type=int, default=50)
    parser.add_argument('-ne', '--num_epochs', type=int, default=10)
    parser.add_argument('-ns', '--num_steps', type=int, default=1000)
    parser.add_argument('-cf', '--checkpoint_freq', type=int, default=1000)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4)
    parser.add_argument('-ws', '--warmup_steps', type=int, default=2000)
    parser.add_argument('--num_channels_TCN', type=str, default='64-64-64-64-64')  # All TCN models
    parser.add_argument('--kernel_size_TCN', type=int, default=3)  # All TCN models
    parser.add_argument('--dilation_factor_TCN', type=int, default=2)  # All TCN models
    parser.add_argument('--stride_TCN', type=int, default=1)  # All TCN models
    parser.add_argument('--hidden_dim_MLP', type=int, default=256)  # All classifier and discriminators

    # The weight of the domain classification loss
    parser.add_argument('-w_d', '--weight_domain', type=float, default=0.1)
    # Below weights are defined for DACAD
    parser.add_argument('--weight_loss_src', type=float, default=0.0)
    parser.add_argument('--weight_loss_trg', type=float, default=0.0)
    parser.add_argument('--weight_loss_ts', type=float, default=0.0)
    parser.add_argument('--weight_loss_disc', type=float, default=0.5)
    parser.add_argument('--weight_loss_pred', type=float, default=1.0)
    parser.add_argument('--weight_loss_src_sup', type=float, default=0.1)
    parser.add_argument('--weight_loss_trg_inj', type=float, default=0.1)

    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='results')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='smd')

    parser.add_argument('--path_src', type=str, default='../../datasets/MSL_SMAP') #../datasets/Boiler/   ../datasets/MSL_SMAP
    parser.add_argument('--path_trg', type=str, default='../../datasets/MSL_SMAP') #../datasets/SMD/test
    parser.add_argument('--age_src', type=int, default=-1)
    parser.add_argument('--age_trg', type=int, default=-1)
    parser.add_argument('--id_src', type=str, default='1-5')
    parser.add_argument('--id_trg', type=str, default='1-1')

    parser.add_argument('--task', type=str, default='decompensation')

    parser.add_argument('-l', '--log', type=str, default='train.log')

    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--features', type=str, nargs='+', default=None, help='List of feature column names to use.')
    args = parser.parse_args()

    if not os.path.exists(args.experiments_main_folder):
        os.mkdir(args.experiments_main_folder)
    if not os.path.exists(os.path.join(args.experiments_main_folder, args.experiment_folder)):
        os.makedirs(os.path.join(args.experiments_main_folder, args.experiment_folder), exist_ok=True)
    if not os.path.exists(os.path.join(args.experiments_main_folder, args.experiment_folder,
                                       str(args.id_src) + "-" + str(args.id_trg))):
        os.mkdir(os.path.join(args.experiments_main_folder, args.experiment_folder,
                              str(args.id_src) + "-" + str(args.id_trg)))

    with open(os.path.join(args.experiments_main_folder, args.experiment_folder,
                           str(args.id_src) + "-" + str(args.id_trg), 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    main(args)