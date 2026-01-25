import sys
sys.path.insert(0, 'D:\\DACAD-_no_target') 
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

    # Configure logger
    log = get_logger(
        os.path.join(args.experiments_main_folder, args.experiment_folder, 
                     str(args.id_src) + "-" + str(args.id_trg), args.log))
    log(f"Using device: {DEVICE}")
    log("=== Source-Only Training Mode (No Target Domain) ===")

    dataset_type = get_dataset_type(args)
    batch_size = args.batch_size

    # Load Source Train Dataset (calculates its own mean/std)
    log("Loading Source Train...")
    dataset_src = get_dataset(args, domain_type="source", split_type="train")
    
    # Get mean and std from Source Train
    d_mean, d_std = dataset_src.get_statistic()
    log(f"Using Mean/Std from Source Train (mean[0]={d_mean[0]}, std[0]={d_std[0]})")

    # Create DataLoader for Source Train
    dataloader_src = DataLoader(dataset_src, batch_size=batch_size,
                                shuffle=True, num_workers=0, drop_last=True)

    # Get input dimensions
    input_channels_dim = dataset_src[0]['sequence'].shape[1]
    input_static_dim = dataset_src[0]['static'].shape[0] if 'static' in dataset_src[0] else 0

    # Initialize algorithm
    algorithm = get_algorithm(args, input_channels_dim=input_channels_dim, 
                             input_static_dim=input_static_dim, device=DEVICE)

    experiment_folder_path = os.path.join(args.experiments_main_folder, args.experiment_folder,
                                          str(args.id_src) + "-" + str(args.id_trg))

    # Training loop
    count_step = 0
    src_mean, src_std = d_mean, d_std
    
    for i in range(args.num_epochs):
        # Initialize ProgressMeter at epoch start
        progress = ProgressMeter(
            len(dataloader_src),
            algorithm.return_metrics(),
            prefix="Epoch: [{}]".format(i))
        
        progress_freq = 10 
        
        for i_batch, sample_batched_src in enumerate(dataloader_src):
            # Move data to device
            for key, value in sample_batched_src.items():
                sample_batched_src[key] = sample_batched_src[key].to(device=DEVICE, non_blocking=True)
            
            # Skip batch if size doesn't match (for queue compatibility)
            if len(sample_batched_src['sequence']) != batch_size:
                continue

            # Training step (NO TARGET)
            algorithm.step(sample_batched_src, sample_batched_trg=None, 
                          count_step=count_step, epoch=i,
                          src_mean=src_mean, src_std=src_std, 
                          trg_mean=None, trg_std=None)

            # Display progress periodically
            if (i_batch + 1) % progress_freq == 0:
                log(progress.display(i_batch + 1, is_logged=True))

            count_step += 1
            
            # End of epoch
            if count_step % len(dataloader_src) == 0:
                # Log final progress
                log(progress.display(i_batch + 1, is_logged=True))

                # Reset metrics for next epoch
                algorithm.init_metrics()

                # Save model
                log(f"Epoch {i} complete. Saving model state...")
                algorithm.save_state(experiment_folder_path)
                break


if __name__ == '__main__':
    parser = ArgumentParser(description="Source-Only Training (No Target Domain)")

    parser.add_argument('--algo_name', type=str, default='dacad')
    parser.add_argument('-dr', '--dropout', type=float, default=0.1)
    parser.add_argument('-mo', '--momentum', type=float, default=0.99)
    parser.add_argument('-qs', '--queue_size', type=int, default=98304)
    parser.add_argument('--use_batch_norm', action='store_true')
    parser.add_argument('--use_mask', action='store_true')
    parser.add_argument('-wr', '--weight_ratio', type=float, default=10.0)
    parser.add_argument('-bs', '--batch_size', type=int, default=200)
    parser.add_argument('-ebs', '--eval_batch_size', type=int, default=200)
    parser.add_argument('-nvi', '--num_val_iteration', type=int, default=50)
    parser.add_argument('-ne', '--num_epochs', type=int, default=10)
    parser.add_argument('-ns', '--num_steps', type=int, default=1000)
    parser.add_argument('-cf', '--checkpoint_freq', type=int, default=1000)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4)
    parser.add_argument('-ws', '--warmup_steps', type=int, default=2000)
    parser.add_argument('--num_channels_TCN', type=str, default='64-64-64-64-64')
    parser.add_argument('--kernel_size_TCN', type=int, default=3)
    parser.add_argument('--dilation_factor_TCN', type=int, default=2)
    parser.add_argument('--stride_TCN', type=int, default=1)
    parser.add_argument('--hidden_dim_MLP', type=int, default=256)

    # Loss weights
    parser.add_argument('-w_d', '--weight_domain', type=float, default=0.1)
    parser.add_argument('--weight_loss_src', type=float, default=0.0)
    parser.add_argument('--weight_loss_trg', type=float, default=0.0)
    parser.add_argument('--weight_loss_ts', type=float, default=0.0)
    parser.add_argument('--weight_loss_disc', type=float, default=0)
    parser.add_argument('--weight_loss_pred', type=float, default=1.0)
    parser.add_argument('--weight_loss_src_sup', type=float, default=0.5)
    parser.add_argument('--weight_loss_trg_inj', type=float, default=0)

    # Paths
    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='results')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='smd')
    parser.add_argument('--path_src', type=str, default='../../datasets/MSL_SMAP')
    parser.add_argument('--path_trg', type=str, default='../../datasets/MSL_SMAP')
    parser.add_argument('--age_src', type=int, default=-1)
    parser.add_argument('--age_trg', type=int, default=-1)
    parser.add_argument('--id_src', type=str, default='1-5')
    parser.add_argument('--id_trg', type=str, default='1-1')
    parser.add_argument('--task', type=str, default='decompensation')
    parser.add_argument('-l', '--log', type=str, default='train.log')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--features', type=str, nargs='+', default=None)
    
    args = parser.parse_args()

    # Create directories
    if not os.path.exists(args.experiments_main_folder):
        os.mkdir(args.experiments_main_folder)
    if not os.path.exists(os.path.join(args.experiments_main_folder, args.experiment_folder)):
        os.makedirs(os.path.join(args.experiments_main_folder, args.experiment_folder), exist_ok=True)
    if not os.path.exists(os.path.join(args.experiments_main_folder, args.experiment_folder,
                                       str(args.id_src) + "-" + str(args.id_trg))):
        os.mkdir(os.path.join(args.experiments_main_folder, args.experiment_folder,
                              str(args.id_src) + "-" + str(args.id_trg)))

    # Save arguments
    with open(os.path.join(args.experiments_main_folder, args.experiment_folder,
                           str(args.id_src) + "-" + str(args.id_trg), 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    main(args)