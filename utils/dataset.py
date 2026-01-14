import sys
# 使用絕對路徑強制 E:\DACAD 進入搜尋路徑
# 確保 E:\\DACAD 是您專案的正確路徑
sys.path.insert(0, 'D:\\DACAD')

import ast
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.augmentations import Injector # <-- 確保這一行在頂部
# ... (檔案的其餘部分保持不變)
from sklearn.model_selection import train_test_split

def get_dataset(args, domain_type, split_type,d_mean=None, d_std=None):
    """
    Return the correct dataset object that will be fed into datalaoder
    args: args of main script
    domain_type: "source" or "target"
    split_type: "train" or "val" or "test"
    """
    
    if "SMD" in args.path_src:
        if domain_type == "source":
            return SMDDataset(args.path_src, subject_id=args.id_src, split_type=split_type, is_cuda=True)
        else:
            return SMDDataset_trg(args.path_trg, subject_id=args.id_trg, split_type=split_type, is_cuda=True)

    elif "MSL" in args.path_src:
        if domain_type == "source":
            return MSLDataset(args.path_src, subject_id=args.id_src, split_type=split_type, is_cuda=True)
        else:
            return MSLDataset_trg(args.path_trg, subject_id=args.id_trg, split_type=split_type, is_cuda=True)

    elif "Boiler" in args.path_src:
        if domain_type == "source":
            return BoilerDataset(args.path_src, subject_id=args.id_src, split_type=split_type, is_cuda=True)
        else:
            return BoilerDataset_trg(args.path_trg, subject_id=args.id_trg, split_type=split_type, is_cuda=True)
            
    elif "HVAC" in args.path_src: # <-- HVAC 區塊
        feature_columns = getattr(args, 'features', None) 
        if domain_type == "source":
            # 將 d_mean, d_std 傳遞給 Dataset
            return HVACDataset(args.path_src, subject_id=args.id_src, split_type=split_type, is_cuda=True,
                                feature_columns=feature_columns, d_mean=d_mean, d_std=d_std)
        else:
            # 將 d_mean, d_std 傳遞給 Dataset
            return HVACDataset_trg(args.path_trg, subject_id=args.id_trg, split_type=split_type, is_cuda=True,
                                feature_columns=feature_columns, d_mean=d_mean, d_std=d_std)

# =============================================================================
# START: 修改 HVAC 類別
# =============================================================================
class HVACDataset(Dataset):
    def __init__(self, root_dir, subject_id, split_type="train", is_cuda=True, verbose=False, 
                 feature_columns=None, w_size=10, stride=1, 
                 d_mean=None, d_std=None): # <-- 1. 新增 d_mean, d_std 參數
        
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda and torch.cuda.is_available()
        self.verbose = verbose
        self.feature_columns = feature_columns
        self.w = w_size
        self.s = stride

        # 傳入的 d_mean, d_std
        self.d_mean = d_mean
        self.d_std = d_std
        
        self.sequence = None
        self.label = None

        self.load_sequence() # 載入並處理資料
        
        self.sequence , self.label = self.convert_to_windows(self.w, self.s)
        
        self.positive = self.sequence[self.label == 0]
        self.negative = self.sequence[self.label == 1]

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, id_):
        # 使用我們上次修正過的 __getitem__ 邏輯，以防萬一
        sequence = self.sequence[id_]
        pid_ = np.random.randint(0, len(self.positive))
        positive = self.positive[pid_]

        random_choice = np.random.randint(0, 10)
        if len(self.negative) > 0:
            nid_ = np.random.randint(0, len(self.negative))
            negative = self.negative[nid_]
        else:
            # 防呆機制：如果 Source Data 裡完全沒有 Label 1 資料，程式會報錯提醒您
            raise ValueError("錯誤：Source Training Data 中找不到任何 Label 為 1 的異常資料，無法作為負樣本！請檢查 cab_data_select.py 或 csv 檔案。")

        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]

        if self.is_cuda:
            # ... (torch 轉換邏輯不變) ...
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            positive = torch.Tensor(positive).float().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).float()
            negative = torch.Tensor(negative).float()
            label = torch.Tensor([label]).long()

        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}
        return sample

    def load_sequence(self):
        if self.split_type == "test":
            # 根據你的最新說明, 測試檔案叫做 "test_data.csv"
            filename = "test_data.csv"
        else:
            # self.subject_id 是 "source_data", self.split_type 是 "train"
            # 結果會是 "source_data_train.csv" (這就是你要的)
            filename = f"{self.subject_id}_{self.split_type}.csv"
        
        path_sequence = os.path.join(self.root_dir, filename)
        if self.verbose: print(f"[HVACDataset] Loading file: {path_sequence}")
        
        try:
            df = pd.read_csv(path_sequence)
        except FileNotFoundError:
            print(f"錯誤：找不到檔案 {path_sequence}")
            print(f"請確保你的檔案名稱正確 (例如: {filename})")
            print(f"並且確認你的 main_HVAC.py 中 id_src 是 'source_data', id_trg 是 'target_data'")
            raise

        # 3. 決定使用哪些特徵欄位 (邏輯不變)
        cols_to_use = []
        if self.feature_columns:
            cols_to_use = self.feature_columns
        else:
            cols_to_use = df.columns[2:-1]

        features = df[cols_to_use].astype(float)
        self.label = df.iloc[:, -1].values

        # 4. 【關鍵】處理標準化
        if self.d_mean is None:
            # 如果沒有傳入 mean/std (只會在 source_train 時發生)
            if self.verbose: print(f"[HVACDataset] Calculating new mean/std from {filename}")
            self.d_mean = np.mean(features.values, axis=0)
            self.d_std = np.std(features.values, axis=0)
            self.d_std[self.d_std==0.0] = 1.0
        else:
            # 如果有傳入 mean/std (val/test/target 時發生)
            if self.verbose: print(f"[HVACDataset] Using provided mean/std.")
            pass # self.d_mean 和 self.d_std 已經被 __init__ 設定

        # 5. 標準化
        self.sequence = (features - self.d_mean) / self.d_std
        self.sequence = self.sequence.values
            
        # --- END: 修正邏輯 ---

    def get_statistic(self):
        # 這個函式現在回傳的是【訓練集】的統計數據
        return self.d_mean, self.d_std

    def convert_to_windows(self, w_size, stride):
        # ... (此函式不變) ...
        windows = []
        wlabels = []
        sz = int((self.sequence.shape[0]-w_size)/stride)
        for i in range(0, sz):
            st = i * stride
            w = self.sequence[st:st+w_size]
            if self.label[st:st+w_size].any() > 0:
                lbl = 1
            else: lbl=0
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)

class HVACDataset_trg(Dataset):
    def __init__(self, root_dir, subject_id, split_type="train", is_cuda=True, verbose=False,
                 feature_columns=None, w_size=10, stride=1,
                 d_mean=None, d_std=None): # <-- 1. 新增 d_mean, d_std 參數
        
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda and torch.cuda.is_available()
        self.verbose = verbose
        self.feature_columns = feature_columns
        self.w = w_size
        self.s = stride

        # 傳入的 d_mean, d_std
        self.d_mean = d_mean
        self.d_std = d_std
        
        self.sequence = None
        self.label = None

        self.load_sequence() # 載入並處理資料
        
        self.sequence , self.label = self.convert_to_windows(self.w, self.s)
        self.positive = self.sequence[self.label == 1]
        self.negative = self.sequence[self.label == 0]

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, id_):
        # ... ( __getitem__ 邏輯不變 ) ...
        sequence = self.sequence[id_]
        pid_ = abs(id_ - np.random.randint(1, 11))
        positive = self.sequence[pid_]
        self.positive = positive
        negative = get_injector(sequence, self.d_mean, self.d_std) # 使用 self.d_mean
        self.negative = negative

        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]

        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            positive = torch.Tensor(positive).float().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).float()
            negative = torch.Tensor(negative).float()
            label = torch.Tensor([label]).long()

        sample = {"sequence": sequence, "sequence_mask": sequence_mask, "positive": positive, "negative": negative, "label": label}
        return sample

    def load_sequence(self):
        if self.split_type == "test":
            # 根據你的最新說明, 測試檔案叫做 "test_data.csv"
            filename = "test_data.csv"
        else:
            # self.subject_id 是 "target_data", self.split_type 是 "train"
            # 結果會是 "target_data_train.csv" (這就是你要的)
            filename = f"{self.subject_id}_{self.split_type}.csv"
        
        path_sequence = os.path.join(self.root_dir, filename)
        if self.verbose: print(f"[HVACDataset_trg] Loading file: {path_sequence}")
        
        try:
            df = pd.read_csv(path_sequence)
        except FileNotFoundError:
            print(f"錯誤：找不到檔案 {path_sequence}")
            print(f"請確保你的檔案名稱正確 (例如: {filename})")
            print(f"並且確認你的 main_HVAC.py 中 id_src 是 'source_data', id_trg 是 'target_data'")
            raise

        # 3. 決定使用哪些特徵欄位 (邏輯不變)
        cols_to_use = []
        if self.feature_columns:
            cols_to_use = self.feature_columns
        else:
            cols_to_use = df.columns[2:-1]

        features = df[cols_to_use].astype(float)
        self.label = df.iloc[:, -1].values

        # 4. 【關鍵】處理標準化
        if self.d_mean is None:
            # target_train 應該【總是】使用 source_train 的 mean/std
            # 所以這裡【不應該】計算新的
            if self.verbose: print(f"[HVACDataset_trg] Warning: d_mean is None. Using self-calculated mean/std.")
            self.d_mean = np.mean(features.values, axis=0)
            self.d_std = np.std(features.values, axis=0)
            self.d_std[self.d_std==0.0] = 1.0
        else:
            if self.verbose: print(f"[HVACDataset_trg] Using provided mean/std.")
            pass # self.d_mean 和 self.d_std 已經被 __init__ 設定

        # 5. 標準化
        self.sequence = (features - self.d_mean) / self.d_std
        self.sequence = self.sequence.values
            
        # --- END: 修正邏輯 ---

    def get_statistic(self):
        return self.d_mean, self.d_std

    def convert_to_windows(self, w_size, stride):
        # ... (此函式不變) ...
        windows = []
        wlabels = []
        sz = int((self.sequence.shape[0]-w_size)/stride)
        for i in range(0, sz):
            st = i * stride
            w = self.sequence[st:st+w_size]
            if self.label[st:st+w_size].any() > 0:
                lbl = 1
            else: lbl=0
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)
# =============================================================================
# END: 修改 HVAC 類別
# =============================================================================

def get_injector(sample_batched, d_mean, d_std):
    sample_batched = (sample_batched * d_std) + d_mean
    injected_window = Injector(sample_batched)
    injected_window.injected_win = (injected_window.injected_win - d_mean) / d_std

    return injected_window.injected_win


def get_output_dim(args):
    output_dim = -1

    if "SMD" in args.path_src:
        output_dim = 1
    elif "MSL" in args.path_src:
        output_dim = 1
    elif "Boiler" in args.path_src:
        output_dim = 1
    elif "HVAC" in args.path_src: # <-- 新增 HVAC 
        output_dim = 1
    else:
        output_dim = 6

    return output_dim

def collate_test(batch):
    #The input is list of dictionaries
    out = {}
    for key in batch[0].keys():
        val = []
        for sample in batch:
            val.append(sample[key])
        val = torch.cat(val, dim=0)
        out[key] = val
    return out