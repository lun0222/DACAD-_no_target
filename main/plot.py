import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker # 匯入 ticker

def parse_log(log_file_path):
    """
    解析日誌檔案並提取損失值。
    
    日誌格式範例:
    Epoch: [0][10/75]	L_Src_Sup 1.9952e+00 (1.9952e+00)	L_Trg_Inj 1.9919e+00 (1.9941e+00)	Loss Disc 7.1908e-01 (7.2222e-01)	Loss Pred 1.5814e+00 (1.6022e+00)	Loss TOTAL 2.3430e+00 (2.3635e+00)
    """
    
    # ==================================================================
    # START: 修改 Regex
    # ==================================================================
    # 我們需要捕獲 (epoch), (batch), (total_batches) 以及括號中的平均值
    pattern = re.compile(
        r"Epoch: \[(\d+)\]\[(\d+)/(\d+)\]\s+"  # Group 1: epoch, Group 2: batch, Group 3: total_batches
        r"L_Src_Sup .*? \((.*?)\)\s+"      # Group 4: l_src_sup
        r"L_Trg_Inj .*? \((.*?)\)\s+"      # Group 5: l_trg_inj
        r"Loss Disc .*? \((.*?)\)\s+"      # Group 6: l_disc
        r"Loss Pred .*? \((.*?)\)\s+"      # Group 7: l_pred
        r"Loss TOTAL .*? \((.*?)\)"       # Group 8: l_total
    )
    # ==================================================================
    # END: 修改 Regex
    # ==================================================================
    
    data = []
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                
                # ==================================================================
                # START: 新增判斷式
                # ==================================================================
                if match:
                    epoch = int(match.group(1))
                    batch = int(match.group(2))
                    total_batches = int(match.group(3))

                    # 檢查這是否是該 epoch 的最後一個 batch
                    if batch == total_batches:
                        l_src_sup = float(match.group(4))
                        l_trg_inj = float(match.group(5))
                        l_disc = float(match.group(6))
                        l_pred = float(match.group(7))
                        l_total = float(match.group(8))
                        
                        data.append({
                            "epoch": epoch, # X 軸將使用 epoch
                            "L_Src_Sup": l_src_sup,
                            "L_Trg_Inj": l_trg_inj,
                            "Loss_Disc": l_disc,
                            "Loss_Pred": l_pred,
                            "Loss_TOTAL": l_total
                        })
                # ==================================================================
                # END: 新增判斷式
                # ==================================================================
                    
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {log_file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"解析檔案時發生錯誤 {log_file_path}: {e}")
        return pd.DataFrame()

    return pd.DataFrame(data)

def plot_loss_curves(df, plot_name, output_dir):
    """
    根據 DataFrame 繪製 4 種主要損失曲線並儲存。
    """
    if df.empty:
        print(f"沒有資料可供繪圖： {plot_name} (可能是因為 epoch 未完成或 log 格式錯誤)")
        return

    plt.figure(figsize=(14, 8))
    
    # ==================================================================
    # START: 修改 X 軸
    # ==================================================================
    # 繪製您關心的 4 種損失
    plt.plot(df["epoch"], df["L_Src_Sup"], label="L_Src_Sup (源域監督損失)", alpha=0.8, marker='o')
    plt.plot(df["epoch"], df["L_Trg_Inj"], label="L_Trg_Inj (目標域注入損失)", alpha=0.8, marker='o')
    plt.plot(df["epoch"], df["Loss_Disc"], label="Loss_Disc (辨別器損失)", alpha=0.8, marker='o')
    plt.plot(df["epoch"], df["Loss_Pred"], label="Loss_Pred (SVDD 預測損失)", alpha=0.8, marker='o')
    
    plt.title(f'4 種主要 Loss 曲線 (End-of-Epoch) - {plot_name}')
    plt.xlabel('訓練週期 (Epoch)') # <-- X 軸標籤已修改
    # ==================================================================
    # END: 修改 X 軸
    # ==================================================================
    
    plt.ylabel('損失值 (Loss Value)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # 設定 y 軸從 0 開始，更容易比較
    plt.ylim(bottom=0)
    
    # 確保 X 軸只顯示整數 (例如 0, 1, 2, ... 而不是 0.0, 0.5, 1.0)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 儲存圖片
    output_filename = f"{plot_name}.png"
    save_path = os.path.join(output_dir, output_filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"圖表已儲存至： {save_path}")

def main():
    # 1. 設定儲存圖表的資料夾
    #    (假設此腳本 plot_loss.py 儲存在 'main' 資料夾中)
    output_dir = 'D:/DACAD/results/HVAC/source_data-target_data' 
    
    # 2. 設定您要處理的日誌檔案
    #    'log_path': 日誌檔案的實際路徑
    #    'plot_name': 儲存的圖片檔名 (不含 .png)
    files_to_plot = [
        {
            "log_path": "D:/DACAD/results/HVAC/source_data-target_data/train.log", #
            "plot_name": "source_data-target_data_Epoch_Losses"
        },
    ]

    # 確保輸出資料夾存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已建立資料夾: {output_dir}")
        
    for file_info in files_to_plot:
        log_path = file_info["log_path"]
        plot_name = file_info["plot_name"]
        
        print(f"正在處理 {log_path}...")
        
        df = parse_log(log_path)
        
        # 呼叫繪圖函數
        plot_loss_curves(df, plot_name, output_dir)
        
    print("所有繪圖已完成。")

if __name__ == "__main__":
    # 確保 matplotlib 可以處理中文字體 (如果需要)
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Heiti TC']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"無法設定中文字體，可能顯示為方框: {e}")
        
    main()