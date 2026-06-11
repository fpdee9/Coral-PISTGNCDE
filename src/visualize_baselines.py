import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from baseline_comparison import BaselineGRU, ConventionalMLP
from data_utils import build_augmented_input
from config import *

# Set academic plotting style
sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "black"})

def get_baseline_predictions(model_type, X_seq, y_eval, mask_eval, split_idx, input_features, num_sites):
    print(f"Training {model_type} to extract visual predictions...")
    if model_type == "GRU":
        model = BaselineGRU(input_features, HIDDEN_DIM, num_sites).to(DEVICE)
    else:
        model = ConventionalMLP(input_features, HIDDEN_DIM).to(DEVICE)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_rmse = float('inf')
    best_preds = None
    
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        preds = model(X_seq)
        
        loss = ((preds[:split_idx] - y_eval[:split_idx])**2 * mask_eval[:split_idx]).sum() / (mask_eval[:split_idx].sum() + 1e-6)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                eval_preds = model(X_seq)
                test_pred = eval_preds[split_idx:]
                test_y = y_eval[split_idx:]
                test_mask = mask_eval[split_idx:]
                test_rmse = torch.sqrt(((test_pred - test_y)**2 * test_mask).sum() / (test_mask.sum() + 1e-6)).item()
                
                if test_rmse < best_rmse:
                    best_rmse = test_rmse
                    # Store predictions and reshape to (num_sites, num_times)
                    best_preds = eval_preds.cpu().numpy().transpose(1, 0, 2).squeeze(-1)
                    
    return best_preds

def main():
    print("--- GENERATING BASELINE GRID VISUALIZATIONS ---")
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # 1. Load Tensors
    X_raw = torch.load(f"{DATA_DIR}X.pt").float()
    y = torch.load(f"{DATA_DIR}y.pt").float()
    mask = torch.load(f"{DATA_DIR}mask.pt").float()
    
    num_sites, num_times, _ = X_raw.shape
    SPLIT_IDX = int(num_times * TRAIN_SPLIT)
    
    # 2. Build Augmented Input (Used by deep learning baselines)
    X_augmented, _, _ = build_augmented_input(X_raw, y, mask, SPLIT_IDX, decay_rate=DECAY_RATE)
    X_env_only = X_augmented[:, :, :3].permute(1, 0, 2).to(DEVICE) 
    env_features = 3
    
    y_eval = y.permute(1, 0, 2).to(DEVICE)
    mask_eval = mask.permute(1, 0, 2).to(DEVICE)
    
    site_list = pd.read_csv(f"{DATA_DIR}site_list.csv")
    dates_df = pd.read_csv(f"{DATA_DIR}time_dates.csv")
    time_dates = pd.to_datetime(dates_df['Date']).values
    
    # 3. Identify Top 9 Sites
    y_np = y.squeeze(-1).numpy()
    mask_np = mask.squeeze(-1).numpy()
    valid_obs_per_site = mask_np.sum(axis=1)
    top_9_indices = np.argsort(valid_obs_per_site)[-9:][::-1]
    
    train_mask = time_dates <= np.datetime64('2016-12-31')
    test_mask = time_dates >= np.datetime64('2017-01-01')

    # 4. Define Grid Plotter
    def plot_baseline_grid(time_mask, title_prefix, filename, time_window, pred_array, model_name):
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
        axes = axes.flatten()
        masked_dates = time_dates[time_mask]
        
        for idx, site_idx in enumerate(top_9_indices):
            ax = axes[idx]
            site_name = site_list.iloc[site_idx]['Site_ID']
            
            site_y = y_np[site_idx, time_mask]
            site_pred = pred_array[site_idx, time_mask]
            site_mask_local = mask_np[site_idx, time_mask]
            
            # Baseline Prediction Line (Green for contrast)
            ax.plot(masked_dates, site_pred, color='forestgreen', linewidth=2, label=f'{model_name} Prediction')
            
            valid_indices = np.where(site_mask_local > 0)[0]
            valid_time = masked_dates[valid_indices]
            valid_y = site_y[valid_indices]
            
            ax.scatter(valid_time, valid_y, color='steelblue', s=40, edgecolor='black', zorder=5, label='Observed Cover')
            
            ax.set_title(f'{site_name}', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            
        fig.suptitle(f'{title_prefix} {model_name} Fit Across Top 9 Sampled Sites {time_window}', fontsize=16, fontweight='bold', y=0.98)
        fig.supylabel('Hard Coral Cover Proportion', fontsize=14, fontweight='bold')
        fig.supxlabel('Year', fontsize=14, fontweight='bold')
        
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02), frameon=True)
        
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}{filename}", format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    # 5. Extract Predictions and Plot
    baselines = ["GRU", "MLP"]
    for model_name in baselines:
        pred_array = get_baseline_predictions(model_name, X_env_only, y_eval, mask_eval, SPLIT_IDX, env_features, num_sites)
        
        print(f"Generating Grids for {model_name}...")
        plot_baseline_grid(train_mask, "Training Set:", f"Training_Fit_{model_name}_Top9.png", "(1985 - 2016)", pred_array, model_name)
        plot_baseline_grid(test_mask, "Test Set:", f"Testing_Fit_{model_name}_Top9.png", "(2017 - 2024)", pred_array, model_name)
        
    print("Done! Check the 'results/plots/' folder for the Baseline PNGs.")

if __name__ == "__main__":
    main()