import torch
import torchcde
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from coral_model import CoralSTGNCDE
from baseline_comparison import BaselineGRU, ConventionalMLP
from data_utils import build_augmented_input, normalize_adjacency
from config import *

# Set academic plotting style
sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "black"})

def get_naive_predictions(y_np, mask_np, split_idx):
    """
    Computes the Naive Persistence baseline.
    For the test set, it persists the very last observation from the training set.
    """
    num_sites, num_times = y_np.shape
    naive_preds = np.zeros_like(y_np)
    
    for s in range(num_sites):
        last_val = np.nan
        for t in range(num_times):
            # Assign the last known value (if it exists, else 0 or first obs)
            naive_preds[s, t] = last_val if not np.isnan(last_val) else 0.0
            
            # Update the last known value IF we have an observation AND we are in the training set
            if mask_np[s, t] > 0 and t < split_idx:
                last_val = y_np[s, t]
                
    return naive_preds

def get_baseline_predictions(model_type, X_seq, y_eval, mask_eval, split_idx, input_features, num_sites):
    print(f"  > Fast-training {model_type} to extract trajectory...")
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
                    best_preds = eval_preds.cpu().numpy().transpose(1, 0, 2).squeeze(-1)
                    
    return best_preds

def main():
    print("=== GENERATING MASTER COMBINED GRID VISUALIZATIONS ===")
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # 1. Load Data
    print("Loading Data and Tensors...")
    X_raw = torch.load(f"{DATA_DIR}X.pt").float()
    y = torch.load(f"{DATA_DIR}y.pt").float()
    mask = torch.load(f"{DATA_DIR}mask.pt").float()
    adj = normalize_adjacency(torch.load(f"{DATA_DIR}adjacency_matrix.pt").float())

    site_list = pd.read_csv(f"{DATA_DIR}site_list.csv")
    dates_df = pd.read_csv(f"{DATA_DIR}time_dates.csv")
    time_dates = pd.to_datetime(dates_df['Date']).values
    
    num_sites, num_times, _ = X_raw.shape
    SPLIT_IDX = int(num_times * TRAIN_SPLIT)
    
    # 2. Extract Ground Truth Arrays
    y_np = y.squeeze(-1).numpy()
    mask_np = mask.squeeze(-1).numpy()
    
    # 3. Generate Predictions: Naive
    print("Extracting Naive Baseline...")
    pred_naive = get_naive_predictions(y_np, mask_np, SPLIT_IDX)
    
    # 4. Generate Predictions: Deep Learning Baselines (MLP & GRU)
    print("Extracting Discrete DL Baselines...")
    X_augmented, _, _ = build_augmented_input(X_raw, y, mask, SPLIT_IDX, decay_rate=DECAY_RATE)
    X_env_only = X_augmented[:, :, :3].permute(1, 0, 2).to(DEVICE) 
    y_eval = y.permute(1, 0, 2).to(DEVICE)
    mask_eval = mask.permute(1, 0, 2).to(DEVICE)
    
    pred_mlp = get_baseline_predictions("MLP", X_env_only, y_eval, mask_eval, SPLIT_IDX, 3, num_sites)
    pred_gru = get_baseline_predictions("GRU", X_env_only, y_eval, mask_eval, SPLIT_IDX, 3, num_sites)
    
    # 5. Generate Predictions: PISTGNCDE
    print("Extracting PISTGNCDE Continuous Trajectory...")
    input_features_ncde = X_augmented.shape[-1]
    X_flat = X_augmented.permute(1, 0, 2).reshape(num_times, -1).to(DEVICE)
    coeffs = torchcde.linear_interpolation_coeffs(X_flat)
    
    model = CoralSTGNCDE(num_sites, input_features_ncde, HIDDEN_DIM, 1, adj.to(DEVICE)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_LATEST_PATH, map_location=DEVICE)) # Note: using MODEL_LATEST_PATH per standard config
    model.eval()
    with torch.no_grad():
        pred_ncde = model(coeffs).permute(1, 0, 2).squeeze(-1).cpu().numpy()
        
    # 6. Identify Top 9 Sites
    valid_obs_per_site = mask_np.sum(axis=1)
    top_9_indices = np.argsort(valid_obs_per_site)[-9:][::-1]
    
    train_mask = time_dates <= np.datetime64('2016-12-31')
    test_mask = time_dates >= np.datetime64('2017-01-01')

    # 7. Define Master Plotter
    def plot_combined_grid(time_mask, title_prefix, filename, time_window):
        fig, axes = plt.subplots(3, 3, figsize=(16, 12.5), sharex=True, sharey=True)
        axes = axes.flatten()
        masked_dates = time_dates[time_mask]
        
        for idx, site_idx in enumerate(top_9_indices):
            ax = axes[idx]
            site_name = site_list.iloc[site_idx]['Site_ID']
            
            # Sub-slice data
            site_y = y_np[site_idx, time_mask]
            site_mask_local = mask_np[site_idx, time_mask]
            
            # Plot Baselines
            ax.plot(masked_dates, pred_naive[site_idx, time_mask], color='darkviolet', linewidth=1.5, linestyle='--', alpha=0.8, label='Naive Prediction')
            ax.plot(masked_dates, pred_mlp[site_idx, time_mask], color='royalblue', linewidth=1.5, alpha=0.8, label='MLP Prediction')
            ax.plot(masked_dates, pred_gru[site_idx, time_mask], color='forestgreen', linewidth=1.5, alpha=0.9, label='GRU Prediction')
            
            # Plot Proposed Model (Thicker, highest z-order)
            ax.plot(masked_dates, pred_ncde[site_idx, time_mask], color='firebrick', linewidth=2.5, zorder=4, label='PISTGNCDE (Proposed)')
            
            # Plot Ground Truth Observations
            valid_indices = np.where(site_mask_local > 0)[0]
            ax.scatter(masked_dates[valid_indices], site_y[valid_indices], color='black', s=40, edgecolor='white', linewidth=0.5, zorder=5, label='Observed Cover')
            
            ax.set_title(f'{site_name}', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            
        fig.suptitle(f'{title_prefix} Model Comparison Across 9 Randomly Selected Sites {time_window}', fontsize=16, fontweight='bold', y=0.98)
        fig.supylabel('Hard Coral Cover Proportion', fontsize=14, fontweight='bold')
        fig.supxlabel('Year', fontsize=14, fontweight='bold')
        
        # Unified Legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}{filename}", format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f"{PLOT_DIR}{filename.replace('.pdf', '.png')}", format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    # 8. Execute the plots
    print("Generating Training Set Master Grid...")
    plot_combined_grid(train_mask, "Training Set:", "Master_Training_Fit_Top9.pdf", "(1985 - 2016)")
    
    print("Generating Test Set Master Grid...")
    plot_combined_grid(test_mask, "Test Set:", "Master_Testing_Fit_Top9.pdf", "(2017 - 2024)")
    
    print("Done! Check the 'results/plots/' folder for the Master PDFs and PNGs.")

if __name__ == "__main__":
    main()