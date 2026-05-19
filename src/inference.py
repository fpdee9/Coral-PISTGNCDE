import torch
import torchcde
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd            
from coral_model import CoralSTGNCDE
from data_utils import build_augmented_input, normalize_adjacency
from config import *

def main():
    print(f"--- STARTING INFERENCE & BASELINE COMPARISON ON {DEVICE} ---")
    
    # 1. Load Data
    X_raw = torch.load(f"{DATA_DIR}X.pt").float()
    y = torch.load(f"{DATA_DIR}y.pt").float()
    mask = torch.load(f"{DATA_DIR}mask.pt").float()
    adj = torch.load(f"{DATA_DIR}adjacency_matrix.pt").float()
    
    dates_df = pd.read_csv(f"{DATA_DIR}time_dates.csv")
    time_dates = pd.to_datetime(dates_df['Date']).values
    site_list = pd.read_csv(f"{DATA_DIR}site_list.csv")
    
    num_sites, num_times, _ = X_raw.shape
    adj = normalize_adjacency(adj)
    SPLIT_IDX = int(num_times * TRAIN_SPLIT)
    
    # 2. Build Input
    X_augmented, historical_y, _ = build_augmented_input(X_raw, y, mask, SPLIT_IDX, decay_rate=DECAY_RATE)
    num_features = X_augmented.shape[-1]
    
    X_time_first = X_augmented.permute(1, 0, 2)
    X_flat = X_time_first.reshape(num_times, -1).to(DEVICE)
    
    print("Interpolating Environmental Trajectory...")
    coeffs = torchcde.linear_interpolation_coeffs(X_flat)
    
    # 3. Load Model
    print(f"Loading Best Weights from {MODEL_PATH}...")
    model = CoralSTGNCDE(
        num_sites=num_sites,
        input_features=num_features,
        hidden_dim=HIDDEN_DIM,
        output_features=1,
        adj_matrix=adj.to(DEVICE)
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 4. Execute ODE Physics Engine
    print("Running Spatiotemporal ODE Solver...")
    with torch.no_grad():
        pred = model(coeffs)
        
    # 5. Global Accuracy & Baseline Check
    y_test_sparse = y.permute(1, 0, 2).to(DEVICE)
    mask_test_sparse = mask.permute(1, 0, 2).to(DEVICE)
    
    test_pred = pred[SPLIT_IDX:]
    test_y    = y_test_sparse[SPLIT_IDX:]
    test_mask = mask_test_sparse[SPLIT_IDX:]
    
    # AI RMSE
    test_mse = ((test_pred - test_y) ** 2 * test_mask).sum() / (test_mask.sum() + 1e-6)
    ai_test_rmse = torch.sqrt(test_mse)
    
    # Naive Baseline RMSE
    # Predicts that the future will remain exactly the same as the last known observation
    last_known_state = historical_y[:, SPLIT_IDX - 1, :].to(DEVICE) # Shape: (Sites, 1)
    naive_pred = last_known_state.unsqueeze(0).expand(test_pred.shape[0], num_sites, 1)
    naive_mse = ((naive_pred - test_y) ** 2 * test_mask).sum() / (test_mask.sum() + 1e-6)
    naive_test_rmse = torch.sqrt(naive_mse)
    
    print(f"\n--- RESULTS: AI vs BASELINE ---")
    print(f"Naive Persistence Test RMSE : {naive_test_rmse:.4f} (Baseline)")
    print(f"Physics-Informed AI RMSE    : {ai_test_rmse:.4f} (STG-NCDE)")
    
    improvement = ((naive_test_rmse - ai_test_rmse) / naive_test_rmse) * 100
    print(f"Model Improvement           : {improvement:.1f}% better than baseline")
    
    # 6. Visualization: The Two Target Sites
    target_site_names = ['23082S', '22104S']
    
    for site_name in target_site_names:
        # Find the index of the site
        target_site = site_list[site_list['Site_ID'] == site_name].index[0]
        
        site_pred = pred[:, target_site, 0].cpu().numpy()
        site_y = y_test_sparse[:, target_site, 0].cpu().numpy()
        site_mask = mask_test_sparse[:, target_site, 0].cpu().numpy()
        valid_idx = np.where(site_mask > 0)[0]
        
        # Generate two plots: one with DHW, one without.
        for show_dhw in [False, True]:
            plt.figure(figsize=(12, 6))
            
            plt.plot(time_dates, site_pred, color='blue', linewidth=2.5, label='AI ODE Trajectory')
            
            train_idx = valid_idx[valid_idx < SPLIT_IDX]
            plt.plot(time_dates[train_idx], site_y[train_idx], color='black', marker='o', linestyle='', alpha=0.6, label='Training Data (Known)')
            
            test_idx = valid_idx[valid_idx >= SPLIT_IDX]
            plt.plot(time_dates[test_idx], site_y[test_idx], color='red', marker='D', markersize=8, linestyle='', label='Test Data (Unknown Forecast)')
            
            plt.axvline(x=time_dates[SPLIT_IDX], color='red', linestyle='--', linewidth=2, label='Start of Blind Inference')
            
            if show_dhw:
                dhw_raw = X_raw[target_site, :, 1].numpy()
                plt.fill_between(time_dates, 0, dhw_raw * 0.05, color='orange', alpha=0.3, label='DHW Heatwaves')
                title_suffix = "(with DHW Overlay)"
                filename_suffix = "_with_DHW"
            else:
                title_suffix = "(Clean)"
                filename_suffix = "_clean"
            
            plt.title(f"Forecasting Accuracy: Site {site_name} {title_suffix}", fontsize=14, fontweight='bold')
            plt.ylim(-0.05, 1.05)
            plt.ylabel("Coral Cover")
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)
            
            os.makedirs(PLOT_DIR, exist_ok=True)
            save_path = f"{PLOT_DIR}inference_forecast_{site_name}{filename_suffix}.png"
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
        print(f"\nSaved Clean and DHW forecast visualizations for Site {site_name}.")

if __name__ == "__main__":
    main()