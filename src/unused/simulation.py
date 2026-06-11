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
    print(f"--- SYNTHETIC STRESS TEST ---")
    
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
    X_augmented, _, _ = build_augmented_input(X_raw, y, mask, SPLIT_IDX, decay_rate=DECAY_RATE)
    
    # ---------------------------------------------------------
    # THE SYNTHETIC INJECTION (THE STRESS TEST)
    # ---------------------------------------------------------
    target_site_name = '23082S'
    target_site = site_list[site_list['Site_ID'] == target_site_name].index[0]
    
    # Create a cloned, synthetic tensor
    X_synthetic = X_augmented.clone()
    
    # First Strike (2018)
    injection_date_1 = np.datetime64('2018-01-01')
    idx_1 = np.argmin(np.abs(time_dates - injection_date_1))
    X_synthetic[target_site, idx_1:idx_1+3, 5] = 8.0
    X_synthetic[target_site, idx_1:idx_1+3, 1] = 3.0
    
    # Second Strike (2020)
    injection_date_2 = np.datetime64('2020-01-01')
    idx_2 = np.argmin(np.abs(time_dates - injection_date_2))
    X_synthetic[target_site, idx_2:idx_2+3, 5] = 8.0
    X_synthetic[target_site, idx_2:idx_2+3, 1] = 3.0
    
    # Extract the raw DHW for plotting
    dhw_raw_synthetic = X_synthetic[target_site, :, 5].numpy()
    
    num_features = X_synthetic.shape[-1]
    
    # Prepare both original and synthetic tensors for the ODE solver
    X_orig_flat = X_augmented.permute(1, 0, 2).reshape(num_times, -1).to(DEVICE)
    X_synth_flat = X_synthetic.permute(1, 0, 2).reshape(num_times, -1).to(DEVICE)
    
    coeffs_orig = torchcde.linear_interpolation_coeffs(X_orig_flat)
    coeffs_synth = torchcde.linear_interpolation_coeffs(X_synth_flat)
    
    # 3. Load Model
    model = CoralSTGNCDE(
        num_sites=num_sites, input_features=num_features, hidden_dim=HIDDEN_DIM, 
        output_features=1, adj_matrix=adj.to(DEVICE)
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 4. Run
    print(f"Simulating Original Trajectory vs Synthetic Heatwave at {injection_date_1}...")
    with torch.no_grad():
        pred_orig = model(coeffs_orig)[:, target_site, 0].cpu().numpy()
        pred_synth = model(coeffs_synth)[:, target_site, 0].cpu().numpy()
        
    # 5. Visualization
    plt.figure(figsize=(12, 6))
    
    # Plot Original Trajectory
    plt.plot(time_dates, pred_orig, color='blue', linewidth=2, label='Original Forecast')
    
    # Plot Synthetic Trajectory
    plt.plot(time_dates, pred_synth, color='red', linewidth=2.5, linestyle='--', label='Simulated Forecast (With Synthetic Heatwave)')
    
    # Fill the synthetic DHW spike
    plt.fill_between(time_dates, 0, dhw_raw_synthetic * 0.05, color='orange', alpha=0.4, label='Synthetic DHW Spike (8.0°C-weeks)')
    
    plt.axvline(x=time_dates[SPLIT_IDX], color='black', linestyle=':', label='Start of Blind Inference')
    
    plt.title(f"Synthetic Stress Test: Site {target_site_name}", fontsize=14, fontweight='bold')
    plt.ylim(-0.05, 1.05)
    plt.ylabel("Coral Cover / Normalized DHW")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    os.makedirs(PLOT_DIR, exist_ok=True)
    save_path = f"{PLOT_DIR}synthetic_stress_test_{target_site_name}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Synthetic Stress Test visualization to: {save_path}")

if __name__ == "__main__":
    main()