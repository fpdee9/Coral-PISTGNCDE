import torch
import torchcde
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.sparse.csgraph import connected_components
from coral_model import CoralSTGNCDE
from data_utils import build_augmented_input, normalize_adjacency
from config import *

def main():
    print("--- GENERATING NEIGHBORHOOD VISUALIZATIONS ---")
    os.makedirs(NEIGHBORHOOD_PLOT_DIR, exist_ok=True)
    
    # 1. Load Data
    X_raw = torch.load(f"{DATA_DIR}X.pt").float()
    y = torch.load(f"{DATA_DIR}y.pt").float()
    mask = torch.load(f"{DATA_DIR}mask.pt").float()
    adj = torch.load(f"{DATA_DIR}adjacency_matrix.pt").float()

    adj = normalize_adjacency(adj)

    site_list = pd.read_csv(f"{DATA_DIR}site_list.csv")
    dates_df = pd.read_csv(f"{DATA_DIR}time_dates.csv")
    
    num_sites, num_times, input_features = X_raw.shape
    time_dates = pd.to_datetime(dates_df['Date']).values
    split_date = time_dates[int(num_times * 0.8)]
    
    # 2. Discover Neighborhoods using Graph Theory
    # Convert normalized adjacency to binary (1 if connected, 0 if not)
    binary_adj = (adj.numpy() > 0).astype(int)
    n_components, labels = connected_components(csgraph=binary_adj, directed=False, return_labels=True)
    
    print(f"Found {n_components} distinct Coral Neighborhoods!")
    
    # 3. Prepare Inputs & Model
    SPLIT_IDX = int(num_times * TRAIN_SPLIT)
    X_augmented, _, _ = build_augmented_input(X_raw, y, mask, SPLIT_IDX, decay_rate=DECAY_RATE)

    input_features = X_augmented.shape[-1]

    X_flat = X_augmented.permute(1, 0, 2).reshape(num_times, -1).to(DEVICE)
    coeffs = torchcde.linear_interpolation_coeffs(X_flat)
    
    model = CoralSTGNCDE(num_sites, input_features, HIDDEN_DIM, 1, adj).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    with torch.no_grad():
        pred = model(coeffs) 
        
    y_np = y.squeeze(-1).numpy()
    pred_np = pred.permute(1, 0, 2).squeeze(-1).cpu().numpy()
    mask_np = mask.squeeze(-1).numpy()
    X_env = X_raw.numpy()
    
    # 4. Plot each Neighborhood
    for cluster_id in range(n_components):
        # Find all sites belonging to this specific neighborhood
        cluster_site_indices = np.where(labels == cluster_id)[0]
        
       # Skip isolated "orphan" reefs (neighborhoods with only 1 site)
        if len(cluster_site_indices) < 2:
            continue
            
        print(f"\nPlotting Neighborhood {cluster_id} ({len(cluster_site_indices)} reefs)")

        # Extract and print the exact list of reefs in this neighborhood
        neighborhood_sites = site_list.iloc[cluster_site_indices]['Site_ID'].tolist()
        for site in neighborhood_sites:
            print(f"   - {site}")
        print("-" * 40)
        
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # Plot every reef in the neighborhood
        for idx in cluster_site_indices:
            site_name = site_list.iloc[idx]['Site_ID']
            site_pred = pred_np[idx, :]
            site_y = y_np[idx, :]
            site_mask = mask_np[idx, :]
            
            valid_indices = np.where(site_mask > 0)[0]
            valid_time = time_dates[valid_indices] 
            valid_y = site_y[valid_indices]
            
            # AI Predictions (Blue Swarm)
            ax1.plot(time_dates, site_pred, color='blue', alpha=0.4, linewidth=1.5)
            # Biological Ground Truth (Red Swarm)
            ax1.plot(valid_time, valid_y, color='red', marker='o', markersize=3, linestyle='-', alpha=0.4)
            
        ax1.axvline(x=split_date, color='black', linestyle='--', alpha=0.8, label='Train / Test Split')
        
        # --- SECONDARY AXIS (Average Neighborhood DHW) ---
        ax2 = ax1.twinx()
        # Calculate the average DHW for the entire neighborhood
        avg_dhw = np.mean(X_env[cluster_site_indices, :, 1], axis=0) 
        ax2.plot(time_dates, avg_dhw, color='darkorange', label='Avg Neighborhood DHW', linewidth=2.0, alpha=0.9)
        ax2.set_ylabel("DHW (°C-weeks)", color='darkorange', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='darkorange')
        ax2.set_ylim(0, max(np.max(avg_dhw) * 2.5, 5))
        
        # Formatting
        ax1.set_title(f"Metapopulation Dynamics: Neighborhood {cluster_id} ({len(cluster_site_indices)} Connected Reefs)", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Year", fontweight='bold')
        ax1.set_ylabel("Coral Cover (0.0 - 1.0)", color='blue', fontweight='bold')
        ax1.set_ylim(-0.05, 1.05)
        plt.xlim(time_dates[0], time_dates[-1])
        ax1.grid(True, alpha=0.3)
        
        plt.savefig(f"{NEIGHBORHOOD_PLOT_DIR}Neighborhood_{cluster_id}.png", bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()