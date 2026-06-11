import torch
import torchcde
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from coral_model import CoralSTGNCDE
from data_utils import build_augmented_input, normalize_adjacency
from config import *

# Set academic plotting style
sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "black"})

def main():
    print("--- GENERATING 9 GRID VISUALIZATIONS ---")
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # 1. Load Data
    print("Loading Data and Tensors...")
    if not os.path.exists(f"{DATA_DIR}X.pt"):
        print("Error: processed data not found.")
        return

    X_raw = torch.load(f"{DATA_DIR}X.pt").float()
    y = torch.load(f"{DATA_DIR}y.pt").float()
    mask = torch.load(f"{DATA_DIR}mask.pt").float()
    adj = torch.load(f"{DATA_DIR}adjacency_matrix.pt").float()
    adj = normalize_adjacency(adj)

    site_list = pd.read_csv(f"{DATA_DIR}site_list.csv")
    dates_df = pd.read_csv(f"{DATA_DIR}time_dates.csv")
    time_dates = pd.to_datetime(dates_df['Date']).values
    
    num_sites, num_times, input_features = X_raw.shape
    
    # 2. Prepare Inputs and Control Path for the NCDE
    print("Building Continuous Control Path...")
    SPLIT_IDX = int(num_times * TRAIN_SPLIT)
    X_augmented, _, _ = build_augmented_input(X_raw, y, mask, SPLIT_IDX, decay_rate=DECAY_RATE)

    input_features = X_augmented.shape[-1]
    X_time_first = X_augmented.permute(1, 0, 2)
    X_flat = X_time_first.reshape(num_times, -1).to(DEVICE)
    
    coeffs = torchcde.linear_interpolation_coeffs(X_flat)
    
    # 3. Load Model
    print(f"Loading Model: {MODEL_LATEST_PATH}...")
    model = CoralSTGNCDE(
        num_sites=num_sites,
        input_features=input_features,
        hidden_dim=HIDDEN_DIM,
        output_features=1,
        adj_matrix=adj
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_LATEST_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: {MODEL_LATEST_PATH} not found.")
        return
        
    model.eval()
    
    # 4. Run Inference
    print("Running Inference...")
    with torch.no_grad():
        pred = model(coeffs) 
    
    # Convert tensors back to numpy arrays for plotting
    y_np = y.squeeze(-1).numpy()
    pred_np = pred.permute(1, 0, 2).squeeze(-1).cpu().numpy()
    mask_np = mask.squeeze(-1).numpy()
    
    # 5. Identify the 9 Sites with the most biological data
    print("Calculating 9 Randomly Selected Sites...")
    valid_obs_per_site = mask_np.sum(axis=1)
    top_9_indices = np.argsort(valid_obs_per_site)[-9:][::-1]
    
    # 6. Create boolean masks for the temporal split
    train_mask = time_dates <= np.datetime64('2016-12-31')
    test_mask = time_dates >= np.datetime64('2017-01-01')

    # 7. Define the Grid Plotting Function
    def plot_grid(time_mask, title_prefix, filename, time_window):
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
        axes = axes.flatten()
        
        masked_dates = time_dates[time_mask]
        
        for idx, site_idx in enumerate(top_9_indices):
            ax = axes[idx]
            site_name = site_list.iloc[site_idx]['Site_ID']
            
            # Isolate the data for this specific site and time period
            site_y = y_np[site_idx, time_mask]
            site_pred = pred_np[site_idx, time_mask]
            site_mask_local = mask_np[site_idx, time_mask]
            
            # Plot the Continuous AI Prediction
            ax.plot(masked_dates, site_pred, color='firebrick', linewidth=2, label='PISTGNCDE Prediction')
            
            # Filter ground truth to only plot points where mask == 1
            valid_indices = np.where(site_mask_local > 0)[0]
            valid_time = masked_dates[valid_indices]
            valid_y = site_y[valid_indices]
            
            # Plot Discrete Diver Surveys
            ax.scatter(valid_time, valid_y, color='steelblue', s=40, edgecolor='black', zorder=5, label='Observed Cover')
            
            ax.set_title(f'{site_name}', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1) # Coral cover is bounded [0, 1]
            ax.tick_params(axis='x', rotation=45)
            
        # Global formatting
        fig.suptitle(f'{title_prefix} Model Fit Across 9 Randomly Sampled Sites {time_window}', fontsize=16, fontweight='bold', y=0.98)
        fig.supylabel('Hard Coral Cover Proportion', fontsize=14, fontweight='bold')
        fig.supxlabel('Year', fontsize=14, fontweight='bold')
        
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02), frameon=True)
        
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}{filename}", format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    # 8. Execute the plots
    print("Generating Training Set Grid...")
    plot_grid(train_mask, "Training Set:", "Training_Fit_Top9.png", "(1985 - 2016)")
    
    print("Generating Test Set Grid...")
    plot_grid(test_mask, "Test Set:", "Testing_Fit_Top9.png", "(2017 - 2024)")
    
    print("Done! Check the 'results/plots/' folder.")

if __name__ == "__main__":
    main()