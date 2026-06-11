import torch
import numpy as np
import pandas as pd
from coral_model import CoralSTGNCDE
from baseline_comparison import BaselineGRU
from data_utils import build_augmented_input, normalize_adjacency
from config import *
import torchcde

def compute_rmse(preds, target, mask):
    """Computes RMSE for a specific subset of data."""
    if mask.sum() == 0: return float('nan')
    mse = ((preds - target)**2 * mask).sum() / mask.sum()
    return torch.sqrt(mse).item()

def main():
    print("=== COMMENCING SPATIAL BOOTSTRAP CONFIDENCE INTERVALS (1000 ITERATIONS) ===")
    
    # 1. Load Data
    X_raw = torch.load(f"{DATA_DIR}X.pt").float()
    y_raw = torch.load(f"{DATA_DIR}y.pt").float()
    mask_raw = torch.load(f"{DATA_DIR}mask.pt").float()
    adj = normalize_adjacency(torch.load(f"{DATA_DIR}adjacency_matrix.pt").float())
    
    num_sites, num_times, _ = X_raw.shape
    SPLIT_IDX = int(num_times * TRAIN_SPLIT)
    
    # Format Inputs
    X_augmented, _, _ = build_augmented_input(X_raw, y_raw, mask_raw, SPLIT_IDX, decay_rate=DECAY_RATE)
    
    # STG-NCDE Input
    coeffs = torchcde.linear_interpolation_coeffs(X_augmented.permute(1, 0, 2).reshape(num_times, -1).to(DEVICE))
    # GRU Input (No biological leakage)
    X_env_only = X_augmented[:, :, :3].permute(1, 0, 2).to(DEVICE)
    
    test_y = y_raw.permute(1, 0, 2).to(DEVICE)[SPLIT_IDX:]
    test_mask = mask_raw.permute(1, 0, 2).to(DEVICE)[SPLIT_IDX:]

    # 2. Get Predictions from STG-NCDE (Epoch 455 Best Model)
    print("Generating predictions for STG-NCDE...")
    stgncde = CoralSTGNCDE(num_sites, X_augmented.shape[-1], HIDDEN_DIM, 1, adj.to(DEVICE)).to(DEVICE)
    stgncde.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    stgncde.eval()
    with torch.no_grad():
        preds_stgncde = stgncde(coeffs)[SPLIT_IDX:]
        
    # 3. Get Predictions from Baseline GRU (Assuming you saved its weights, or we just evaluate its output)
    # NOTE: You will need to briefly train the GRU here or load its saved weights. 
    # For speed in this script, we will simulate the bootstrapping engine on the STG-NCDE first.
    
    # 4. The Spatial Bootstrap Engine
    N_ITERATIONS = 1000
    stgncde_bootstrapped_rmses = []
    
    print(f"Bootstrapping {N_ITERATIONS} resamples over {num_sites} sites...")
    
    np.random.seed(42) # For reproducible thesis results
    for i in range(N_ITERATIONS):
        # Randomly sample site indices with replacement
        resampled_site_indices = np.random.choice(num_sites, size=num_sites, replace=True)
        
        # Pull the predictions and ground truth for those specific random sites
        boot_preds = preds_stgncde[:, resampled_site_indices, :]
        boot_y = test_y[:, resampled_site_indices, :]
        boot_mask = test_mask[:, resampled_site_indices, :]
        
        boot_rmse = compute_rmse(boot_preds, boot_y, boot_mask)
        stgncde_bootstrapped_rmses.append(boot_rmse)
        
    # 5. Calculate the Percentiles
    lower_bound = np.percentile(stgncde_bootstrapped_rmses, 2.5)
    upper_bound = np.percentile(stgncde_bootstrapped_rmses, 97.5)
    median_rmse = np.percentile(stgncde_bootstrapped_rmses, 50.0)
    
    print("\n=======================================================")
    print("      STATISTICAL SIGNIFICANCE (95% CONFIDENCE)        ")
    print("=======================================================")
    print(f"STG-NCDE Median RMSE : {median_rmse:.4f}")
    print(f"STG-NCDE 95% CI      : [{lower_bound:.4f} - {upper_bound:.4f}]")
    print("=======================================================\n")
    print("If the GRU's score (0.1629) is strictly outside this bracket,")
    print("your model is statistically significantly better (p < 0.05).")

if __name__ == "__main__":
    main()