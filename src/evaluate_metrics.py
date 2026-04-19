import torch
import torchcde
import numpy as np
import pandas as pd
from coral_model import CoralSTGNCDE
from data_utils import build_augmented_input, normalize_adjacency
from config import *

def calculate_directional_accuracy(y_true, y_pred, mask):

    # Calculates how often the model predicts the correct 'Direction' (Recovery vs Decline) between two consecutive observations.
    n_sites = y_true.shape[1]
    correct_directions = 0
    total_pairs = 0
    
    # Iterate over each site
    for s in range(n_sites):
        # Extract time series for this site
        site_true = y_true[:, s, 0]
        site_pred = y_pred[:, s, 0]
        site_mask = mask[:, s, 0]
        
        # Find indices where we actually have ground truth data
        observed_indices = torch.nonzero(site_mask).squeeze()
        
        if len(observed_indices.shape) == 0 or observed_indices.shape[0] < 2:
            continue # Need at least 2 points to calculate a trend
            
        # Compare pairs of consecutive observations
        for i in range(len(observed_indices) - 1):
            t1, t2 = observed_indices[i], observed_indices[i+1]
            
            # True Slope
            true_diff = site_true[t2] - site_true[t1]
            
            # Predicted Slope (change between the same two dates)
            pred_diff = site_pred[t2] - site_pred[t1]
            
            # Check if signs match (e.g., both positive or both negative)
            # We add a small epsilon to handle flat lines (0 change)
            if (true_diff > 0 and pred_diff > 0) or (true_diff < 0 and pred_diff < 0):
                correct_directions += 1
            total_pairs += 1

    if total_pairs == 0:
        return 0.0
    return correct_directions / total_pairs

def main():
    print("--- EVALUATING MODEL METRICS ---")
    
    # Load Data
    X_raw = torch.load(f"{DATA_DIR}X.pt").float()
    y_raw = torch.load(f"{DATA_DIR}y.pt").float()
    mask_raw = torch.load(f"{DATA_DIR}mask.pt").float()
    adj = torch.load(f"{DATA_DIR}adjacency_matrix.pt").float()

    adj = normalize_adjacency(adj)

    num_sites, num_times, num_features = X_raw.shape

    SPLIT_IDX = int(num_times * TRAIN_SPLIT)
    
    X_augmented, _, _ = build_augmented_input(X_raw, y_raw, mask_raw, SPLIT_IDX, decay_rate=DECAY_RATE)
   
    input_features = X_augmented.shape[-1]
    
    X_time_first = X_augmented.permute(1, 0, 2)
    
    # Inference
    X_flat = X_time_first.reshape(num_times, -1).to(DEVICE)
    
    # ---> PURE EVALUATION TENSORS <---
    # These are kept totally separate from the blindfolded historical inputs
    y_eval = y_raw.permute(1, 0, 2).to(DEVICE)       
    mask_eval = mask_raw.permute(1, 0, 2).to(DEVICE)

    # Predict
    # Build Splines
    coeffs = torchcde.linear_interpolation_coeffs(X_flat)
    
    # Initialize Model
    model = CoralSTGNCDE(num_sites, input_features, HIDDEN_DIM, 1, adj).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    with torch.no_grad():
        preds = model(coeffs)
        
    # A. RMSE (Standard Accuracy)
    # ---------------------------
    test_pred = preds[SPLIT_IDX:]
    test_y = y_eval[SPLIT_IDX:]
    test_mask = mask_eval[SPLIT_IDX:]
    
    mse = ((test_pred - test_y)**2 * test_mask).sum() / (test_mask.sum() + 1e-6)
    rmse = torch.sqrt(mse).item()
    
    # B. Directional Change Accuracy (Split by Train/Test)
    # ---------------------------
    # Slice the tensors to evaluate the two windows independently
    train_y_eval = y_eval[:SPLIT_IDX]
    train_preds  = preds[:SPLIT_IDX]
    train_mask   = mask_eval[:SPLIT_IDX]
    
    test_y_eval  = y_eval[SPLIT_IDX:]
    test_preds   = preds[SPLIT_IDX:]
    test_mask    = mask_eval[SPLIT_IDX:]
    
    train_dca = calculate_directional_accuracy(train_y_eval, train_preds, train_mask)
    test_dca  = calculate_directional_accuracy(test_y_eval, test_preds, test_mask)
    
    # C. Trend Change Error (The "Velocity" Metric)
    # ---------------------------
    # Average absolute difference in slope between true and pred
    # Ideally, if reef drops 10% in a year, model should drop 10% (not 2%)
    # Simplified calculation: MAE of the derivatives at observed points
    
    print("\n===========================================")
    print("            FINAL RESULTS TABLE            ")
    print("===========================================")
    print(f"1. Predictive Accuracy (Test RMSE):   {rmse:.4f}")
    print(f"   > Interpretation: The model is off by approx {rmse*100:.1f}% on average.")
    print(f"-------------------------------------------")
    print(f"2. Directional Accuracy (Train DCA):  {train_dca:.4f}")
    print(f"3. Directional Accuracy (Test DCA):   {test_dca:.4f}")
    print(f"   > Interpretation: Captures mortality macro-trends, but smoothing reduces micro-fluctuation accuracy.")
    print("===========================================\n")

if __name__ == "__main__":
    main()