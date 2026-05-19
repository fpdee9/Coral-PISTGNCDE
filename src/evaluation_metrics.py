import torch
import torchcde
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import pandas as pd            
from coral_model import CoralSTGNCDE
from data_utils import build_augmented_input, normalize_adjacency
from config import *


def calculate_trend_metrics(y_true, y_pred, mask):
    n_sites = y_true.shape[1]
    correct_directions, total_direction_pairs = 0, 0
    missed_trend_changes, total_trend_changes = 0, 0
    false_alarms, stable_trends = 0, 0
    
    for s in range(n_sites):
        site_true, site_pred, site_mask = y_true[:, s, 0], y_pred[:, s, 0], mask[:, s, 0]
        observed_indices = torch.nonzero(site_mask).squeeze()
        
        if len(observed_indices.shape) == 0 or observed_indices.shape[0] < 3: continue 
            
        for i in range(1, len(observed_indices)):
            t_curr, t_prev = observed_indices[i], observed_indices[i-1]
            true_diff = site_true[t_curr] - site_true[t_prev]
            pred_diff = site_pred[t_curr] - site_pred[t_prev]
            
            if (true_diff > 0 and pred_diff > 0) or (true_diff < 0 and pred_diff < 0): correct_directions += 1
            total_direction_pairs += 1
            
            if i >= 2:
                t_prev2 = observed_indices[i-2]
                prev_true_diff = site_true[t_prev] - site_true[t_prev2]
                actual_sign, prev_actual_sign, pred_sign = torch.sign(true_diff), torch.sign(prev_true_diff), torch.sign(pred_diff)
                
                if actual_sign != prev_actual_sign and actual_sign != 0 and prev_actual_sign != 0:
                    total_trend_changes += 1
                    if pred_sign != actual_sign: missed_trend_changes += 1
                elif actual_sign == prev_actual_sign and actual_sign != 0:
                    stable_trends += 1
                    if pred_sign != actual_sign: false_alarms += 1

    dca = (correct_directions / max(total_direction_pairs, 1)) * 100
    tce = (missed_trend_changes / max(total_trend_changes, 1)) * 100
    far = (false_alarms / max(stable_trends, 1)) * 100
    return dca, tce, far

def calculate_pearson_r(y_true, y_pred, mask):
    true_flat, pred_flat = y_true[mask > 0], y_pred[mask > 0]
    if len(true_flat) < 2: return 0.0
    vx, vy = true_flat - torch.mean(true_flat), pred_flat - torch.mean(pred_flat)
    return (torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))).item()

def calculate_bleaching_rmse(y_true, y_pred, mask, dhw_tensor, threshold=1.0):
    bleach_mask = (dhw_tensor > threshold).unsqueeze(-1) * mask
    if bleach_mask.sum() == 0: return float('nan')
    mse = ((y_pred - y_true)**2 * bleach_mask).sum() / bleach_mask.sum()
    return torch.sqrt(mse).item()

def main():
    print(f"--- STARTING UNIFIED INFERENCE & EVALUATION ON {DEVICE} ---")
    
    X_raw, y_raw, mask_raw = torch.load(f"{DATA_DIR}X.pt").float(), torch.load(f"{DATA_DIR}y.pt").float(), torch.load(f"{DATA_DIR}mask.pt").float()
    adj = normalize_adjacency(torch.load(f"{DATA_DIR}adjacency_matrix.pt").float())
    dates_df = pd.read_csv(f"{DATA_DIR}time_dates.csv")
    time_dates = pd.to_datetime(dates_df['Date']).values
    site_list = pd.read_csv(f"{DATA_DIR}site_list.csv")
    
    num_sites, num_times, _ = X_raw.shape
    SPLIT_IDX = int(num_times * TRAIN_SPLIT)
    
    X_augmented, _, _ = build_augmented_input(X_raw, y_raw, mask_raw, SPLIT_IDX, decay_rate=DECAY_RATE)
    coeffs = torchcde.linear_interpolation_coeffs(X_augmented.permute(1, 0, 2).reshape(num_times, -1).to(DEVICE))
    
    y_eval, mask_eval = y_raw.permute(1, 0, 2).to(DEVICE), mask_raw.permute(1, 0, 2).to(DEVICE)
    # Extract DHW (assuming channel 1 is Normalized DHW based on earlier scripts)
    dhw_eval = X_raw[:, :, 1].permute(1, 0).to(DEVICE) 
    
    print(f"Loading Best Weights from {MODEL_PATH}...")
    model = CoralSTGNCDE(num_sites, X_augmented.shape[-1], HIDDEN_DIM, 1, adj.to(DEVICE)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    with torch.no_grad(): preds = model(coeffs)
        
    train_y, train_preds, train_mask = y_eval[:SPLIT_IDX], preds[:SPLIT_IDX], mask_eval[:SPLIT_IDX]
    test_y, test_preds, test_mask, test_dhw = y_eval[SPLIT_IDX:], preds[SPLIT_IDX:], mask_eval[SPLIT_IDX:], dhw_eval[SPLIT_IDX:]
    
    train_rmse = torch.sqrt(((train_preds - train_y)**2 * train_mask).sum() / (train_mask.sum() + 1e-6)).item()
    test_rmse = torch.sqrt(((test_preds - test_y)**2 * test_mask).sum() / (test_mask.sum() + 1e-6)).item()
    bleach_rmse = calculate_bleaching_rmse(test_y, test_preds, test_mask, test_dhw, threshold=1.0)
    pearson_r = calculate_pearson_r(test_y, test_preds, test_mask)
    
    train_dca, train_tce, train_far = calculate_trend_metrics(train_y, train_preds, train_mask)
    test_dca, test_tce, test_far = calculate_trend_metrics(test_y, test_preds, test_mask)
    
    gen_gap = test_rmse - train_rmse
    dca_gap = test_dca - train_dca
    
    print("\n===========================================")
    print("            FINAL RESULTS TABLE            ")
    print("===========================================")
    print(f"1. Predictive Acc (Train RMSE)   : {train_rmse:.4f}")
    print(f"2. Predictive Acc (Test RMSE)    : {test_rmse:.4f}")
    print(f"   > Generalization Gap          : {gen_gap:+.4f}")
    print(f"3. Heatwave Stress RMSE          : {bleach_rmse:.4f} (During DHW > 1.0)")
    print(f"4. Pearson Correlation (r)       : {pearson_r:.4f}")
    print(f"-------------------------------------------")
    print(f"5. Directional Accuracy (Test DCA) : {test_dca:.1f}%")
    print(f"   > Train DCA: {train_dca:.1f}% | Gap: {dca_gap:+.1f}%")
    print(f"6. Trend Change Error (Test TCE)   : {test_tce:.1f}%")
    print(f"7. False Alarm Rate (Test FAR)     : {test_far:.1f}%")
    print("===========================================\n")
    
    print("Exporting metrics for baseline comparison...")
    os.makedirs("results", exist_ok=True)
    with open("results/model_metrics.json", "w") as f:
        json.dump({
            "test_rmse": test_rmse, 
            "bleach_rmse": bleach_rmse,
            "pearson_r": pearson_r,
            "test_dca": test_dca,
            "test_tce": test_tce,
            "test_far": test_far
        }, f)
        
    print("Generating Inference Plots for all sites...")
    os.makedirs(PLOT_DIR, exist_ok=True)
    all_site_names = site_list['Site_ID'].tolist()
    
    for site_name in all_site_names:
        target_site = site_list[site_list['Site_ID'] == site_name].index[0]
        site_pred, site_y, site_mask = preds[:, target_site, 0].cpu().numpy(), y_eval[:, target_site, 0].cpu().numpy(), mask_eval[:, target_site, 0].cpu().numpy()
        valid_idx = np.where(site_mask > 0)[0]
        
        plt.figure(figsize=(10, 5))
        plt.plot(time_dates, site_pred, color='blue', linewidth=2, label='AI Forecast')
        plt.plot(time_dates[valid_idx[valid_idx < SPLIT_IDX]], site_y[valid_idx[valid_idx < SPLIT_IDX]], color='black', marker='o', linestyle='', alpha=0.5, label='Training Data')
        plt.plot(time_dates[valid_idx[valid_idx >= SPLIT_IDX]], site_y[valid_idx[valid_idx >= SPLIT_IDX]], color='red', marker='D', linestyle='', label='Test Data')
        
        plt.fill_between(time_dates, 0, X_raw[target_site, :, 1].numpy() * 0.05, color='orange', alpha=0.3, label='DHW Heatwaves')
        plt.axvline(x=time_dates[SPLIT_IDX], color='red', linestyle='--', label='Start of Blind Inference')
        plt.title(f"Forecasting Accuracy: Site {site_name}")
        plt.ylim(-0.05, 1.05)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{PLOT_DIR}forecast_{site_name}.png", bbox_inches='tight')
        plt.close()
    print(f"Successfully generated {len(all_site_names)} plots.")

if __name__ == "__main__":
    main()