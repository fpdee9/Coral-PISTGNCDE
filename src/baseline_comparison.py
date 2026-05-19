import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from statsmodels.tsa.api import VAR
from config import *
from data_utils import build_augmented_input

def calculate_trend_metrics(y_true, y_pred, mask):
    n_sites = y_true.shape[1]
    correct_directions, total_direction_pairs = 0, 0
    missed_trend_changes, total_trend_changes = 0, 0
    false_alarms, stable_trends = 0, 0
    
    for s in range(n_sites):
        site_true, site_pred, site_mask = y_true[:, s, 0], y_pred[:, s, 0], mask[:, s, 0]
        observed_indices = torch.nonzero(site_mask).squeeze()
        
        if len(observed_indices.shape) == 0 or observed_indices.shape[0] < 3:
            continue 
            
        for i in range(1, len(observed_indices)):
            t_curr, t_prev = observed_indices[i], observed_indices[i-1]
            true_diff = site_true[t_curr] - site_true[t_prev]
            pred_diff = site_pred[t_curr] - site_pred[t_prev]
            
            if (true_diff > 0 and pred_diff > 0) or (true_diff < 0 and pred_diff < 0):
                correct_directions += 1
            total_direction_pairs += 1
            
            if i >= 2:
                t_prev2 = observed_indices[i-2]
                prev_true_diff = site_true[t_prev] - site_true[t_prev2]
                actual_sign, prev_actual_sign, pred_sign = torch.sign(true_diff), torch.sign(prev_true_diff), torch.sign(pred_diff)
                
                # Trend Change Error
                if actual_sign != prev_actual_sign and actual_sign != 0 and prev_actual_sign != 0:
                    total_trend_changes += 1
                    if pred_sign != actual_sign:
                        missed_trend_changes += 1
                # False Alarm Rate (Trend did not change, but model predicted a change)
                elif actual_sign == prev_actual_sign and actual_sign != 0:
                    stable_trends += 1
                    if pred_sign != actual_sign:
                        false_alarms += 1

    dca = (correct_directions / max(total_direction_pairs, 1)) * 100
    tce = (missed_trend_changes / max(total_trend_changes, 1)) * 100
    far = (false_alarms / max(stable_trends, 1)) * 100
    return dca, tce, far

class ConventionalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x)

class BaselineGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_sites):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        gru_out, _ = self.gru(x.permute(1, 0, 2)) 
        return self.sigmoid(self.out(gru_out.permute(1, 0, 2)))

def evaluate_naive(y_eval, mask_eval, split_idx):
    print("Evaluating Naive Persistence Baseline...")
    last_obs = torch.zeros(y_eval.shape[1], 1, device=DEVICE)
    for s in range(y_eval.shape[1]):
        obs_idx = torch.nonzero(mask_eval[:split_idx, s, 0]).squeeze()
        if obs_idx.numel() > 0:
            last_t = obs_idx[-1] if obs_idx.dim() > 0 else obs_idx
            last_obs[s] = y_eval[last_t, s, 0]
            
    test_y, test_mask = y_eval[split_idx:], mask_eval[split_idx:]
    naive_pred = last_obs.unsqueeze(0).expand(test_y.shape[0], -1, 1)
    
    rmse = torch.sqrt(((naive_pred - test_y) ** 2 * test_mask).sum() / (test_mask.sum() + 1e-6)).item()
    dca, tce, far = calculate_trend_metrics(test_y, naive_pred, test_mask)
    return rmse, dca, tce, far

def evaluate_starima(y_raw, mask_raw, split_idx):
    print("Evaluating VAR...")
    df = pd.DataFrame(y_raw[:, :, 0].numpy().T).replace(0, np.nan)
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)              
    
    try:
        results = VAR(df.iloc[:split_idx].values).fit(maxlags=3)
        forecast_tensor = torch.tensor(results.forecast(results.endog, steps=df.shape[0] - split_idx), dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        test_y, test_mask = y_raw.permute(1, 0, 2).to(DEVICE)[split_idx:], mask_raw.permute(1, 0, 2).to(DEVICE)[split_idx:]
        
        rmse = torch.sqrt(((forecast_tensor - test_y)**2 * test_mask).sum() / (test_mask.sum() + 1e-6)).item()
        dca, tce, far = calculate_trend_metrics(test_y, forecast_tensor, test_mask)
        return rmse, dca, tce, far
    except Exception as e:
        return float('nan'), 0.0, 100.0, 100.0

def evaluate_deep_learning(model_type, X_seq, y_eval, mask_eval, split_idx, input_features, num_sites):
    print(f"Training Deep Learning Baseline: {model_type}...")
    model = BaselineGRU(input_features, HIDDEN_DIM, num_sites).to(DEVICE) if model_type == "GRU" else ConventionalMLP(input_features, HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_rmse, best_dca, best_tce, best_far = float('inf'), 0.0, 100.0, 100.0
    
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
                test_pred, test_y, test_mask = eval_preds[split_idx:], y_eval[split_idx:], mask_eval[split_idx:]
                test_rmse = torch.sqrt(((test_pred - test_y)**2 * test_mask).sum() / (test_mask.sum() + 1e-6)).item()
                
                if test_rmse < best_rmse:
                    best_rmse = test_rmse
                    best_dca, best_tce, best_far = calculate_trend_metrics(test_y, test_pred, test_mask)
    return best_rmse, best_dca, best_tce, best_far

def main():
    print("=== COMMENCING BASELINE BENCHMARKING ===")
    X_raw, y_raw, mask_raw = torch.load(f"{DATA_DIR}X.pt").float(), torch.load(f"{DATA_DIR}y.pt").float(), torch.load(f"{DATA_DIR}mask.pt").float()
    num_sites, num_times, _ = X_raw.shape
    SPLIT_IDX = int(num_times * TRAIN_SPLIT)
    
    X_augmented, _, _ = build_augmented_input(X_raw, y_raw, mask_raw, SPLIT_IDX, decay_rate=DECAY_RATE)
    y_eval, mask_eval = y_raw.permute(1, 0, 2).to(DEVICE), mask_raw.permute(1, 0, 2).to(DEVICE)
    
    X_env_only = X_augmented[:, :, :3].permute(1, 0, 2).to(DEVICE) 
    env_features = 3
    
    n_rmse, n_dca, n_tce, n_far = evaluate_naive(y_eval, mask_eval, SPLIT_IDX)
    s_rmse, s_dca, s_tce, s_far = evaluate_starima(y_raw, mask_raw, SPLIT_IDX)
    m_rmse, m_dca, m_tce, m_far = evaluate_deep_learning("Conventional (MLP)", X_env_only, y_eval, mask_eval, SPLIT_IDX, env_features, num_sites)
    g_rmse, g_dca, g_tce, g_far = evaluate_deep_learning("GRU", X_env_only, y_eval, mask_eval, SPLIT_IDX, env_features, num_sites)
    
    try:
        with open("results/model_metrics.json", "r") as f:
            metrics = json.load(f)
            rmse_stgncde = metrics["test_rmse"]
            stgncde_dca = metrics["test_dca"]
            stgncde_tce = metrics["test_tce"]
            stgncde_far = metrics["test_far"]
    except FileNotFoundError:
        print("\n[WARNING] results/model_metrics.json not found! Run evaluation_metrics.py first.")
        print("Using placeholder values for STG-NCDE.\n")
        rmse_stgncde, stgncde_dca, stgncde_tce, stgncde_far = 0.1526, 0.0, 0.0, 0.0

    best_baseline = min(n_rmse, m_rmse, g_rmse, s_rmse if not np.isnan(s_rmse) else float('inf'))
    skill_score = 1.0 - (rmse_stgncde / best_baseline)

    print("\n=========================================================================================")
    print("                      Baseline Comparison Results                      ")
    print("=========================================================================================")
    print(f"Model                      | Test RMSE  | Dir. Acc (DCA) | Trend Err (TCE) | False Alarm")
    print("-----------------------------------------------------------------------------------------")
    print(f"(i)   Naive Method         | {n_rmse:.4f}     | {n_dca:5.1f}%          | {n_tce:5.1f}%           | {n_far:5.1f}%")
    print(f"(ii)  MLP                  | {m_rmse:.4f}     | {m_dca:5.1f}%          | {m_tce:5.1f}%           | {m_far:5.1f}%")
    print(f"(iii) VAR                  | {s_rmse:.4f}     | {s_dca:5.1f}%          | {s_tce:5.1f}%           | {s_far:5.1f}%")
    print(f"(iv)  GRU                  | {g_rmse:.4f}     | {g_dca:5.1f}%          | {g_tce:5.1f}%           | {g_far:5.1f}%")
    print("-----------------------------------------------------------------------------------------")
    print(f"[*]   Proposed STG-NCDE    | {rmse_stgncde:.4f}     | {stgncde_dca:5.1f}%          | {stgncde_tce:5.1f}%           | {stgncde_far:5.1f}%")
    print("=========================================================================================\n")
    print(f"Model Skill Score (SS) vs Best Baseline: {skill_score:.3f}")
    print(f"(>0 means model beats all baselines; >0.3 is generally considered highly significant)")

if __name__ == "__main__":
    main()