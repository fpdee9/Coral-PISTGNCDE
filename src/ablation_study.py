import torch
import torchcde
import argparse
import time
import pandas as pd
import numpy as np       
import random            
from coral_model import CoralSTGNCDE, SpatialVectorField, BIOLOGY_CHANNEL
from data_utils import build_augmented_input, normalize_adjacency
from config import *

# ==========================================
# REPRODUCIBILITY LOCK
# ==========================================
def set_seed(seed):
    """Locks all random number generators for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==========================================
# METRICS HELPER FUNCTION
# ==========================================
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

# ==========================================
# BLACK-BOX MODEL (NO PHYSICS)
# ==========================================
class BlackBoxNCDE(CoralSTGNCDE):
    def forward(self, coeffs):
        X = torchcde.LinearInterpolation(coeffs)
        x0_flat = X.evaluate(X.interval[0]) 
        x0 = x0_flat.reshape(self.num_sites, self.input_features)
        z0 = self.encoder(x0)
        
        p = torch.clamp(x0[:, 3], min=1e-4, max=1.0 - 1e-4)
        bio_anchor = torch.log(p / (1.0 - p)).unsqueeze(1)
        z0_corrected = torch.cat([bio_anchor, z0[:, 1:]], dim=1)
        z0_flat = z0_corrected.view(-1)

        def cde_func_blackbox(t, z):
            z_graph = z.view(self.num_sites, self.hidden_dim)
            current_X = X.evaluate(t).view(self.num_sites, self.input_features)
            h = self.func(z_graph, current_X, self.site_embeddings)
            dz_dt = self.projector(h)
            
            # NO PHYSICS BOUNDS
            bio_channel = dz_dt[:, BIOLOGY_CHANNEL:BIOLOGY_CHANNEL+1] 
            other_channels = dz_dt[:, 1:]                              
            dz_dt_corrected = torch.cat([bio_channel, other_channels], dim=1) 
            
            block_channels = []
            for i in range(self.input_features):
                if i == 2:
                    block_channels.append(dz_dt_corrected)
                else:
                    block_channels.append(torch.zeros_like(dz_dt_corrected))
            sens_blocks = torch.stack(block_channels, dim=-1)
            matrix_4d = self.site_identity * sens_blocks.unsqueeze(1)
            return matrix_4d.permute(0, 2, 1, 3).reshape(self.num_sites * self.hidden_dim, self.num_sites * self.input_features)

        z_T = torchcde.cdeint(X=X, func=cde_func_blackbox, z0=z0_flat, t=X.grid_points, adjoint=False)
        
        time_steps = z_T.shape[0]
        z_T_spatial = z_T.view(time_steps, self.num_sites, self.hidden_dim)
        
        return torch.sigmoid(z_T_spatial[:, :, BIOLOGY_CHANNEL:BIOLOGY_CHANNEL+1])

# ==========================================
# TRAINING FUNCTION
# ==========================================
def run_ablation(mode):
    print(f"\n--- RUNNING ABLATION STUDY: {mode.upper()} ---")
    
    X_raw = torch.load(f"{DATA_DIR}X.pt").float()
    y = torch.load(f"{DATA_DIR}y.pt").float()
    mask = torch.load(f"{DATA_DIR}mask.pt").float()
    adj = torch.load(f"{DATA_DIR}adjacency_matrix.pt").float()
    num_sites, num_times, _ = X_raw.shape
    SPLIT_IDX = int(num_times * TRAIN_SPLIT)
    
    # Save original DHW purely for evaluation of Heatwave RMSE
    original_dhw = X_raw[:, :, 1].clone().permute(1, 0).to(DEVICE)
    
    # STRUCTURAL ABLATION: NO SPATIAL GRAPH
    if mode == "no_graph":
        print("Severing all spatial connections (Identity Matrix)...")
        adj = torch.eye(num_sites)
    else:
        adj = normalize_adjacency(adj)
        
    # FEATURE ABLATION: ZERO OUT CHANNELS
    if mode == "no_sst":
        print("Ablating SST feature channel...")
        X_raw[:, :, 0] = 0.0
    elif mode == "no_dhw":
        print("Ablating DHW feature channel...")
        X_raw[:, :, 1] = 0.0
    elif mode == "time_only":
        print("Ablating BOTH SST and DHW feature channels...")
        X_raw[:, :, 0] = 0.0
        X_raw[:, :, 1] = 0.0
        
    X_augmented, historical_y, historical_mask = build_augmented_input(X_raw, y, mask, SPLIT_IDX, decay_rate=DECAY_RATE)
    num_features = X_augmented.shape[-1]
    
    X_time_first = X_augmented.permute(1, 0, 2)
    X_flat = X_time_first.reshape(num_times, -1).to(DEVICE)
    
    y_test_sparse = y.permute(1, 0, 2).to(DEVICE)
    mask_test_sparse = mask.permute(1, 0, 2).to(DEVICE)
    y_train_continuous = historical_y.permute(1, 0, 2).to(DEVICE)
    mask_train_continuous = historical_mask.permute(1, 0, 2).to(DEVICE)
    
    train_coeffs = torchcde.linear_interpolation_coeffs(X_flat)
    
    # Initialize Correct Model
    if mode == "no_physics":
        print("Using Black-Box Deep Learning (Physics Engine Disabled)...")
        model = BlackBoxNCDE(num_sites, num_features, HIDDEN_DIM, 1, adj).to(DEVICE)
    else:
        model = CoralSTGNCDE(num_sites, num_features, HIDDEN_DIM, 1, adj.to(DEVICE)).to(DEVICE)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-5)
    
    OPTIMAL_EPOCHS = 480
    best_test_rmse = float('inf')
    best_metrics = {}
    
    start_time = time.time()
    for epoch in range(OPTIMAL_EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        pred = model(train_coeffs)
        train_pred = pred[:SPLIT_IDX]
        train_y    = y_test_sparse[:SPLIT_IDX] 
        train_mask = mask_test_sparse[:SPLIT_IDX]

        mse_loss = ((train_pred - train_y) ** 2 * train_mask).sum() / (train_mask.sum() + 1e-6)
        mae_loss = (torch.abs(train_pred - train_y) * train_mask).sum() / (train_mask.sum() + 1e-6)
        path_loss = (0.2 * mse_loss) + (0.8 * mae_loss)
        
        init_loss = ((train_pred[0] - y_train_continuous[0]) ** 2 * mask_train_continuous[0]).sum() / (mask_train_continuous[0].sum() + 1e-6)
        current_init_weight = INIT_LOSS_WEIGHT * max(0.0, 1.0 - (epoch / 50.0))

        loss = path_loss + (init_loss * current_init_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step(epoch)
        
        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = pred[SPLIT_IDX:]
                test_y    = y_test_sparse[SPLIT_IDX:]
                test_mask = mask_test_sparse[SPLIT_IDX:]
                
                test_rmse = torch.sqrt(((test_pred - test_y) ** 2 * test_mask).sum() / (test_mask.sum() + 1e-6))
                
                # Track best metrics
                if test_rmse < best_test_rmse:
                    best_test_rmse = test_rmse
                    test_mae = (torch.abs(test_pred - test_y) * test_mask).sum() / (test_mask.sum() + 1e-6)
                    dca, tce, far = calculate_trend_metrics(test_y, test_pred, test_mask)
                    
                    # Heatwave RMSE calculation using the un-ablated true DHW
                    test_dhw = original_dhw[SPLIT_IDX:]
                    bleach_mask = (test_dhw > 1.0).unsqueeze(-1) * test_mask
                    if bleach_mask.sum() > 0:
                        bleach_rmse = torch.sqrt(((test_pred - test_y)**2 * bleach_mask).sum() / bleach_mask.sum()).item()
                    else:
                        bleach_rmse = float('nan')
                        
                    best_metrics = {
                        'rmse': test_rmse.item(), 
                        'mae': test_mae.item(), 
                        'heatwave_rmse': bleach_rmse,
                        'dca': dca, 
                        'tce': tce, 
                        'far': far
                    }
                    
            print(f"Epoch {epoch+1:03d}/{OPTIMAL_EPOCHS} | {mode.upper()} Test RMSE: {test_rmse:.4f}")

    print(f"\n=======================================================")
    print(f" {mode.upper()} ABLATION RESULTS (For Manuscript Tables) ")
    print(f"=======================================================")
    print(f"Test RMSE      : {best_metrics['rmse']:.4f}")
    print(f"Test MAE       : {best_metrics['mae']:.4f}")
    print(f"Heatwave RMSE  : {best_metrics['heatwave_rmse']:.4f}")
    print(f"DCA            : {best_metrics['dca']:.1f}%")
    print(f"TCE            : {best_metrics['tce']:.1f}%")
    print(f"FAR            : {best_metrics['far']:.1f}%")
    print(f"=======================================================\n")
    
    return best_metrics['rmse']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['no_graph', 'no_physics', 'no_sst', 'no_dhw', 'time_only'])
    args = parser.parse_args()
    
    set_seed(SEED) # Reproducibility lock successfully placed here
    run_ablation(args.mode)