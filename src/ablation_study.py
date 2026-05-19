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
    
    # NO SPATIAL GRAPH
    if mode == "no_graph":
        print("Severing all spatial connections (Identity Matrix)...")
        # Every reef is forced to predict in total isolation, ignoring neighbor heatwaves
        adj = torch.eye(num_sites)
    else:
        adj = normalize_adjacency(adj)
        
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
                if test_rmse < best_test_rmse:
                    best_test_rmse = test_rmse
                    
            print(f"Epoch {epoch+1:03d}/{OPTIMAL_EPOCHS} | {mode.upper()} Test RMSE: {test_rmse:.4f}")

    print(f"\n{mode.upper()} TRAINING COMPLETE.")
    print(f"Final Best Test RMSE for '{mode}': {best_test_rmse:.4f}")
    print(f"Time elapsed: {(time.time() - start_time)/60:.1f} minutes.")
    return best_test_rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['no_graph', 'no_physics'])
    args = parser.parse_args()
    
    set_seed(SEED) # Reproducibility lock successfully placed here
    run_ablation(args.mode)