import torch
import torchcde
import numpy as np
import time
import os
import copy
import random
import matplotlib.pyplot as plt
import pandas as pd            
from coral_model import CoralSTGNCDE
from data_utils import build_augmented_input, normalize_adjacency
from config import *

def set_seed(seed):
    """Locks all random number generators for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Force deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():

    set_seed(SEED)

    print(f"--- STARTING TRAINING ON {DEVICE} ---")
    
    if not os.path.exists(f"{DATA_DIR}X.pt"):
        print(f"Error: {DATA_DIR}X.pt not found.")
        return

    # Load Data
    X_raw = torch.load(f"{DATA_DIR}X.pt").float() # (Sites, Time, Features)
    y = torch.load(f"{DATA_DIR}y.pt").float() # (Sites, Time, 1)
    mask = torch.load(f"{DATA_DIR}mask.pt").float()
    adj = torch.load(f"{DATA_DIR}adjacency_matrix.pt").float()

    # Load labels for the live monitor
    dates_df = pd.read_csv(f"{DATA_DIR}time_dates.csv")
    time_dates = pd.to_datetime(dates_df['Date']).values
    site_list = pd.read_csv(f"{DATA_DIR}site_list.csv")
    MONITOR_SITE_IDX = 92 # Monitor the last reef in the list
    monitor_site_name = site_list.iloc[MONITOR_SITE_IDX]['Site_ID']
    
    # Inspect Dimensions
    num_sites, num_times, num_features = X_raw.shape
    print(f"   > Sites: {num_sites}, Time Steps: {num_times}, Features: {num_features}")

    # ABLATION TEST (Turn off the spatial graph):
    # torch.eye creates an Identity Matrix (1s on the diagonal, 0s everywhere else).
    # This means Reef A only talks to Reef A. The network is now purely temporal (comment this out to restore the graph)
    # adj = torch.eye(num_sites)
    
    # Mathematically guarantee the graph cannot explode from overlapping nodes
    adj = normalize_adjacency(adj)

    SPLIT_IDX = int(num_times * TRAIN_SPLIT)
    
    # Call the single source of truth for interpolation
    X_augmented, historical_y, historical_mask = build_augmented_input(
        X_raw, y, mask, SPLIT_IDX, decay_rate=DECAY_RATE
    )

    # Update the feature count so the AI knows to look for 5 features
    num_features = X_augmented.shape[-1]

    # Use X_augmented moving forward
    # Reshape X for CDE Solver
    # The CDE solver needs a single path.
    # Flatten (Sites, Features) into one large "Channels" dimension
    # Target Shape: (Time, Sites * Features)
    # Permute to (Time, Sites, Features)
    X_time_first = X_augmented.permute(1, 0, 2)
    # Flatten to (Time, Sites * Features) for the single-path solver
    # .reshape() handles non-contiguous memory from permute
    X_flat = X_time_first.reshape(num_times, -1).to(DEVICE)
    
    # Note: torchcde supports unbatched inputs of shape (Time, Channels)
    # This effectively treats the whole system as Batch=1

    # Move other tensors to Device
    # y and mask need to match the output shape (Time, Sites, 1) for loss calc
    # THE LOSS TARGETS
    # Use continuous interpolated path for TRAINING so the AI learns to weave smoothly
    y_train_continuous = historical_y.permute(1, 0, 2).to(DEVICE)
    mask_train_continuous = historical_mask.permute(1, 0, 2).to(DEVICE)
    # Use the raw sparse data for TESTING so we don't cheat the validation metrics
    y_test_sparse = y.permute(1, 0, 2).to(DEVICE) # (Time, Sites, 1)
    mask_test_sparse = mask.permute(1, 0, 2).to(DEVICE)
    adj = adj.to(DEVICE)
    
    print(f"Data Loaded. X Shape: {X_flat.shape}")

    # Interpolation (The "Continuous" part)
    # Interpolate Data (Create Continuous Path)
    print("Interpolating Data...")
    # Converts discrete daily data into a continuous mathematical function X(t)
    # Coefficients for the single continuous path
    train_coeffs = torchcde.linear_interpolation_coeffs(X_flat)
    
    # Initialize Model
    
    model = CoralSTGNCDE(
        num_sites=num_sites,
        input_features=num_features, # Auto-detects
        hidden_dim=HIDDEN_DIM,
        output_features=1, # 1 = Coral Cover
        adj_matrix=adj
    ).to(DEVICE)
    
    # Weight Decay helps prevent getting stuck
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)

    # T_0=50 triggers the first massive reset right when the anchor snaps!
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=50, 
        T_mult=2, 
        eta_min=1e-5
    )
    
    best_test_rmse = float('inf')
    best_epoch = 0
    
    # Training Loop
    print("\n--- BEGINNING TRAINING ---")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # Forward Pass
        # Returns: (Time, Sites, 1) 
        # Pass the coefficients, and the model solves the integral
        pred = model(train_coeffs)
        
        # SPARSE TRAJECTORY TRAINING 
        # Calculate Training Loss: Calculate Loss ONLY on Observed Data
        # Slice [0:SPLIT_IDX] for training
        # Use 'y_test_sparse', which contains ONLY the raw, uninterpolated red dots.
        train_pred = pred[:SPLIT_IDX]
        train_y    = y_test_sparse[:SPLIT_IDX] 
        train_mask = mask_test_sparse[:SPLIT_IDX]

        # BLENDED TRAJECTORY LOSS
        # MSE aggressively attacks large errors, while MAE provides stable tracking for the overall trend.
        # Blending them prevents sparse biological outliers from warping the entire curve.
        mse_loss = ((train_pred - train_y) ** 2 * train_mask).sum() / (train_mask.sum() + 1e-6)
        mae_loss = (torch.abs(train_pred - train_y) * train_mask).sum() / (train_mask.sum() + 1e-6)
        
        path_loss = (0.4 * mse_loss) + (0.6 * mae_loss)
        
        # INITIAL STATE ANCHOR
        # Use the continuous backfilled tensor JUST for t=0, to ensure the anchor holds
        pred_t0 = train_pred[0] 
        true_t0 = y_train_continuous[0]
        mask_t0 = mask_train_continuous[0]
        
        # Calculate how badly the AI missed the starting point
        init_loss = ((pred_t0 - true_t0) ** 2 * mask_t0).sum() / (mask_t0.sum() + 1e-6)

        # Starts at INIT_LOSS_WEIGHT (10.0) and decays to a minimum of 1.0.
        # Forces the anchor early, but lets the AI prioritize the spatiotemporal weave later
        # Reaches 0.0 by epoch 300, giving the final 200 epochs pure trajectory learning
        current_init_weight = INIT_LOSS_WEIGHT * max(0.0, 1.0 - (epoch / 50.0))

        loss = path_loss + (init_loss * current_init_weight)

        loss.backward()
        
        # Gradient clipping to prevent zig-zags from exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()

        scheduler.step(epoch)
        
        # Reporting
        # Validation / Saving
        if (epoch + 1) % 5 == 0: 
            # Calculate Test Error/Loss
            model.eval()
            with torch.no_grad():
                # Test Loss (Future Data on Sparse Points only)
                test_pred = pred[SPLIT_IDX:]
                test_y    = y_test_sparse[SPLIT_IDX:]
                test_mask = mask_test_sparse[SPLIT_IDX:]
                
                # RMSE (Root Mean Squared Error) - more interpretable
                test_mse = ((test_pred - test_y) ** 2 * test_mask).sum() / (test_mask.sum() + 1e-6)
                test_rmse = torch.sqrt(test_mse)
                train_rmse = torch.sqrt(loss)
                
                print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train: {train_rmse:.4f} | Test: {test_rmse:.4f}", end="")
                
                # --------- LIVE MONITORING GRAPH ----------
                # Extract the 1D prediction and truth arrays for our chosen site
                live_pred = pred[:, MONITOR_SITE_IDX, 0].cpu().numpy()
                live_y = y_test_sparse[:, MONITOR_SITE_IDX, 0].cpu().numpy()
                live_mask = mask_test_sparse[:, MONITOR_SITE_IDX, 0].cpu().numpy()
                
                # Only plot the red dots where there are observations
                valid_idx = np.where(live_mask > 0)[0]
                
                plt.figure(figsize=(10, 5))
                plt.plot(time_dates, live_pred, color='blue', label=f'Epoch {epoch+1} Prediction', linewidth=2)
                plt.plot(time_dates[valid_idx], live_y[valid_idx], color='red', marker='o', linestyle='', label='Observed Data')
                
                # Add the train/test split line
                plt.axvline(x=time_dates[SPLIT_IDX], color='black', linestyle='--', label='Train/Test Split')
                
                plt.title(f"Live Training Heartbeat: {monitor_site_name} (Epoch {epoch+1})", fontweight='bold')
                plt.ylim(-0.05, 1.05)
                plt.legend(loc='upper left')
                plt.grid(True, alpha=0.3)
                
                # Save and close (overwrites the same file so it doesn't flood the hard drive)
                os.makedirs(PLOT_DIR, exist_ok=True)
                plt.savefig(f"{PLOT_DIR}live_monitor.png", bbox_inches='tight')
                plt.close()
                # ---------------------------------

                # SAVE AS LATEST MODEL
                torch.save(model.state_dict(), MODEL_LATEST_PATH)
                
                # SAVE MODEL WITH LOWEST TEST RMSE
                if test_rmse < best_test_rmse:
                    best_test_rmse = test_rmse
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), MODEL_PATH)
                    print(f"  <-- SAVED (New Best & Latest)")
                else:
                    print(f"  <-- SAVED (Latest Only)")
    print(f"\n--- TRAINING COMPLETE ---")
    duration = time.time() - start_time
    print(f"\nTraining Finished in {duration/60:.1f} minutes.")
    print(f"Best Test RMSE: {best_test_rmse:.4f} at Epoch {best_epoch}")
    print(f"Best Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()

    