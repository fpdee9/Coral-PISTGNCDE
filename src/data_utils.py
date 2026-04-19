import torch

def normalize_tensor(tensor):
    """Standardizes a tensor (Mean=0, Std=1) over the Time dimension."""
    # Safeguard 1: Catch stray NaNs
    tensor = torch.nan_to_num(tensor, nan=0.0) 
    
    mean = tensor.mean(dim=1, keepdim=True)
    std = tensor.std(dim=1, keepdim=True)
    
    # Safeguard 2: Prevent division by zero or tiny numbers 
    std[std < 1e-4] = 1.0  
    
    normed = (tensor - mean) / std
    return torch.nan_to_num(normed, nan=0.0)

def normalize_adjacency(adj):
    """
    Normalizes the adjacency matrix by row sums.
    Mathematically guarantees graph signals do not explode during message passing.
    """
    row_sums = adj.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return adj / row_sums

def _interpolate_site(s, historical_y, historical_mask, decay_rate):
    """Helper to linearly interpolate and apply confidence decay for a single site."""
    obs_idx = torch.nonzero(historical_mask[s, :, 0]).squeeze(-1)
    
    if len(obs_idx) == 0:
        return
        
    # Back-fill before the first observation
    first_idx = obs_idx[0]
    historical_y[s, :first_idx, 0] = historical_y[s, first_idx, 0]
    historical_mask[s, 0, 0] = 1.0 # Anchor the t=0 mask

    # Linearly interpolate the slope between observations
    for i in range(len(obs_idx) - 1):
        t1, t2 = obs_idx[i], obs_idx[i+1]
        val1, val2 = historical_y[s, t1, 0], historical_y[s, t2, 0]
        
        slope = (val2 - val1) / (t2 - t1)
        historical_mask[s, t1, 0] = 1.0 # Absolute truth observation

        for t in range(t1 + 1, t2):
            historical_y[s, t, 0] = val1 + slope * (t - t1)
            historical_mask[s, t, 0] = historical_mask[s, t-1, 0] * decay_rate
            
    # Forward-fill after the last observation
    last_idx = obs_idx[-1]
    historical_y[s, last_idx:, 0] = historical_y[s, last_idx, 0]
    historical_mask[s, last_idx, 0] = 1.0 

def build_augmented_input(X_raw, y, mask, split_idx, decay_rate=0.98):
    """Single source of truth for data preparation across all scripts."""
    X_normalized = normalize_tensor(X_raw)
    num_sites, num_times, _ = X_raw.shape
    
    # Compress time linearly between 0.0 and 1.0
    time_tensor = torch.linspace(0, 1.0, num_times).view(1, num_times, 1).expand(num_sites, num_times, 1)

    historical_y = y.clone()
    historical_mask = mask.clone()
    
    # Blindfold the AI during the specified Test phase
    historical_mask[:, split_idx:, :] = 0.0

    for s in range(num_sites):
        _interpolate_site(s, historical_y, historical_mask, decay_rate)

    # Append the raw DHW (index 1 of X_raw) as the 6th channel [SST_norm, DHW_norm, Time, Cover, Mask, DHW_raw]
    # This keeps normalized DHW for the neural net (channel 1) and provides biologically-scaled DHW for the physics engine (new channel 5)
    X_augmented = torch.cat([
        X_normalized,       # channels 0-1: SST_norm, DHW_norm (for neural net)
        time_tensor,        # channel 2: time (for CDE solver)
        historical_y,       # channel 3: coral cover
        historical_mask,    # channel 4: confidence mask
        X_raw[:, :, 1:2]    # channel 5: DHW_raw (for physics; real °C-weeks)
    ], dim=-1)

    # Return the augmented tensor, plus the historical components for loss calculations
    return X_augmented, historical_y, historical_mask