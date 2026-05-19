import torch

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================

# 1. File Paths
DATA_DIR = "data/processed/"
MODEL_PATH = "coral_model_best.pth"
MODEL_LATEST_PATH = "coral_model_latest.pth"
PLOT_DIR = "results/plots/"
NEIGHBORHOOD_PLOT_DIR = "results/neighborhood_plots/"

# 2. Pipeline Parameters
TRAIN_SPLIT = 0.8  # 80% Train, 20% Test
DECAY_RATE = 0.98  # Confidence decay for the interpolation mask

# 3. Model Architecture
HIDDEN_DIM = 64
EMBED_DIM = 16

# 4. Training Hyperparameters
EPOCHS = 1000
LR = 0.0003
WEIGHT_DECAY = 1e-5
INIT_LOSS_WEIGHT = 10.0
SEED = 42

# 5. Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 6. Biological Physics
DHW_BLEACHING_THRESHOLD = 1.0       
RECOVERY_SUPPRESSION_THRESHOLD = 3.0 
TRAUMA_SCALE = 50.0                 
NEURAL_RECOVERY_SCALE = 25.0 
MORTALITY_SCALE = 0.05            
REBOUND_SCALE = 2.0              