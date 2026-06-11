import torch

# Load your mask tensor (adjust the path to match your DATA_DIR)
mask = torch.load(r"C:\Users\ipedee\Downloads\Coral-STGNCDE\data\processed\mask.pt")

# Sum all the 1s in the tensor
total_observations = int(mask.sum().item())

print(f"Total valid biological observations: {total_observations}")