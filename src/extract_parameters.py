import torch
import pandas as pd
from coral_model import CoralSTGNCDE
from config import *

def main():
    print(f"--- EXTRACTING BIOLOGICAL PARAMETERS ---")
    
    # 1. Load the site list and adjacency matrix
    site_list = pd.read_csv(f"{DATA_DIR}site_list.csv")
    num_sites = len(site_list)
    adj = torch.load(f"{DATA_DIR}adjacency_matrix.pt").float()
    
    # 2. Initialize the empty architecture
    print(f"Loading Frozen Architecture from {MODEL_PATH}...")
    model = CoralSTGNCDE(
        num_sites=num_sites,
        input_features=6, # Standard 6 features
        hidden_dim=HIDDEN_DIM,
        output_features=1,
        adj_matrix=adj
    ).to(DEVICE)
    
    # 3. Load the winning Epoch 480 weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 4. Extract the learned parameters
    # Apply the exact same clamps used in the physics block
    resilience = torch.clamp(model.site_resilience, min=0.1, max=10.0).detach().cpu().numpy()
    sensitivity = torch.clamp(model.site_sensitivity, min=0.1, max=5.0).detach().cpu().numpy()
    
    # Calculate isolation (alpha) from the mix_param
    isolation = torch.sigmoid(model.func.mix_param).detach().cpu().numpy()
    
    # 5. Build a DataFrame
    results = pd.DataFrame({
        'Site_ID': site_list['Site_ID'],
        'Learned_Resilience': resilience,
        'Learned_Sensitivity': sensitivity,
        'Graph_Isolation_Factor': isolation.flatten()
    })
    
    # Calculate a "Survival Score" (High Resilience / Low Sensitivity)
    results['Survival_Score'] = results['Learned_Resilience'] / results['Learned_Sensitivity']
    
    # Sort to find the absolute strongest and weakest reefs
    results = results.sort_values(by='Survival_Score', ascending=False)
    
    # Save to CSV for your SP Report
    output_csv = "results/learned_reef_parameters.csv"
    results.to_csv(output_csv, index=False)
    
    print("\n🏆 TOP 5 MOST HEAT-RESISTANT REEFS (Survivors):")
    print(results.head(5).to_string(index=False))
    
    print("\n⚠️ TOP 5 MOST VULNERABLE REEFS (High Mortality):")
    print(results.tail(5).to_string(index=False))
    
    print(f"\nFull parameter list saved to: {output_csv}")

if __name__ == "__main__":
    main()