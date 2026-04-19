import torch
import torch.nn as nn
import torchcde
from config import *

# --- ARCHITECTURAL CONSTANTS ---
# Defines the specific hidden channel mathematically reserved for Biological State (Coral Cover)
BIOLOGY_CHANNEL = 0

class SpatialVectorField(nn.Module):
    def __init__(self, num_sites, hidden_channels, input_features, adj_matrix):
        super().__init__()
        self.num_sites = num_sites
        self.hidden_channels = hidden_channels
        self.input_features = input_features
        self.adj = adj_matrix # The static physical map
        
        # State-Aware Integration: Accepts Memory + Real-Time Environment
        # The vector field should ONLY look at [SST, DHW, Time]. 
        # Sever its access to the artificial decaying mask.
        env_dim = 3 + EMBED_DIM
        combined_dim = hidden_channels + env_dim

        # Spatial Graph Convolution
        # process all sites -> transform features -> mix with neighbors
        # Deeper vector fields to allow for complex "weaving"
        self.gcn_layer1 = nn.Linear(combined_dim, hidden_channels * 2)
        self.gcn_layer2 = nn.Linear(hidden_channels * 2, hidden_channels)
        
        # Temporal Evolution
        self.time_layer1 = nn.Linear(combined_dim, hidden_channels * 2)
        self.time_layer2 = nn.Linear(hidden_channels * 2, hidden_channels)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.tanh = nn.Tanh()

        # Initialize at 0.0 -> sigmoid(0) = 0.5 -> equal mixing to start, allowing gradient flow
        self.mix_param = nn.Parameter(torch.full((num_sites, 1), -5.0))
        
    def forward(self, z_graph, current_X, site_embeddings):
        # z_graph shape: (Sites, Hidden)
        # current_X features are: [0:SST, 1:DHW, 2:Time, 3:Coral_Cover, 4:Mask]
        # Message Passing: Broadcast the matrix multiplication ((Sites, Sites) @ (Sites, Hidden))

        # PURE BIOLOGICAL ATTENTION (Self-Sustaining)
        # Look at the actively evolving biological hidden state, not the static input
        coral_cover = torch.sigmoid(z_graph[:, BIOLOGY_CHANNEL]).unsqueeze(1) # Shape: (Sites, 1)

        # Calculate the absolute mathematical difference in coral cover between all pairs
        cover_diff = torch.abs(coral_cover - coral_cover.transpose(0, 1))

        # Convert difference to an attention score. 
        # The * 10.0 multiplier means even a 20% difference in biology kills the connection.
        scores = torch.exp(-cover_diff * 10.0)
        
        # Apply the static physical distance mask
        mask = (self.adj > 0).float()
        attention_weights = scores * mask

        # Normalize weights
        row_sums = attention_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        attention_weights = attention_weights / row_sums
        
        # UNIFIED RESIDUAL MIXING
        # Calculate alpha here using the vector field's own parameter
        alpha = torch.sigmoid(self.mix_param) # (Sites, 1) — each reef learns its own isolation

        # Extract the raw coral cover (Channel 0) and the hidden context (Channels 1 to 63)
        z_bio = z_graph[:, BIOLOGY_CHANNEL:BIOLOGY_CHANNEL+1] # (Sites, 1)
        z_context = z_graph[:, BIOLOGY_CHANNEL+1:]            # (Sites, Hidden-1)

        # Apply Graph Mixing ONLY to the context/memory channels.
        # Neighbors can share stress and temperature history, but NEVER raw coral cover.
        z_context_neighbor = alpha * torch.matmul(attention_weights, z_context) + (1.0 - alpha) * z_context
        
        # Recombine them safely
        z_neighbor = torch.cat([z_bio, z_context_neighbor], dim=1)
        
        # Extract purely the environmental features [SST, DHW, Time]
        env_X = current_X[:, :3]

        # Concatenate the unique site embeddings to the spatial and temporal paths.
        z_combined_spatial = torch.cat([z_neighbor, env_X, site_embeddings], dim=-1)
        z_combined_temporal = torch.cat([z_graph, env_X, site_embeddings], dim=-1)

        # Update States

        # Spatial Processing
        gcn_out = self.leaky_relu(self.gcn_layer1(z_combined_spatial))
        gcn_out = self.gcn_layer2(gcn_out)
        

        # Temporal Processing
        time_out = self.leaky_relu(self.time_layer1(z_combined_temporal))
        time_out = self.time_layer2(time_out)

        # Combine spatial and temporal outputs and output RAW derivative
        z_out = gcn_out + time_out
        
        # Apply Tanh to safely bound the derivative
        return self.tanh(z_out)

class CoralSTGNCDE(nn.Module):
    def __init__(self, num_sites, input_features, hidden_dim, output_features, adj_matrix):
        super().__init__()
        self.num_sites = num_sites
        self.hidden_dim = hidden_dim
        self.input_features = input_features
        self.adj_matrix = nn.Parameter(adj_matrix, requires_grad=False)
        
        # Encoder: Lifts raw env data (SST, DHW) to hidden state
        # Gives the AI the brainpower to align starting state 
        self.encoder = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Vector Field/ Differential Equation f(z)
        self.func = SpatialVectorField(num_sites, hidden_dim, input_features, self.adj_matrix)
        
        # Define the projector to output a single vector (dz/dt) instead of a massive matrix for all environmental variables.
        self.projector = nn.Linear(hidden_dim, hidden_dim)
        
        # Prevent Tanh Saturation on Epoch 1
        # Zeroes out the projector so dz_dt starts at exactly 0.0
        # This guarantees tanh(0.0) = 0.0, keeping the gradient signal at maximum strength (1.0)
        torch.nn.init.zeros_(self.projector.weight)
        torch.nn.init.zeros_(self.projector.bias)

        self.site_embeddings = nn.Parameter(torch.randn(num_sites, EMBED_DIM) * 0.01)

        # Per-site biological parameters (Genetic Memory)
        # Initialized to 1.0 so the physics engine starts neutral. 
        # The optimizer will independently push these up or down for each specific reef.
        self.site_resilience = nn.Parameter(torch.ones(num_sites))
        self.site_sensitivity = nn.Parameter(torch.ones(num_sites))
        
        # Pre-allocate identity matrix for block-diagonal construction
        # This speeds up the cde_func significantly
        self.register_buffer('site_identity', torch.eye(num_sites).view(num_sites, num_sites, 1, 1))

    def forward(self, coeffs):
        # Construct the continuous path X(t) from the spline coefficients
        # coeffs represents the continuous environmental path X(t)
        X = torchcde.LinearInterpolation(coeffs)
        
        # --- INITIAL STATE ---
        # Evaluate spline at initial hidden state t=0. Shape is (Sites * Features)
        x0_flat = X.evaluate(X.interval[0]) 
        # Reshape to (Sites, Features)
        x0 = x0_flat.reshape(self.num_sites, self.input_features)

        # Safely map the starting state using the neural network
        z0 = self.encoder(x0)
        
        # MATHEMATICAL ANCHOR LOCK
        # Clamp the real starting cover to avoid math errors, then convert it to "Logit Space" (Inverse Sigmoid).
        p = torch.clamp(x0[:, 3], min=1e-4, max=1.0 - 1e-4)

        # Clean concatenation to preserve Encoder gradients
        bio_anchor = torch.log(p / (1.0 - p)).unsqueeze(1)
        other_z0 = z0[:, 1:]
        z0_corrected = torch.cat([bio_anchor, other_z0], dim=1)

        z0_flat = z0_corrected.view(-1)

        # Clamp to prevent biologically impossible runaway resilience or negative sensitivity
        resilience = torch.clamp(self.site_resilience, min=0.1, max=10.0)
        sensitivity = torch.clamp(self.site_sensitivity, min=0.1, max=5.0)

        # --- CDE DYNAMICS ---
        def cde_func(t, z):
            # z is flattened (Sites*Hidden)
            # Reshape z to graph: (Sites, Hidden)
            z_graph = z.view(self.num_sites, self.hidden_dim)

            # Extract real-time environment
            current_X_flat = X.evaluate(t)
            current_X = current_X_flat.view(self.num_sites, self.input_features)

            # Evolve z spatially (GCN)
            h = self.func(z_graph, current_X, self.site_embeddings)

            # CDE ENGINE OVERRIDE
            # Project to control sensitivity
            dz_dt = self.projector(h) # Shape: (Sites, Hidden)
            
           # MACRO-SCALE PHYSICS + ECOLOGICAL REBOUND
            current_coral = torch.sigmoid(z_graph[:, BIOLOGY_CHANNEL])
            dhw_raw = current_X[:, 5]
            soft_ceiling = torch.clamp(1.5 - current_coral, min=0.2, max=1.0)
            trauma_stress = torch.relu(dhw_raw - DHW_BLEACHING_THRESHOLD)
            heatwave_trauma = (trauma_stress 
                               * TRAUMA_SCALE 
                               * sensitivity  # Site-specific trauma
                               * current_coral)
            scaled_bio = dz_dt[:, BIOLOGY_CHANNEL] * resilience
            neural_recovery = (torch.tanh(scaled_bio) 
                               * NEURAL_RECOVERY_SCALE 
                               * soft_ceiling)
            baseline_mortality = -MORTALITY_SCALE * current_coral
            recovery_stress = torch.relu(dhw_raw - RECOVERY_SUPPRESSION_THRESHOLD)
            soft_no_stress = torch.exp(-recovery_stress * 2.0)
            rebound_boost = soft_ceiling * REBOUND_SCALE * soft_no_stress
            ai_recovery = baseline_mortality + neural_recovery + rebound_boost

            # Preserving the Autograd Graph otherwise PyTorch will sever the gradient back to the projector.
            bio_channel = (ai_recovery - heatwave_trauma).unsqueeze(1) # Shape: (Sites, 1)
            other_channels = dz_dt[:, 1:]                              # Shape: (Sites, Hidden-1)
            
            # Concatenate them side-by-side to create a brand new, unbroken tensor
            dz_dt_corrected = torch.cat([bio_channel, other_channels], dim=1) 

            # Fully Differentiable Matrix Construction
            # Build the matrix channel-by-channel and stack them to preserve autograd
            block_channels = []
            for i in range(self.input_features):
                if i == 2:  # The Time Channel
                    block_channels.append(dz_dt_corrected)
                else:
                    # Use zeros_like so the blank channels safely participate in the graph
                    block_channels.append(torch.zeros_like(dz_dt_corrected))
            
            # Stack them side-by-side to create the final (Sites, Hidden, Inputs) matrix
            sens_blocks = torch.stack(block_channels, dim=-1)

            # Construct Block Diagonal Matrix
            # Matrix of shape (Sites*Hidden, Sites*Inputs) where the diagonal blocks are 'sens_blocks' and off-diagonals are zero.
            # Broadcast multiply Identity with Blocks (Sites, Sites, 1, 1) * (Sites, 1, Hidden, Inputs) -> (Sites, Sites, Hidden, Inputs)
            # The broadcast matches the first 'Sites' dim of identity with 'Sites' dim of blocks
            # Note: Align "Row Site" with "Col Site" via the diagonal
            matrix_4d = self.site_identity * sens_blocks.unsqueeze(1)
            
            # Reshape to 2D Matrix: (Sites*Hidden, Sites*Inputs)
            # Permute to (Site_Row, Hidden, Site_Col, Input) before flattening
            matrix_2d = matrix_4d.permute(0, 2, 1, 3).reshape(
                self.num_sites * self.hidden_dim, 
                self.num_sites * self.input_features
            )
            return matrix_2d

        # Solve the CDE integration
        # This computes z(t) for every time step
        z_T = torchcde.cdeint(X=X, func=cde_func, z0=z0_flat, t=X.grid_points, adjoint=False)
        
        # Reshape output: (Time, Sites, Hidden)
        time_steps = z_T.shape[0]
        z_T_spatial = z_T.view(time_steps, self.num_sites, self.hidden_dim)
        
        # PURE OUTPUT
        # Slice exactly Dimension 0 (the protected biological state) 
        # and pass it through a Sigmoid to return it to a clean 0.0 to 1.0 percentage.
        prediction = torch.sigmoid(z_T_spatial[:, :, BIOLOGY_CHANNEL:BIOLOGY_CHANNEL+1])
        return prediction