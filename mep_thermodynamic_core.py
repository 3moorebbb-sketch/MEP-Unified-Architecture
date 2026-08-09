import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Set random seed for consistent graph generation properties and trajectories
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# 1. THE COMBINATORIAL OPTIMIZATION SOLVER
# =============================================================================

class ThermodynamicIsingNetwork(nn.Module):
    """
    A continuous-time dynamical system solver for Ising/Max-Cut problems.
    Uses Euler-Maruyama integration to relax into a low-energy stable state.
    """
    def __init__(self, num_nodes, gamma=0.20, sigma=0.25, dt=0.04):
        super(ThermodynamicIsingNetwork, self).__init__()
        self.num_nodes = num_nodes
        self.gamma = gamma    # Viscous dissipation factor (friction)
        self.sigma = sigma    # Environment temperature amplitude (thermal noise)
        self.dt = dt          # Time integration interval
        
        # J matrix explicitly set by the graph configuration (no gradients required)
        self.register_buffer('J', torch.zeros(num_nodes, num_nodes))

    def set_problem_topology(self, adjacency_matrix):
        """
        Maps a problem graph directly to the physical system coupling weights.
        Antiferromagnetic coupling (negative weights) forces adjacent nodes into opposite states.
        """
        self.J = -1.0 * adjacency_matrix.float()

    def forward(self, steps=250, batch_size=4):
        device = self.J.device
        
        # Initialize continuous macrostates with near-zero random thermal offsets
        x = (torch.rand(batch_size, self.num_nodes, device=device) - 0.5) * 0.2
        v = (torch.rand(batch_size, self.num_nodes, device=device) - 0.5) * 0.1
        
        # Matrix buffer tracking trajectories over time for structural analysis
        trajectory = []

        # Time evolution using Stochastic Euler-Maruyama integration
        for step in range(steps):
            # Dynamic annealing schedule: gradually lower thermal noise over time
            current_sigma = self.sigma * (1.0 - (step / steps) * 0.8)
            
            # Non-linear restoring force derived from the local double-well profile
            f_internal = x - torch.pow(x, 3)
            
            # Entangled spatial coupling force vector from neighboring cells
            f_coupling = torch.matmul(x, self.J.T)
            
            # Ambient thermal fluctuation input matrix
            noise = torch.randn_like(x) * current_sigma * (self.dt ** 0.5)
            
            # State vector evaluation equations (Kinetic updates)
            dv = (f_internal + f_coupling - self.gamma * v) * self.dt + noise
            v = v + dv
            x = x + v * self.dt
            
            # Prevent numerical boundary escape
            x = torch.clamp(x, -2.2, 2.2)
            trajectory.append(x.clone())
            
        return x, torch.stack(trajectory)

# =============================================================================
# 2. THE GENERATIVE CONTRASTIVE DIVERGENCE MODEL
# =============================================================================

class GenerativeThermodynamicNetwork(nn.Module):
    """
    An energy-based generative model using Contrastive Divergence.
    Learns to sculpt its internal energy landscape to match target data distributions.
    """
    def __init__(self, num_nodes, gamma=0.25, sigma=0.40, dt=0.04):
        super(GenerativeThermodynamicNetwork, self).__init__()
        self.num_nodes = num_nodes
        self.gamma = gamma    # Energy dissipation factor
        self.sigma = sigma    # Peak ambient temperature
        self.dt = dt
        
        # Trainable symmetric coupling matrix J (diagonal initialized to zero)
        raw_j = torch.randn(num_nodes, num_nodes) * 0.1
        self.J_raw = nn.Parameter((raw_j + raw_j.T) / 2.0)
        self.bias = nn.Parameter(torch.zeros(num_nodes))

    def get_coupling_matrix(self):
        # Maintain zero-diagonal symmetry to prevent self-loops
        J = (self.J_raw + self.J_raw.T) / 2.0
        mask = torch.eye(self.num_nodes, device=J.device)
        return J * (1.0 - mask)

    def evolve_system(self, steps=150, clamped_mask=None, clamped_states=None, batch_size=32):
        device = self.J_raw.device
        J = self.get_coupling_matrix()
        
        # Initialize continuous macrostates near the unstable center ridge (0.0)
        x = (torch.rand(batch_size, self.num_nodes, device=device) - 0.5) * 0.2
        v = (torch.rand(batch_size, self.num_nodes, device=device) - 0.5) * 0.1
        
        for _ in range(steps):
            # Apply data clamping to visible boundary nodes if a mask is provided
            if clamped_mask is not None and clamped_states is not None:
                x = torch.where(clamped_mask, clamped_states, x)
                v = torch.where(clamped_mask, torch.zeros_like(v), v)

            # Continuous non-linear dynamics equations
            f_internal = x - torch.pow(x, 3)
            f_coupling = torch.matmul(x, J.T) + self.bias
            noise = torch.randn_like(x) * self.sigma * (self.dt ** 0.5)
            
            dv = (f_internal + f_coupling - self.gamma * v) * self.dt + noise
            v = v + dv
            x = x + v * self.dt
            
            x = torch.clamp(x, -2.2, 2.2)
            
        if clamped_mask is not None and clamped_states is not None:
            x = torch.where(clamped_mask, clamped_states, x)
            
        return x

# =============================================================================
# 3. THE HYBRID DIGITAL-ANALOG HIERARCHICAL MODEL
# =============================================================================

class HierarchicalThermodynamicNetwork(nn.Module):
    """
    A hybrid architecture where standard digital layers (Encoder/Decoder) 
    sandwich the continuous thermodynamic core. Acts as a drop-in PyTorch module.
    """
    def __init__(self, d_in, d_l0, d_l1, d_l2, d_out, gamma=0.25, sigma=0.30, dt=0.04):
        super(HierarchicalThermodynamicNetwork, self).__init__()
        self.d_l0 = d_l0  # Visible Layer 0 Dimension
        self.d_l1 = d_l1  # Hidden Layer 1 Dimension
        self.d_l2 = d_l2  # Deep Latent Layer 2 Dimension
        self.total_nodes = d_l0 + d_l1 + d_l2
        
        self.gamma = gamma
        self.sigma = sigma
        self.dt = dt

        # --- DIGITAL INTERFACE: ENCODER ---
        # Maps regular numerical data to target continuous well coordinates
        self.digital_encoder = nn.Sequential(
            nn.Linear(d_in, d_l0),
            nn.Tanh() # Restricts initialization to the boundaries of the double-well landscape
        )

        # --- THERMODYNAMIC CORE: TRAINABLE STRUCTURAL PARAMETERS ---
        raw_j = torch.randn(self.total_nodes, self.total_nodes) * 0.05
        self.J_raw = nn.Parameter((raw_j + raw_j.T) / 2.0)
        self.bias = nn.Parameter(torch.zeros(self.total_nodes))

        # --- DIGITAL INTERFACE: DECODER ---
        # Processes finalized spatial features into normal categorical output probabilities
        self.digital_decoder = nn.Linear(self.total_nodes, d_out)

    def _get_hierarchical_coupling(self):
        """
        Enforces strict symmetric constraints and clears self-loops.
        Ensures energy functions remain mathematically stable across layered jumps.
        """
        J = (self.J_raw + self.J_raw.T) / 2.0
        mask = torch.eye(self.total_nodes, device=J.device)
        return J * (1.0 - mask)

    def evolve_core_physics(self, inputs_clamped_l0, steps=120):
        batch_size = inputs_clamped_l0.size(0)
        device = self.J_raw.device
        J = self._get_hierarchical_coupling()

        # Initialize full system state tensors (All layers concatenated)
        x = (torch.rand(batch_size, self.total_nodes, device=device) - 0.5) * 0.1
        v = (torch.rand(batch_size, self.total_nodes, device=device) - 0.5) * 0.05

        # Create a boolean selection template to force-clamp only Layer 0
        clamped_mask = torch.zeros(batch_size, self.total_nodes, dtype=torch.bool, device=device)
        clamped_mask[:, :self.d_l0] = True

        for step in range(steps):
            # Enforce Layer 0 clamping based on input data
            x = torch.where(clamped_mask, inputs_clamped_l0.repeat(1, 1), x)
            v = torch.where(clamped_mask, torch.zeros_like(v), v)

            # Continuous physical dynamics calculations
            f_internal = x - torch.pow(x, 3)  # Restoring force within the wells
            f_coupling = torch.matmul(x, J.T) + self.bias
            noise = torch.randn_like(x) * self.sigma * (self.dt ** 0.5)

            dv = (f_internal + f_coupling - self.gamma * v) * self.dt + noise
            v = v + dv
            x = x + v * self.dt
            
            x = torch.clamp(x, -2.2, 2.2)

        # Final pass verification ensuring visible values remained locked
        x = torch.where(clamped_mask, inputs_clamped_l0, x)
        return x

    def forward(self, raw_digital_data, steps=120):
        # 1. Encode digital data into initial continuous coordinates
        clamped_l0 = self.digital_encoder(raw_digital_data)

        # 2. Allow physics engine to settle into an energy-optimized state
        finalized_structural_states = self.evolve_core_physics(clamped_l0, steps=steps)

        # 3. Decode spatial vectors back into regular categorical outputs
        prediction_logits = self.digital_decoder(finalized_structural_states)
        return prediction_logits


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_random_graph(num_nodes, edge_probability=0.4):
    """Generates a symmetric adjacency matrix representing a random problem graph."""
    adj = torch.rand(num_nodes, num_nodes) < edge_probability
    adj = adj.int()
    adj = (adj + adj.T > 0).int() # Enforce exact symmetry
    adj.fill_diagonal_(0)         # Remove node self-loops
    return adj.float()

def calculate_cut_value(state_assignments, adjacency_matrix):
    """Computes the total structural cut weight of the current configuration."""
    num_nodes = adjacency_matrix.shape[0]
    cut_value = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i, j] > 0 and state_assignments[i] != state_assignments[j]:
                cut_value += 1
    return cut_value


# =============================================================================
# EXECUTION & VERIFICATION PIPELINE
# =============================================================================
if __name__ == "__main__":
    
    print("\n=======================================================")
    print("1. TESTING COMBINATORIAL SOLVER (MAX-CUT)")
    print("=======================================================")
    nodes_count = 10
    problem_graph = generate_random_graph(num_nodes=nodes_count, edge_probability=0.5)
    print(f"Total internal graph edges present to resolve: {int(problem_graph.sum() / 2)}")
    
    computing_fabric = ThermodynamicIsingNetwork(num_nodes=nodes_count, gamma=0.25, sigma=0.35, dt=0.04)
    computing_fabric.set_problem_topology(problem_graph)
    final_positions, _ = computing_fabric(steps=300, batch_size=1)
    
    final_positions_vector = final_positions[0].detach().cpu()
    binary_solution = (final_positions_vector > 0).int().numpy()
    achieved_cut = calculate_cut_value(binary_solution, problem_graph)
    print(f"Calculated Cut Value: {achieved_cut} crossed edges.")


    print("\n=======================================================")
    print("2. TESTING GENERATIVE THERMODYNAMIC NETWORK (CONTRASTIVE)")
    print("=======================================================")
    num_visible = 4
    num_hidden = 4
    total_nodes = num_visible + num_hidden

    model = GenerativeThermodynamicNetwork(num_nodes=total_nodes, sigma=0.35)
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    
    # Target dataset: binary stable attractors
    target_data = torch.cat([
        torch.ones(16, num_visible),
        -1.0 * torch.ones(16, num_visible)
    ], dim=0)

    print("Running Contrastive Thermodynamic Gradient Descent for 20 epochs...")
    for epoch in range(20):
        permutation = torch.randperm(target_data.size(0))
        batch_visible = target_data[permutation[:16]]
        b_size = batch_visible.size(0)
        
        clamped_mask = torch.zeros(b_size, total_nodes, dtype=torch.bool)
        clamped_mask[:, :num_visible] = True  
        clamped_states = torch.zeros(b_size, total_nodes)
        clamped_states[:, :num_visible] = batch_visible
        
        # Clamped Phase vs Free Phase
        x_clamped = model.evolve_system(steps=100, clamped_mask=clamped_mask, clamped_states=clamped_states, batch_size=b_size)
        x_free = model.evolve_system(steps=100, clamped_mask=None, clamped_states=None, batch_size=b_size)
        
        corr_clamped = torch.matmul(x_clamped.T, x_clamped) / b_size
        corr_free = torch.matmul(x_free.T, x_free) / b_size
        
        loss = torch.mean(torch.pow(corr_clamped - corr_free, 2))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Final Energy Divergence Loss: {loss.item():.5f}")


    print("\n=======================================================")
    print("3. TESTING HYBRID HIERARCHICAL NETWORK")
    print("=======================================================")
    sample_digital_batch = torch.randn(16, 32)
    mep_system = HierarchicalThermodynamicNetwork(
        d_in=32,    # Source Input Features
        d_l0=16,    # Visible Boundary Layer 0
        d_l1=32,    # Latent Hidden Layer 1
        d_l2=16,    # Deep Latent Layer 2
        d_out=3     # Decoded Target Class Target Array
    )
    
    output_predictions = mep_system(sample_digital_batch, steps=100)
    print("Target Output Class Logits Matrix Shape:", output_predictions.shape)
    print("First sample prediction distribution across target categories:\n", 
          F.softmax(output_predictions[0], dim=-1).detach().numpy())
    print("=======================================================\n")
