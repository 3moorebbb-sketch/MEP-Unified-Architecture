import torch
import torch.nn as nn
import math

class ThermodynamicSwarm(nn.Module):
    """
    A continuous-time, spatially dynamic PyTorch emulator for the MEP Phononic Swarm.
    Replaces O(N^2) loops with highly optimized, vectorized tensor broadcasting.
    """
    def __init__(self, num_nodes, space_size=100.0, dt=0.04):
        super(ThermodynamicSwarm, self).__init__()
        self.num_nodes = num_nodes
        self.space_size = space_size # Size of the 2D bounding box
        self.dt = dt
        
        self.gamma = 0.08       # Dissipation term (Entropy Tax)
        self.kbT_base = 0.02    # Boltzmann constant * Temp for ambient thermal noise
        self.coupling_k = 150.0 # Base strength of Kuramoto entanglement
        self.epsilon = 1e-5     # Prevents division by zero in 1/d^2 calculations

    # State Tensors (Registered as buffers so they move to GPU automatically)
        # 1. Spatial Positions (X, Y)
        self.register_buffer('positions', torch.rand(num_nodes, 2) * space_size)
        
        # 2. Velocities (vX, vY) - Ambient drift
        self.register_buffer('velocities', (torch.rand(num_nodes, 2) - 0.5) * 2.0)
        
        # 3. Thermodynamic State (Heat/Magnitude: 0.0 to 1.0)
        self.register_buffer('heat', torch.zeros(num_nodes))
        
        # 4. Euler Phase Angles (theta)
        self.register_buffer('phases', torch.rand(num_nodes) * 2 * math.pi)
        
        # 5. Base Identity (to simulate cooling/drift back to resting state)
        self.register_buffer('base_phases', self.phases.clone())

    def forward(self, steps=100, inject_heat_indices=None):
        """
        Evolves the swarm forward in time.
        inject_heat_indices: Optional list of node indices to forcefully heat up (simulating sensory input).
        """
        device = self.positions.device
        
        # Inject initial heat if provided
        if inject_heat_indices is not None:
            self.heat[inject_heat_indices] = 1.0

        history_sync = [] # Track global synchronization over time

        for _ in range(steps):
            # 1. Spatial Kinematics (Drift and Bounds Wrapping)
            self.positions += self.velocities * self.dt
            self.positions = torch.remainder(self.positions, self.space_size)
            
            # Environmental temp is hot at x=0, cold at x=space_size
            env_temp = torch.clamp(1.0 - (self.positions[:, 0] / self.space_size), 0.1, 1.0)
            
            # Harvest free energy from gradient, but pay Landauer dissipation tax
            self.heat = self.heat + (env_temp * 0.03) - (self.gamma * self.heat)
            self.heat = torch.clamp(self.heat, 0.0, 1.0)
            
            # 3. Vectorized Distance Matrix Calculation
            # Computes the pairwise distance between ALL nodes simultaneously
            # Shape: (N, 1, 2) - (1, N, 2) -> (N, N, 2) -> norm -> (N, N)
            diffs = self.positions.unsqueeze(1) - self.positions.unsqueeze(0)
            dist_sq = torch.sum(diffs ** 2, dim=-1) + self.epsilon
            
            # 4. Construct the Dynamic Adjacency Matrix (W)
            # Coupling strength drops by 1/d^2 and is multiplied by the emitting node's heat
            # W_ij represents how strongly node j influences node i
            W = (self.coupling_k / dist_sq) * self.heat.unsqueeze(0) # Broadcasting heat across rows
            
            # Remove self-loops (diagonal = 0)
            W.fill_diagonal_(0)

            # 5. Kuramoto Phase Entanglement (Matrix Multiplication)
            # phase_diffs shape: (N, N) where [i, j] is theta_j - theta_i
            phase_diffs = self.phases.unsqueeze(0) - self.phases.unsqueeze(1)
            
            # d_theta_i = sum_j ( W_ij * sin(theta_j - theta_i) )
            phase_updates = torch.sum(W * torch.sin(phase_diffs), dim=1)
            
            # Langevin Thermal Noise scales with dissipation (gamma) and local temperature
            langevin_noise = torch.randn_like(self.phases) * torch.sqrt(2.0 * self.gamma * env_temp * self.dt)
            
            # Apply updates with explicit dissipation (-gamma * phase) and thermal noise
            self.phases += (phase_updates - self.gamma * self.phases) * self.dt + langevin_noise
            
            # 6. Thermal Exchange (Lossy sharing constrained by the Second Law)
            # If nodes are close and in phase, they share heat, but sharing incurs an entropy tax
            alignment = torch.cos(phase_diffs)
            thermal_transfer = torch.sum(W * alignment * (self.heat.unsqueeze(0) - self.heat.unsqueeze(1)), dim=1)
            
            entropy_tax = 0.005 * torch.abs(thermal_transfer) # Irreversible logic penalty
            self.heat += (thermal_transfer * 0.01 * self.dt) - (entropy_tax * self.dt)
            self.heat = torch.clamp(self.heat, 0.0, 1.0)
            
            # 7. Baseline Drift (Cold nodes lose their acquired phase)
            is_cold = (self.heat < 0.3).float()
            phase_drift = (self.base_phases - self.phases) * 0.05
            self.phases += phase_drift * is_cold * self.dt

            # 8. Calculate Global Synchronization (Order Parameter 'r')
            # r = |(1/N) * sum(e^{i * theta})|
            r_x = torch.mean(torch.cos(self.phases))
            r_y = torch.mean(torch.sin(self.phases))
            sync_metric = torch.sqrt(r_x**2 + r_y**2)
            history_sync.append(sync_metric.item())
            
        return self.positions, self.heat, self.phases, history_sync

if __name__ == "__main__":
    import time
    print("Initializing PyTorch Tensor Swarm (10,000 Nodes)...")
    
    # We can easily simulate 10,000 nodes now, which would have crashed the JS loop.
    swarm = ThermodynamicSwarm(num_nodes=10000, space_size=500.0)
    
    # Randomly ignite 500 nodes to start the chain reaction
    ignition_nodes = torch.randint(0, 10000, (500,))
    
    print("Evolving physics for 200 time steps...")
    start_time = time.time()
    
    # Run the simulation
    pos, heat, phases, sync_history = swarm(steps=200, inject_heat_indices=ignition_nodes)
    
    end_time = time.time()
    
    print(f"Simulation completed in {end_time - start_time:.3f} seconds.")
    print(f"Initial Phase Synchronization: {sync_history[0]*100:.2f}%")
    print(f"Final Phase Synchronization:   {sync_history[-1]*100:.2f}%")
