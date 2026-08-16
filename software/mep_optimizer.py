import torch
from torch.optim.optimizer import Optimizer
import math

class ThermodynamicOptimizer(Optimizer):
    """
    MEP Architecture: A Drop-In PyTorch Optimizer.
    Replaces Adam/SGD with a physically grounded Langevin dynamic.
    Treats the neural network loss landscape as a physical topology, 
    applying Landauer dissipation and thermal noise to kick the model 
    out of local minima.
    """
    def __init__(self, params, lr=1e-3, gamma=0.01, base_temp=0.1, annealing_rate=0.99):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = dict(lr=lr, gamma=gamma, base_temp=base_temp, annealing_rate=annealing_rate)
        super(ThermodynamicOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            gamma = group['gamma']
            temp = group['base_temp']
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # The gradient represents the "force" of the topological landscape
                d_p = p.grad
                
                # 1. Landauer Dissipation (Entropy Tax)
                # Acts as a physically grounded weight decay, pulling weights toward stability
                dissipation = gamma * p
                
                # 2. Langevin Thermal Noise
                # Injects scaled random kinetic energy to bounce out of shallow potholes
                noise = torch.randn_like(p) * math.sqrt(2 * gamma * temp * lr)
                
                # Update the state based on the open dissipative map
                # p_new = p_old - lr * (Gradient + Dissipation) + Thermal Noise
                p.add_(d_p + dissipation, alpha=-lr)
                p.add_(noise)
                
            # Cool the environment slightly after each step (Simulated Annealing)
            group['base_temp'] *= group['annealing_rate']

        return loss
