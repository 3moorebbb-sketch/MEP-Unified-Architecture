import torch
from torch.optim.optimizer import Optimizer
import math

class ThermodynamicOptimizer(Optimizer):
    """
    MEP Architecture: A Drop-In PyTorch Optimizer.
    Replaces Adam/SGD with an Underdamped Langevin dynamic.
    Treats the neural network loss landscape as a physical topology, 
    applying Mass, Landauer dissipation, and thermal noise to kick the model 
    out of local minima.
    """
    def __init__(self, params, lr=1e-3, base_temp=0.1, annealing_rate=0.99, mass=0.9):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = dict(lr=lr, base_temp=base_temp, annealing_rate=annealing_rate, mass=mass)
        super(ThermodynamicOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            temp = group['base_temp']
            lr = group['lr']
            mass = group['mass']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Initialize physical state
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0 # NEW: Time tracker for bias correction
                    state['velocity'] = torch.zeros_like(p)
                    state['heat_capacity'] = torch.zeros_like(p) # NEW: Riemannian Metric
                    
                state['step'] += 1
                
                velocity = state['velocity']
                heat_cap = state['heat_capacity']
                d_p = p.grad
                
                # 1. Update Local Heat Capacity (Curvature of the landscape)
                heat_cap.mul_(0.999).addcmul_(d_p, d_p, value=0.001)
                
                # 2. Prevent Cold Start Explosion (Bias Correction)
                bias_correction = 1.0 - (0.999 ** state['step'])
                heat_cap_hat = heat_cap / bias_correction
                
                # 3. Adaptive Speed (Move faster in flat areas, slower in cliffs)
                adaptive_lr = lr / (torch.sqrt(heat_cap_hat) + 1e-8)
                
                # 4. Langevin Thermal Noise
                noise = torch.randn_like(p) * math.sqrt(temp)
                
                # 5. UNDERDAMPED PHYSICS: Apply force to Velocity (Mass)
                # Correctly scale the gradient by (1 - mass) so velocity doesn't runaway
                velocity.mul_(mass).add_(d_p, alpha=1.0 - mass)
                
                # 6. Update position using the adaptive curved space + thermal kick
                p.addcmul_(velocity, adaptive_lr, value=-1.0)
                p.add_(noise, alpha=lr)
                
            # Cool the environment slightly after each step (Simulated Annealing)
            group['base_temp'] *= group['annealing_rate']

        return loss
