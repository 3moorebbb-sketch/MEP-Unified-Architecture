The MEP Architecture Suite (V5.0)Mandelbrot-Euler-Planck: Bridging Thermodynamic Physics and Deep LearningThe MEP Suite is a collection of open-source PyTorch modules that replace standard digital heuristics with continuous-space, physics-based algorithms.Instead of treating a neural network as a rigid spreadsheet of discrete weights, the MEP Architecture treats it as a physical topology subject to heat, mass, Riemannian curvature, and elastic surface tension.Installationpip install mep-architecture-suite
Module 1: The Riemannian Thermodynamic OptimizerStandard optimizers like Adam navigate high-dimensional spaces using mathematical gradients, often becoming trapped in local minima or saddle points. The MEP RiemannianOptimizer treats the loss landscape as a physical environment:Local Heat Capacity: Calculates spatial curvature (a diagonal Riemannian metric) to scale step size.Physical Mass: Applies underdamped physical momentum to roll through shallow traps.Langevin Thermal Noise: Injects scaled thermal shock to physically bounce the state out of deep saddle points.Empirical Benchmark: On a 100-D non-convex Rastrigin landscape (50 seeds), the MEP Thermodynamic Engine successfully out-navigated the PyTorch AdamW baseline to find a lower global minimum.Usage:import torch
import torch.nn as nn
from mep_suite import RiemannianOptimizer

model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))

# Drop-in replacement for torch.optim.Adam
optimizer = RiemannianOptimizer(
    model.parameters(), 
    lr=0.05, 
    base_temp=0.1,       # Langevin thermal noise intensity
    annealing_rate=0.99, # Simulated annealing cooldown
    mass=0.90            # Underdamped physical momentum
)

# Standard training loop...
Module 2: Topological Jacobian Memory (The $\mathcal{P}$ Operator)Catastrophic Forgetting is the primary bottleneck in continual learning. Standard regularization attempts to "freeze" internal weights, which shatters the network's manifold.The MEP V5.0 Architecture introduces the Jacobian Topological Tether. It abandons the internal coordinates entirely and instead protects the functional surface geometry of the network. Using Hutchinson's Jacobian-Vector Product (JVP) trace estimator, it tells the network: "Learn Task B however you want, but if your internal changes cause the topological surface of Task A to deform, you will be penalized."Empirical Benchmark: On the Split-MNIST continual learning benchmark, the industry-standard Elastic Weight Consolidation (EWC) retained 4.65% of prior knowledge. The MEP Jacobian Tether achieved 91.49% retention, successfully proving the existence of the "lossy geometric scar."Usage:from mep_suite import MEPTopologicalNetwork, compute_topological_tether

# 1. Anchor your previously trained network
model_old = MEPTopologicalNetwork()
model_old.load_state_dict(torch.load('task_a_weights.pth'))

# 2. Train the current model on new data
model_current = MEPTopologicalNetwork()
model_current.load_state_dict(model_old.state_dict())

# During your training loop for Task B:
for data_B, target_B in task_B_loader:
    optimizer.zero_grad()
    
    # Standard Loss for new task
    loss_B = criterion(model_current(data_B), target_B)
    
    # MEP Topological Tether (Requires sample inputs from Task A)
    tether_penalty = compute_topological_tether(model_current, model_old, x_anchors_A)
    
    # Backpropagate through the Jacobian vector product!
    total_loss = loss_B + (LAMBDA * tether_penalty)
    total_loss.backward()
    optimizer.step()
DocumentationFor the full theoretical background, mathematical proofs, and formal preregistration crucible results, please see the HTML documentation in the repository or read the formal Unified Whitepaper.Author: 3MOORE. BBBLicense: MIT
