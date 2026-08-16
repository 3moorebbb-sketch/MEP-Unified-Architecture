import torch
import torch.nn as nn
import time
from mep_optimizer import ThermodynamicOptimizer

# Set random seed so both optimizers face the exact same challenge
torch.manual_seed(42)

# =====================================================================
# 1. THE CHALLENGE: A Bumpy, Non-Linear Landscape
# =====================================================================
# We create a dataset that looks like a highly fluctuating wave.
# Standard optimizers often get stuck in the "valleys" of this data.
X = torch.linspace(-5, 5, 200).unsqueeze(1)
Y = X * torch.sin(5 * X) + torch.randn(X.shape) * 0.2

# =====================================================================
# 2. THE NEURAL NETWORK
# =====================================================================
class RegressionNet(nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Save initial weights so both optimizers start from the exact same starting line
initial_model = RegressionNet()
torch.save(initial_model.state_dict(), 'starting_weights.pth')

def run_benchmark(optimizer_name, epochs=250):
    """Trains the model and returns the final loss."""
    model = RegressionNet()
    model.load_state_dict(torch.load('starting_weights.pth')) # Reset to starting line
    criterion = nn.MSELoss()
    
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    elif optimizer_name == "MEP_Thermodynamic":
        # Using the MEP architecture! [RIEMANNIAN LANGEVIN PHYSICS]
        optimizer = ThermodynamicOptimizer(
            model.parameters(), 
            lr=0.05,             # Aggressive base speed
            base_temp=0.1,       # Warm thermal noise to bounce out of Adam's local minima
            annealing_rate=0.99, # Smooth cooling curve
            mass=0.90            # Heavy rolling momentum
        )

    print(f"\n--- Starting {optimizer_name} Training ---")
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, Y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")
            
    end_time = time.time()
    print(f"[{optimizer_name} finished in {end_time - start_time:.2f} seconds]")
    return loss.item()

# =====================================================================
# 3. THE HORSE RACE
# =====================================================================
if __name__ == "__main__":
    print("Initializing MEP Benchmark...")
    print("Dataset: Non-linear regression (Bumpy landscape)")
    
    # Run standard Adam
    adam_final_loss = run_benchmark("Adam")
    
    # Run MEP Thermodynamic Engine
    mep_final_loss = run_benchmark("MEP_Thermodynamic")
    
    print("\n========================================")
    print("🏁 FINAL RESULTS 🏁")
    print("========================================")
    print(f"Standard Adam Final Loss:       {adam_final_loss:.4f}")
    print(f"Thermodynamic Engine Final Loss: {mep_final_loss:.4f}")
    
    if mep_final_loss < adam_final_loss:
        print("\n🏆 MEP Architecture Wins!")
        print("The Langevin thermal noise successfully kicked the model out of local minima!")
    else:
        print("\nAdam won this round. Try adjusting the parameters in the MEP optimizer!")
