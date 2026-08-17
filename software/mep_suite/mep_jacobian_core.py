import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd.functional import jvp
import numpy as np
import copy
import time

# =====================================================================
# V5.0 PREREGISTRATION PROTOCOL: JACOBIAN TOPOLOGICAL TETHERING
# Benchmark: Split MNIST (Task A: 0-4, Task B: 5-9)
# Seeds: 20 independent random seeds
# =====================================================================

SEEDS = 20
EPOCHS_PER_TASK = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.01
LAMBDA_EWC = 400.0   # EWC Penalty weight
LAMBDA_JVP = 1000.0  # Topological Tether weight

# --- 1. The Core Substrate ---
class MEPTopologicalNetwork(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super(MEPTopologicalNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- 2. The P Operator (Jacobian-Vector Product Tether) ---
def compute_topological_tether(model_current, model_old, x_anchors):
    model_old.eval()
    # Generate random perturbation vector in the input space
    v = torch.randn_like(x_anchors)

    def func_current(inputs):
        return model_current(inputs)
        
    def func_old(inputs):
        return model_old(inputs)

    # Compute Jacobian-Vector Products (JVP)
    # create_graph=True allows backprop through the derivative calculation!
    _, jvp_current = jvp(func_current, x_anchors, v=v, create_graph=True)
    
    with torch.no_grad():
        _, jvp_old = jvp(func_old, x_anchors, v=v, create_graph=False)

    # MSE Loss between how the current and old models bend the random vector
    return F.mse_loss(jvp_current, jvp_old)

# --- 3. EWC Implementation (Baseline) ---
def compute_ewc_fisher(model, dataloader):
    fisher_dict = {n: torch.zeros_like(p.data) for n, p in model.named_parameters() if p.requires_grad}
    model.eval()
    criterion = nn.CrossEntropyLoss()
    for data, target in dataloader:
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher_dict[n] += (p.grad.data ** 2) / len(dataloader)
    return fisher_dict

# --- 4. Data Preparation ---
def get_split_dataloaders():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    def filter_dataset(dataset, classes):
        indices = [i for i, target in enumerate(dataset.targets) if target in classes][:1000]
        subset = torch.utils.data.Subset(dataset, indices)
        return torch.utils.data.DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

    return (
        filter_dataset(trainset, [0, 1, 2, 3, 4]), filter_dataset(testset,  [0, 1, 2, 3, 4]),
        filter_dataset(trainset, [5, 6, 7, 8, 9]), filter_dataset(testset,  [5, 6, 7, 8, 9])
    )

def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100.0 * correct / total

# --- 5. The Crucible ---
def run_crucible():
    print("==================================================")
    print("MEP V5.0 PREREGISTRATION: JACOBIAN TETHERING")
    print(f"Criteria: {SEEDS} Seeds | Target: >80% Retention")
    print("==================================================\n")

    task_A_train, task_A_test, task_B_train, task_B_test = get_split_dataloaders()
    
    # Grab a small batch of unlabeled Anchor Coordinates for the Jacobian Tether
    # We flatten them to match the network input dimensions
    x_anchors = next(iter(task_A_train))[0].view(BATCH_SIZE, -1)

    results = {"Naive_SGD": [], "EWC_Baseline": [], "MEP_Jacobian": []}

    for seed in range(SEEDS):
        torch.manual_seed(seed)
        if (seed + 1) % 5 == 0: 
            print(f"Processing Seed {seed + 1}/{SEEDS}...")

        base_model = MEPTopologicalNetwork()
        criterion = nn.CrossEntropyLoss()
        
        # --- TRAIN TASK A ---
        model_A = copy.deepcopy(base_model)
        optimizer_A = optim.SGD(model_A.parameters(), lr=LEARNING_RATE)
        model_A.train()
        for _ in range(EPOCHS_PER_TASK):
            for data, target in task_A_train:
                optimizer_A.zero_grad()
                loss = criterion(model_A(data), target)
                loss.backward()
                optimizer_A.step()
                
        acc_A_initial = evaluate(model_A, task_A_test)
        
        # Save EWC components
        weights_A = {n: p.data.clone() for n, p in model_A.named_parameters()}
        fisher_A = compute_ewc_fisher(model_A, task_A_train)

        # --- TEST 1: Naive SGD ---
        model_naive = copy.deepcopy(model_A)
        opt_naive = optim.SGD(model_naive.parameters(), lr=LEARNING_RATE)
        model_naive.train()
        for _ in range(EPOCHS_PER_TASK):
            for data, target in task_B_train:
                opt_naive.zero_grad()
                loss = criterion(model_naive(data), target)
                loss.backward()
                opt_naive.step()
        results["Naive_SGD"].append(evaluate(model_naive, task_A_test) / acc_A_initial * 100.0)

        # --- TEST 2: EWC Baseline ---
        model_ewc = copy.deepcopy(model_A)
        opt_ewc = optim.SGD(model_ewc.parameters(), lr=LEARNING_RATE)
        model_ewc.train()
        for _ in range(EPOCHS_PER_TASK):
            for data, target in task_B_train:
                opt_ewc.zero_grad()
                loss_B = criterion(model_ewc(data), target)
                ewc_penalty = 0
                for n, p in model_ewc.named_parameters():
                    ewc_penalty += (fisher_A[n] * (p - weights_A[n]) ** 2).sum()
                loss = loss_B + (LAMBDA_EWC / 2) * ewc_penalty
                loss.backward()
                opt_ewc.step()
        results["EWC_Baseline"].append(evaluate(model_ewc, task_A_test) / acc_A_initial * 100.0)

        # --- TEST 3: MEP Jacobian Tethering ---
        model_mep = copy.deepcopy(model_A)
        opt_mep = optim.SGD(model_mep.parameters(), lr=LEARNING_RATE)
        model_mep.train()
        
        for _ in range(EPOCHS_PER_TASK):
            for data, target in task_B_train:
                opt_mep.zero_grad()
                
                # 1. Standard Task B Loss
                loss_B = criterion(model_mep(data), target)
                
                # 2. Compute the Topological Tether (Geometric Scar)
                tether_penalty = compute_topological_tether(model_mep, model_A, x_anchors)
                
                # 3. Combine and Backpropagate
                loss = loss_B + LAMBDA_JVP * tether_penalty
                loss.backward()
                opt_mep.step()
                
        results["MEP_Jacobian"].append(evaluate(model_mep, task_A_test) / acc_A_initial * 100.0)

    print(f"\n==================================================")
    print(f"🏁 FINAL V5.0 RESULTS (Task A Relative Retention) 🏁")
    print(f"==================================================")
    
    for name, metrics in results.items():
        mean_acc = np.mean(metrics)
        std_acc = np.std(metrics)
        print(f"{name}:")
        print(f"  Mean Retention: {mean_acc:.2f}%  (±{std_acc:.2f}%)")

    print("\nFailure Condition Check:")
    print("1. If MEP_Jacobian < 80.00%, hypothesis FAILED.")
    print("2. If MEP_Jacobian <= EWC_Baseline, hypothesis FAILED.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    run_crucible()
