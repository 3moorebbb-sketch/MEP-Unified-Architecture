import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd.functional import jvp
import numpy as np
import scipy.stats as stats
import copy

# =====================================================================
# V5.1 PREREGISTRATION PROTOCOL: SPLIT-CIFAR10 (Unseen Benchmark)
# Focus: Strict Memory Budgets, Harmonic Means, Statistical Rigor
# =====================================================================

SEEDS = 10
EPOCHS_PER_TASK = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.01

# The Strict Memory Budget: Both baselines get exactly 50 examples.
MEMORY_BUDGET_SIZE = 50  
LAMBDA_JVP = 50.0  

class ReplayBuffer:
    """Experience Replay Buffer to ensure a fair memory-budget baseline."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.x_data = []
        self.y_targets = []

    def add_batch(self, x, y):
        for i in range(x.size(0)):
            if len(self.x_data) < self.capacity:
                self.x_data.append(x[i].clone().detach())
                self.y_targets.append(y[i].clone().detach())

    def sample(self, batch_size):
        if len(self.x_data) == 0:
            return None, None
        indices = np.random.choice(len(self.x_data), min(batch_size, len(self.x_data)), replace=False)
        x_batch = torch.stack([self.x_data[i] for i in indices])
        y_batch = torch.stack([self.y_targets[i] for i in indices])
        return x_batch, y_batch

def calculate_harmonic_mean(acc_a, acc_b):
    if acc_a == 0 or acc_b == 0:
        return 0.0
    return 2 * (acc_a * acc_b) / (acc_a + acc_b)

def print_rigorous_statistics(name, mep_scores, baseline_scores):
    mep_mean = np.mean(mep_scores)
    base_mean = np.mean(baseline_scores)
    t_stat, p_val = stats.ttest_ind(mep_scores, baseline_scores, equal_var=False)
    pooled_std = np.sqrt((np.std(mep_scores)**2 + np.std(baseline_scores)**2) / 2)
    cohens_d = (mep_mean - base_mean) / pooled_std if pooled_std > 0 else 0.0
    
    df = len(mep_scores) + len(baseline_scores) - 2
    se_diff = np.sqrt(np.var(mep_scores, ddof=1)/len(mep_scores) + np.var(baseline_scores, ddof=1)/len(baseline_scores))
    ci_margin = stats.t.ppf(0.975, df) * se_diff
    mean_diff = mep_mean - base_mean
    ci_lower, ci_upper = mean_diff - ci_margin, mean_diff + ci_margin

    print(f"\n--- Statistical Adjudication: MEP vs {name} ---")
    print(f"MEP Harmonic Mean:      {mep_mean:.2f}%")
    print(f"Baseline Harmonic Mean: {base_mean:.2f}%")
    print(f"Difference:             {mean_diff:+.2f}%  [95% CI: {ci_lower:+.2f} to {ci_upper:+.2f}]")
    print(f"Welch's t-test p-value: {p_val:.4e} {'(Significant < 0.05)' if p_val < 0.05 else '(Not Significant)'}")
    print(f"Cohen's d Effect Size:  {cohens_d:.2f}")

class SimpleCNN(nn.Module):
    """A standard small CNN for CIFAR-10."""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def compute_topological_tether(model_current, model_old, x_anchors):
    model_old.eval()
    v = torch.randn_like(x_anchors)
    def func_current(inputs): return model_current(inputs)
    def func_old(inputs): return model_old(inputs)
    _, jvp_current = jvp(func_current, x_anchors, v=v, create_graph=True)
    with torch.no_grad():
        _, jvp_old = jvp(func_old, x_anchors, v=v, create_graph=False)
    return F.mse_loss(jvp_current, jvp_old)

def get_split_cifar10_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    def filter_dataset(dataset, classes):
        indices = [i for i, target in enumerate(dataset.targets) if target in classes][:1500]
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

def run_crucible():
    print("==================================================")
    print("V5.1 PROSPECTIVE CRUCIBLE: SPLIT-CIFAR10")
    print(f"Strict Memory Budget: {MEMORY_BUDGET_SIZE} Samples")
    print("==================================================\n")

    task_A_train, task_A_test, task_B_train, task_B_test = get_split_cifar10_dataloaders()
    results_mep, results_er = [], []

    for seed in range(SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n[Seed {seed + 1}/{SEEDS}] Training Base Task A...")
        
        base_model = SimpleCNN()
        criterion = nn.CrossEntropyLoss()
        
        # --- TRAIN TASK A ---
        model_A = copy.deepcopy(base_model)
        optimizer_A = optim.SGD(model_A.parameters(), lr=LEARNING_RATE)
        for _ in range(EPOCHS_PER_TASK):
            for data, target in task_A_train:
                optimizer_A.zero_grad()
                loss = criterion(model_A(data), target)
                loss.backward()
                optimizer_A.step()
                
        # --- EXTRACT BUDGETED MEMORY (50 points) ---
        buffer_A = ReplayBuffer(MEMORY_BUDGET_SIZE)
        x_anchors_list = []
        for data, target in task_A_train:
            buffer_A.add_batch(data, target)
            for i in range(data.size(0)):
                if len(x_anchors_list) < MEMORY_BUDGET_SIZE:
                    x_anchors_list.append(data[i].clone())
            if len(x_anchors_list) >= MEMORY_BUDGET_SIZE:
                break
        x_anchors_A = torch.stack(x_anchors_list)

        # --- TEST 1: Experience Replay (ER) Baseline ---
        print(f"[Seed {seed + 1}] Testing Experience Replay Baseline...")
        model_er = copy.deepcopy(model_A)
        opt_er = optim.SGD(model_er.parameters(), lr=LEARNING_RATE)
        for _ in range(EPOCHS_PER_TASK):
            for data, target in task_B_train:
                opt_er.zero_grad()
                loss_B = criterion(model_er(data), target)
                
                # Sample 16 points from the buffer of 50
                x_buf, y_buf = buffer_A.sample(16)
                loss_replay = criterion(model_er(x_buf), y_buf) if x_buf is not None else 0
                
                loss = loss_B + loss_replay
                loss.backward()
                opt_er.step()
                
        er_retention = evaluate(model_er, task_A_test)
        er_acquisition = evaluate(model_er, task_B_test)
        er_harmonic = calculate_harmonic_mean(er_retention, er_acquisition)
        results_er.append(er_harmonic)

        # --- TEST 2: MEP Jacobian Tether ---
        print(f"[Seed {seed + 1}] Testing MEP Jacobian Tether...")
        model_mep = copy.deepcopy(model_A)
        opt_mep = optim.SGD(model_mep.parameters(), lr=LEARNING_RATE)
        
        for _ in range(EPOCHS_PER_TASK):
            for data, target in task_B_train:
                opt_mep.zero_grad()
                loss_B = criterion(model_mep(data), target)
                tether_penalty = compute_topological_tether(model_mep, model_A, x_anchors_A)
                loss = loss_B + (LAMBDA_JVP * tether_penalty)
                loss.backward()
                opt_mep.step()
                
        mep_retention = evaluate(model_mep, task_A_test)
        mep_acquisition = evaluate(model_mep, task_B_test)
        mep_harmonic = calculate_harmonic_mean(mep_retention, mep_acquisition)
        results_mep.append(mep_harmonic)
        
        print(f"  -> ER Harmonic:  {er_harmonic:.2f}% (Ret: {er_retention:.1f}%, Acq: {er_acquisition:.1f}%)")
        print(f"  -> MEP Harmonic: {mep_harmonic:.2f}% (Ret: {mep_retention:.1f}%, Acq: {mep_acquisition:.1f}%)")

    # Final Preregistered Statistical Check
    print_rigorous_statistics("Experience Replay (50-Sample Budget)", results_mep, results_er)

if __name__ == "__main__":
    run_crucible()
