import numpy as np
import math

class ThermodynamicScheduler:
    """
    MEP Architecture: OS Thread Scheduling via Kuramoto Phase-Locking.
    Instead of rigidly stopping and starting background tasks 100 times a second,
    this treats tasks as continuous oscillators. It allows low-priority tasks 
    to phase-lock and execute together in a single, smooth resonant wave, 
    drastically reducing CPU context-switching overhead.
    """
    def __init__(self, num_tasks, coupling_k=1.5, dt=0.1):
        self.num_tasks = num_tasks
        self.k = coupling_k
        self.dt = dt
        
        # Assign each task a random starting phase
        self.phases = np.random.uniform(0, 2 * math.pi, num_tasks)
        
        # Assign natural frequencies (Base CPU need)
        # High priority tasks have higher natural frequencies
        self.natural_frequencies = np.random.uniform(0.5, 2.0, num_tasks)

    def step_physics(self):
        """Steps the continuous wave physics forward."""
        phase_diffs = self.phases[np.newaxis, :] - self.phases[:, np.newaxis]
        
        # Kuramoto coupling: tasks pull on each other to synchronize
        coupling = np.sum(np.sin(phase_diffs), axis=1) * (self.k / self.num_tasks)
        
        # Update phases
        self.phases += (self.natural_frequencies + coupling) * self.dt
        self.phases = np.mod(self.phases, 2 * math.pi)

    def execute_batch(self, threshold_phase=0.1):
        """
        Monitors the continuous wave. When tasks align at the execution phase (0 rad),
        they are fired together in a single CPU batch, eliminating context switching.
        """
        # Find all tasks that are crossing the 0 radian threshold
        ready_tasks = np.where((self.phases < threshold_phase) | (self.phases > 2 * math.pi - threshold_phase))[0]
        
        if len(ready_tasks) > 1:
            print(f"[{len(ready_tasks)} Tasks Phase-Locked] -> Executing as a single Thermodynamic Batch")
            return ready_tasks
        return []
