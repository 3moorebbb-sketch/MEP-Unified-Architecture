import numpy as np
import math
import time

class ThermodynamicScheduler:
    """
    MEP Architecture: OS Thread Scheduling via Kuramoto Phase-Locking.
    Treats background tasks as continuous oscillators. Allows tasks to naturally 
    phase-lock and execute in resonant waves, reducing CPU context-switching.
    """
    def __init__(self, num_tasks, coupling_k=1.5, dt=0.05):
        self.num_tasks = num_tasks
        self.k = coupling_k
        self.dt = dt
        
        # Assign each task a random starting phase (0 to 2*PI)
        self.phases = np.random.uniform(0, 2 * math.pi, num_tasks)
        
        # Natural frequencies: How much "heat" or priority a task has.
        # High frequency = fast priority, Low frequency = background task
        self.natural_frequencies = np.random.uniform(0.5, 2.0, num_tasks)

    def step_physics(self):
        """Steps the continuous wave physics forward."""
        # Calculate phase differences between all tasks (Matrix operation)
        phase_diffs = self.phases[np.newaxis, :] - self.phases[:, np.newaxis]
        
        # Kuramoto coupling equation: Tasks pull on each other to synchronize
        coupling = np.sum(np.sin(phase_diffs), axis=1) * (self.k / self.num_tasks)
        
        # Update phases using natural frequency + structural coupling
        self.phases += (self.natural_frequencies + coupling) * self.dt
        
        # Keep phases bound to a circle (0 to 2*PI)
        self.phases = np.mod(self.phases, 2 * math.pi)

    def execute_batch(self, threshold_phase=0.15):
        """
        Monitors the continuous wave. When tasks align at the execution phase (0 rad),
        they are grouped into a single Thermodynamic Batch.
        """
        # Find tasks crossing the 0 / 2*PI threshold
        ready_tasks = np.where((self.phases < threshold_phase) | (self.phases > 2 * math.pi - threshold_phase))[0]
        
        if len(ready_tasks) > 0:
            # We "execute" the tasks by resetting their phase and giving them a new frequency
            self.phases[ready_tasks] = math.pi # Push them away from execution threshold
            self.natural_frequencies[ready_tasks] = np.random.uniform(0.5, 2.0, len(ready_tasks))
            return ready_tasks.tolist()
        return []

if __name__ == "__main__":
    print("==================================================")
    print("Initializing MEP Thermodynamic OS Scheduler...")
    print("Simulating 50 background threads as Kuramoto Oscillators.")
    print("==================================================\n")
    
    scheduler = ThermodynamicScheduler(num_tasks=50, coupling_k=2.0, dt=0.05)
    
    total_context_switches = 0
    total_batches = 0
    
    # Simulate 200 CPU clock cycles
    for tick in range(1, 201):
        scheduler.step_physics()
        ready_batch = scheduler.execute_batch()
        
        if len(ready_batch) > 0:
            total_context_switches += len(ready_batch)
            total_batches += 1
            if len(ready_batch) > 1:
                print(f"Clock Tick {tick:03d} | 🌊 RESONANT WAVE: Executing Phase-Locked Batch of {len(ready_batch)} tasks -> {ready_batch}")
            else:
                print(f"Clock Tick {tick:03d} | Single task execution -> {ready_batch}")
                
        time.sleep(0.01) # Slight delay for visual terminal effect
        
    print("\n==================================================")
    print("🏁 SCHEDULER DIAGNOSTICS 🏁")
    print(f"Total tasks processed: {total_context_switches}")
    print(f"Total CPU wake-ups (Batches): {total_batches}")
    print(f"Context-Switch Reduction: {((1.0 - (total_batches/total_context_switches))*100):.1f}% fewer CPU wake-ups compared to rigid queues.")
    print("==================================================")
