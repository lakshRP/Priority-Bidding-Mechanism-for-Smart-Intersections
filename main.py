import time
import sys
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from test3 import Simulation, Intersection

# Ensure results directory exists
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------
# Fixed-Time Baseline: equal green per link
# ---------------------------------------------
def fixed_compute_schedule(self, t):
    entries = [(link, veh) for link in self.incoming_links for veh in link.queue]
    if not entries:
        self.schedule = []
        return
    dur = self.C / len(self.incoming_links)
    self.schedule = []
    start = 0.0
    for link in self.incoming_links:
        self.schedule.append((link, start, start + dur))
        start += dur

# ---------------------------------------------
# Helper: run simulation without verbose output
# ---------------------------------------------
def run_no_anim(sim):
    # show progress at 10% increments
    total = sim.steps
    interval = max(total // 10, 1)
    for frame in range(total):
        if frame % interval == 0:
            print(f"Running frame {frame}/{total} ({(frame/total)*100:.0f}%)")
        sim.step(frame)
    print(f"Completed {total}/{total} frames (100%)")
    return sim.time_history, sim.cost_history

# ---------------------------------------------
# Experiment 1: PBM vs Fixed-Time
# ---------------------------------------------
def experiment_pbm_vs_fixed(grid_size=4, nveh=20, duration=200.0, trials=3):
    print("\n=== Experiment 1: PBM vs Fixed-Time ===")
    all_q_pbm = []
    all_q_fix = []
    for trial in range(trials):
        print(f" Trial {trial+1}/{trials}...")
        random.seed(trial)
        np.random.seed(trial)
        # PBM run
        sim_pbm = Simulation(size=grid_size, n_vehicles=nveh, duration=duration)
        t_pbm, q_pbm = run_no_anim(sim_pbm)
        # Fixed-Time run
        sim_fix = Simulation(size=grid_size, n_vehicles=nveh, duration=duration)
        for inter in sim_fix.net.intersections.values():
            inter.compute_schedule = fixed_compute_schedule.__get__(inter, Intersection)
        t_fix, q_fix = run_no_anim(sim_fix)
        all_q_pbm.append(q_pbm)
        all_q_fix.append(q_fix)
    # Compute averages
    all_q_pbm = np.array(all_q_pbm)
    all_q_fix = np.array(all_q_fix)
    mean_q_pbm = all_q_pbm.mean(axis=0)
    mean_q_fix = all_q_fix.mean(axis=0)
    # Compute improvement
    valid = mean_q_fix > 0
    improvement = ((mean_q_fix[valid] - mean_q_pbm[valid]) / mean_q_fix[valid]) * 100
    mean_imp = improvement.mean() if len(improvement) > 0 else 0
    print(f"Average queue-length improvement: {mean_imp:.1f}% over {trials} trials.")
    # Save data
    np.savetxt(os.path.join(RESULTS_DIR, 'exp1_mean_q_pbm.csv'), mean_q_pbm, delimiter=',')
    np.savetxt(os.path.join(RESULTS_DIR, 'exp1_mean_q_fix.csv'), mean_q_fix, delimiter=',')
    # Plot averaged curves
    plt.figure(figsize=(8,5))
    plt.plot(t_pbm, mean_q_pbm, label='PBM (avg)', linewidth=2)
    plt.plot(t_fix, mean_q_fix, label='Fixed-Time (avg)', linestyle='--', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Total Queue Length')
    plt.title(f'Average PBM vs Fixed-Time (N={nveh}, Grid={grid_size}×{grid_size})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp1_queue_length.png'))
    plt.show()

# ---------------------------------------------
# Experiment 2: Runtime Scaling
# ---------------------------------------------
def experiment_runtime_scaling(grid_size=4, duration=50.0, cycles=100, trials=3):
    print("\n=== Experiment 2: Runtime Scaling ===")
    vehicle_counts = [10, 20, 50, 100, 200]
    avg_times = []
    std_times = []
    for n in vehicle_counts:
        times = []
        for trial in range(trials):
            random.seed(trial)
            sim = Simulation(size=grid_size, n_vehicles=n, duration=duration)
            sim.step(0)
            start = time.perf_counter()
            for frame in range(1, cycles + 1):
                sim.step(frame)
            end = time.perf_counter()
            times.append((end - start) / cycles)
        mean_t = np.mean(times)
        std_t = np.std(times)
        avg_times.append(mean_t)
        std_times.append(std_t)
        print(f"Vehicles={n:3d}: {mean_t*1000:.2f}±{std_t*1000:.2f} ms")
    # Linear fit
    coef = np.polyfit(vehicle_counts, avg_times, 1)
    fit_line = np.poly1d(coef)
    print(f"Runtime ~ {coef[0]:.2e}*N + {coef[1]:.2e}")
    # Save data
    np.savetxt(os.path.join(RESULTS_DIR, 'exp2_vehicle_counts.csv'), vehicle_counts, delimiter=',', fmt='%d')
    np.savetxt(os.path.join(RESULTS_DIR, 'exp2_avg_times.csv'), avg_times, delimiter=',')
    np.savetxt(os.path.join(RESULTS_DIR, 'exp2_std_times.csv'), std_times, delimiter=',')
    # Plot with error bars and fit
    plt.figure(figsize=(8,5))
    plt.errorbar(vehicle_counts, avg_times, yerr=std_times, fmt='o-', linewidth=2, capsize=5)
    plt.plot(vehicle_counts, fit_line(vehicle_counts), linestyle='--', label='Linear Fit')
    plt.xlabel('Number of Vehicles')
    plt.ylabel('Avg Step Time (s)')
    plt.title(f'Runtime Scaling (Grid={grid_size}×{grid_size})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp2_runtime_scaling.png'))
    plt.show()

# ---------------------------------------------
# Main entry point
# ---------------------------------------------
def main():
    experiment_pbm_vs_fixed()
    experiment_runtime_scaling()

if __name__ == '__main__':
    main()