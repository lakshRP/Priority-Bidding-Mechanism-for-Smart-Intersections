#!/usr/bin/env python3
import os
import time
import csv
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

from helper import Simulation, Intersection

# reuse fixed-time schedule
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

# reuse no-animation runner
def run_no_anim(sim):
    total = sim.steps
    interval = max(total // 10, 1)
    for frame in range(total):
        if frame % interval == 0:
            print(f"[{sim.size}×{sim.size} | N={sim.n_vehicles}] step {frame}/{total}")
        sim.step(frame)
    return sim.time_history, sim.cost_history


def sensitivity_grid_size(grid_sizes, nveh, duration, trials, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    for gs in grid_sizes:
        imps = []
        for seed in range(trials):
            random.seed(seed)
            np.random.seed(seed)
            # PBM
            sim_p = Simulation(size=gs, n_vehicles=nveh, duration=duration)
            t, q_p = run_no_anim(sim_p)
            # Fixed-Time
            sim_f = Simulation(size=gs, n_vehicles=nveh, duration=duration)
            for inter in sim_f.net.intersections.values():
                inter.compute_schedule = fixed_compute_schedule.__get__(inter, Intersection)
            _, q_f = run_no_anim(sim_f)
            valid = q_f > 0
            imp = ((q_f[valid] - q_p[valid]) / q_f[valid] * 100).mean()
            imps.append(imp)
        mean_imp, std_imp = np.mean(imps), np.std(imps)
        print(f"Grid={gs} → {mean_imp:.1f}%±{std_imp:.1f}%")
        summary.append((gs, mean_imp, std_imp))

    # write CSV
    csv_path = os.path.join(out_dir, "sensitivity_grid_size.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["grid_size", "mean_improvement_pct", "std_improvement_pct"])
        writer.writerows(summary)

    # plot
    gs_vals, means, stds = zip(*summary)
    plt.errorbar(gs_vals, means, yerr=stds, fmt="o-", capsize=5)
    plt.xlabel("Grid Size")
    plt.ylabel("Avg Queue-Reduction (%)")
    plt.title(f"Sensitivity to Grid Size (N={nveh}, dur={duration}s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sens_grid_size.png"))
    plt.close()


def sensitivity_vehicle_count(grid_size, veh_counts, duration, trials, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    for nv in veh_counts:
        imps = []
        for seed in range(trials):
            random.seed(seed)
            np.random.seed(seed)
            # PBM
            sim_p = Simulation(size=grid_size, n_vehicles=nv, duration=duration)
            _, q_p = run_no_anim(sim_p)
            # Fixed-Time
            sim_f = Simulation(size=grid_size, n_vehicles=nv, duration=duration)
            for inter in sim_f.net.intersections.values():
                inter.compute_schedule = fixed_compute_schedule.__get__(inter, Intersection)
            _, q_f = run_no_anim(sim_f)
            valid = q_f > 0
            imp = ((q_f[valid] - q_p[valid]) / q_f[valid] * 100).mean()
            imps.append(imp)
        mean_imp, std_imp = np.mean(imps), np.std(imps)
        print(f"Nveh={nv} → {mean_imp:.1f}%±{std_imp:.1f}%")
        summary.append((nv, mean_imp, std_imp))

    # write CSV
    csv_path = os.path.join(out_dir, "sensitivity_vehicle_count.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_vehicles", "mean_improvement_pct", "std_improvement_pct"])
        writer.writerows(summary)

    # plot
    ns, means, stds = zip(*summary)
    plt.errorbar(ns, means, yerr=stds, fmt="s-", capsize=5)
    plt.xlabel("Number of Vehicles")
    plt.ylabel("Avg Queue-Reduction (%)")
    plt.title(f"Sensitivity to Vehicle Count (Grid={grid_size}×{grid_size}, dur={duration}s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sens_vehicle_count.png"))
    plt.close()


def sensitivity_cycle_count(grid_size, nveh, cycles_list, duration, trials, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    for cyc in cycles_list:
        times = []
        for seed in range(trials):
            random.seed(seed)
            np.random.seed(seed)
            sim = Simulation(size=grid_size, n_vehicles=nveh, duration=duration)
            sim.step(0)
            start = time.perf_counter()
            for frame in range(1, cyc + 1):
                sim.step(frame)
            end = time.perf_counter()
            times.append((end - start) / cyc)
        mean_t, std_t = np.mean(times), np.std(times)
        print(f"Cycles={cyc} → {mean_t*1e3:.2f}±{std_t*1e3:.2f} ms/step")
        summary.append((cyc, mean_t, std_t))

    # write CSV
    csv_path = os.path.join(out_dir, "sensitivity_cycle_count.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cycles", "mean_step_time_s", "std_step_time_s"])
        writer.writerows(summary)

    # plot
    cs, means, stds = zip(*summary)
    plt.errorbar(cs, means, yerr=stds, fmt="^-", capsize=5)
    plt.xlabel("Number of Cycles")
    plt.ylabel("Avg Step Time (s)")
    plt.title(f"Sensitivity to Cycle Count (Grid={grid_size}×{grid_size}, N={nveh}, dur={duration}s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sens_cycle_count.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run extended sensitivity analyses on PBM vs. Fixed-Time"
    )
    parser.add_argument(
        "--out-dir", default="results/analytics", help="where to store analytics outputs"
    )
    sub = parser.add_subparsers(dest="test", required=True)

    g1 = sub.add_parser("grid_size")
    g1.add_argument("--sizes", nargs="+", type=int, default=[2, 4, 6, 8])
    g1.add_argument("--nveh", type=int, default=20)
    g1.add_argument("--duration", type=float, default=200.0)
    g1.add_argument("--trials", type=int, default=3)

    g2 = sub.add_parser("vehicle_count")
    g2.add_argument("--grid", type=int, default=4)
    g2.add_argument("--counts", nargs="+", type=int, default=[10, 20, 50, 100, 200])
    g2.add_argument("--duration", type=float, default=200.0)
    g2.add_argument("--trials", type=int, default=3)

    g3 = sub.add_parser("cycle_count")
    g3.add_argument("--grid", type=int, default=4)
    g3.add_argument("--nveh", type=int, default=20)
    g3.add_argument("--cycles", nargs="+", type=int, default=[50, 100, 200, 500])
    g3.add_argument("--duration", type=float, default=50.0)
    g3.add_argument("--trials", type=int, default=3)

    args = parser.parse_args()
    if args.test == "grid_size":
        sensitivity_grid_size(args.sizes, args.nveh, args.duration, args.trials, args.out_dir)
    elif args.test == "vehicle_count":
        sensitivity_vehicle_count(args.grid, args.counts, args.duration, args.trials, args.out_dir)
    elif args.test == "cycle_count":
        sensitivity_cycle_count(args.grid, args.nveh, args.cycles, args.duration, args.trials, args.out_dir)
