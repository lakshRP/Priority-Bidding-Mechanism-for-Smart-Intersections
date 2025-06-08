#!/usr/bin/env python3
import json, os, random
import numpy as np
from helper import Simulation, Intersection
from main import run_no_anim, fixed_compute_schedule

# where scenarios.json lives
SCEN_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'scenarios.json')

def load_scenarios(path=SCEN_PATH):
    with open(path) as f:
        return json.load(f)

def run_scenario(s):
    random.seed(s["random_seed"])
    np.random.seed(s["random_seed"])
    sim = Simulation(size=s["grid_size"], n_vehicles=s["n_vehicles"], duration=s["duration"])
    # PBM
    t_pbm, q_pbm = run_no_anim(sim)
    # Fixed-time
    sim2 = Simulation(size=s["grid_size"], n_vehicles=s["n_vehicles"], duration=s["duration"])
    for inter in sim2.net.intersections.values():
        inter.compute_schedule = fixed_compute_schedule.__get__(inter, Intersection)
    t_fix, q_fix = run_no_anim(sim2)

    # save per-scenario outputs
    out_dir = os.path.join('results', s["name"])
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, 'q_pbm.csv'), q_pbm, delimiter=',')
    np.savetxt(os.path.join(out_dir, 'q_fix.csv'), q_fix, delimiter=',')
    np.savetxt(os.path.join(out_dir, 't.csv'), t_pbm, delimiter=',')
    print(f"Completed scenario '{s['name']}' ➔ results/{s['name']}")

if __name__ == "__main__":
    scenarios = load_scenarios()
    for scen in scenarios:
        run_scenario(scen)
