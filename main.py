from edge_sim_py import *
import os
import glob
import re
import argparse
import json
from algorithms import QIGA, MOHEFT, RR, RA, OE, OC
from config import *
import pandas as pd

# --- Step 1: Define Your Experiment Parameters ---
NUM_RUNS = 5

# -------------------------------------------

# --- Helper Functions ---
def individual_to_dict(ind):
    return {
        'fitness': ind.fitness,
        'energy': ind.energy,
        'latency': ind.latency,
        'cost': ind.cost,
        'qos': ind.qos,
        'resource_utilization': ind.resource_utilization,
        'missed_deadlines': ind.missed_deadlines,
        'completion_time': ind.max_resource_latency
    }

def save_population(scenario_name, run_id, algorithm_name, best_individuals, data):
    output_dir = f"scheme/outputs/{scenario_name}/run_{run_id}/"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save Metrics
    df = pd.DataFrame([individual_to_dict(ind) for ind in best_individuals])
    df.to_csv(f"{output_dir}{algorithm_name}_best_population.csv", index=False)

    # 2. Save Assignments (For Visualizer)
    best_ind = best_individuals[0]
    resources_and_users = decode(data, best_ind)
    user_assignments = {}
    for server, users in resources_and_users.items():
        for user in users:
            user_assignments[user.id] = server.id     
    with open(f"{output_dir}{algorithm_name}_assignments.json", "w") as f:
        json.dump(user_assignments, f, indent=4)

# --- Main Execution Block ---
if __name__ == "__main__":
    
    # --- NEW: Parse Arguments for Selective Running ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios', nargs='+', help='List of specific scenario names to run')
    args = parser.parse_args()
    # --------------------------------------------------

    simulator = Simulator()

    # Auto-discover scenarios
    dataset_files = glob.glob("datasets/*.json")

    if not dataset_files:
        print("No dataset .json files found in 'datasets/' folder. Stopping.")
        print("Please use the dashboard to generate a dataset first.")
        exit()

    print(f"Found {len(dataset_files)} total datasets.")

    # Regex to parse filenames
    filename_regex = re.compile(r"datasets[/\\](.+)_ES-(\d+)_ED-(\d+)\.json")

    # --- Main Experiment Loop ---
    for file_path in dataset_files:
        match = filename_regex.match(file_path)
        if not match:
            print(f"Skipping file with incorrect format: {file_path}")
            continue

        scenario_name = match.group(1)
        
        # --- NEW: Filter Logic ---
        # If user specified scenarios, skip ones that don't match
        if args.scenarios and scenario_name not in args.scenarios:
            continue
        # -------------------------

        es_count = int(match.group(2))
        user_count = int(match.group(3))
        
        print(f"\n=================================================")
        print(f"STARTING SCENARIO: {scenario_name} ({user_count} Users, {es_count} Servers)")
        print(f"=================================================")
        
        for run_id in range(1, NUM_RUNS + 1):
            print(f"\n--- Starting Run {run_id}/{NUM_RUNS} for {scenario_name} ---")

            try:
                simulator.initialize(input_file=file_path)
            except TypeError:
                print(f"[Error] Could not load file: {file_path}")
                break 

            # Build Graph
            data = {
                'BaseStation': BaseStation, 'EdgeServer': EdgeServer, 'User': User,
                'NetworkSwitch': NetworkSwitch, 'NetworkLink': NetworkLink
            }
            graph = {}
            for link in data['NetworkLink'].all():
                node1_id = link.nodes[0].base_station.id
                node2_id = link.nodes[1].base_station.id
                graph.setdefault(node1_id, []).append((node2_id, link.bandwidth))
                graph.setdefault(node2_id, []).append((node1_id, link.bandwidth))
            data['graph'] = graph

            # Run Algorithms
            print(f'Running QIGA...')
            QIGA_alg = QIGA.QIGA(fitness, K_POP_SIZE, K_GEN_SIZE, data)
            QIGA_pop = QIGA_alg.run()
            save_population(scenario_name, run_id, "QIGA", QIGA_pop, data)

            print(f'Running MOHEFT...')
            MOHEFT_alg = MOHEFT.MOHEFT(fitness, K_POP_SIZE, K_GEN_SIZE, data)
            MOHEFT_pop = MOHEFT_alg.run()
            save_population(scenario_name, run_id, "MOHEFT", MOHEFT_pop, data)

            print(f'Running RR...')
            RR_alg = RR.RR(fitness, K_POP_SIZE, K_GEN_SIZE, data)
            RR_pop = RR_alg.run()
            save_population(scenario_name, run_id, "RR", RR_pop, data)

            print(f'Running RA...')
            RA_alg = RA.RA(fitness, K_POP_SIZE, K_GEN_SIZE, data)
            RA_pop = RA_alg.run()
            save_population(scenario_name, run_id, "RA", RA_pop, data)

            print(f'Running OE...')
            OE_alg = OE.OE(fitness, K_POP_SIZE, K_GEN_SIZE, data)
            OE_pop = OE_alg.run()
            save_population(scenario_name, run_id, "OE", OE_pop, data)

            print(f'Running OC...')
            OC_alg = OC.OC(fitness, data)
            OC_pop = OC_alg.run()
            save_population(scenario_name, run_id, "OC", OC_pop, data)
            
            print(f"--- Completed Run {run_id}/{NUM_RUNS} ---")

    print("\nAll requested simulations completed.")