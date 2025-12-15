from edge_sim_py import *
import os
import glob
import re
import argparse
import json
from algorithms import QIGA, MOHEFT, RR, RA, OE, OC, GA, PSO, DE, HybridQIGA
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
    resources_map = decode(data, best_ind)
    user_assignments = {}
    
    for server, task_list in resources_map.items():
        for task_item in task_list:
            if isinstance(task_item, dict) and 'user' in task_item:
                user = task_item['user']
            else:
                user = task_item
            user_assignments[user.id] = server.id     
            
    with open(f"{output_dir}{algorithm_name}_assignments.json", "w") as f:
        json.dump(user_assignments, f, indent=4)

# --- Main Execution Block ---
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios', nargs='+', help='List of specific scenario names to run')
    args = parser.parse_args()

    simulator = Simulator()

    dataset_files = glob.glob("datasets/*.json")

    if not dataset_files:
        print("No dataset .json files found in 'datasets/' folder. Stopping.")
        print("Please use the dashboard to generate a dataset first.")
        exit()

    print(f"Found {len(dataset_files)} total datasets.")

    filename_regex = re.compile(r"datasets[/\\](.+)_ES-(\d+)_ED-(\d+)\.json")

    for file_path in dataset_files:
        match = filename_regex.search(file_path)
        if not match:
            filename = os.path.basename(file_path)
            match = re.match(r"(.+)_ES-(\d+)_ED-(\d+)\.json", filename)
            
        if not match:
            print(f"Skipping file with incorrect format: {file_path}")
            continue

        scenario_name = match.group(1)
        
        if args.scenarios and scenario_name not in args.scenarios:
            continue

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
            best_qiga, QIGA_pop = QIGA_alg.run()
            # Insert best at start for save_population
            if best_qiga not in QIGA_pop: QIGA_pop.insert(0, best_qiga)
            elif QIGA_pop[0] != best_qiga:
                QIGA_pop.remove(best_qiga)
                QIGA_pop.insert(0, best_qiga)
            save_population(scenario_name, run_id, "QIGA", QIGA_pop, data)

            print(f'Running HybridQIGA...')
            HybridQIGA_alg = HybridQIGA.HybridQIGA(fitness, K_POP_SIZE, K_GEN_SIZE, data)
            best_hqiga, HybridQIGA_pop = HybridQIGA_alg.run()
            if best_hqiga not in HybridQIGA_pop: HybridQIGA_pop.insert(0, best_hqiga)
            elif HybridQIGA_pop[0] != best_hqiga:
                HybridQIGA_pop.remove(best_hqiga)
                HybridQIGA_pop.insert(0, best_hqiga)
            save_population(scenario_name, run_id, "HybridQIGA", HybridQIGA_pop, data)

            print(f'Running MOHEFT...')
            MOHEFT_alg = MOHEFT.MOHEFT(fitness, K_POP_SIZE, K_GEN_SIZE, data)
            best_moheft, MOHEFT_pop = MOHEFT_alg.run()
            if best_moheft and best_moheft not in MOHEFT_pop: MOHEFT_pop.insert(0, best_moheft)
            save_population(scenario_name, run_id, "MOHEFT", MOHEFT_pop, data)

            print(f'Running GA...')
            GA_alg = GA.GA(fitness, K_POP_SIZE, K_GEN_SIZE, data)
            best_ga, GA_pop = GA_alg.run()
            if best_ga and best_ga not in GA_pop: GA_pop.insert(0, best_ga)
            save_population(scenario_name, run_id, "GA", GA_pop, data)

            print(f'Running PSO...')
            PSO_alg = PSO.PSO(fitness, K_POP_SIZE, K_GEN_SIZE, data)
            best_pso, PSO_pop = PSO_alg.run()
            if best_pso not in PSO_pop: PSO_pop.insert(0, best_pso)
            elif PSO_pop[0] != best_pso:
                PSO_pop.remove(best_pso)
                PSO_pop.insert(0, best_pso)
            save_population(scenario_name, run_id, "PSO", PSO_pop, data)

            print(f'Running DE...')
            DE_alg = DE.DE(fitness, K_POP_SIZE, K_GEN_SIZE, data)
            best_de, DE_pop = DE_alg.run()
            if best_de not in DE_pop: DE_pop.insert(0, best_de)
            elif DE_pop[0] != best_de:
                DE_pop.remove(best_de)
                DE_pop.insert(0, best_de)
            save_population(scenario_name, run_id, "DE", DE_pop, data)

            print(f'Running RR...')
            RR_alg = RR.RR(fitness, K_POP_SIZE, K_GEN_SIZE, data)
            best_rr, RR_pop = RR_alg.run()
            save_population(scenario_name, run_id, "RR", RR_pop, data)

            print(f'Running RA...')
            RA_alg = RA.RA(fitness, K_POP_SIZE, K_GEN_SIZE, data)
            best_ra, RA_pop = RA_alg.run()
            save_population(scenario_name, run_id, "RA", RA_pop, data)

            print(f'Running OE...')
            OE_alg = OE.OE(fitness, K_POP_SIZE, K_GEN_SIZE, data)
            best_oe, OE_pop = OE_alg.run()
            if best_oe not in OE_pop: OE_pop.insert(0, best_oe)
            elif OE_pop[0] != best_oe:
                OE_pop.remove(best_oe)
                OE_pop.insert(0, best_oe)
            save_population(scenario_name, run_id, "OE", OE_pop, data)

            print(f'Running OC...')
            OC_alg = OC.OC(fitness, data)
            best_oc, OC_pop = OC_alg.run()
            save_population(scenario_name, run_id, "OC", OC_pop, data)
            
            print(f"--- Completed Run {run_id}/{NUM_RUNS} ---")

    print("\nAll requested simulations completed.")