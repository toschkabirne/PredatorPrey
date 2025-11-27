import itertools
import json
import subprocess
import multiprocessing
import sys
import os
import time

# Configuration
PYTHON_EXECUTABLE = "/Users/toschka/Python/venv/bin/python"
HEADLESS_RUNNER = "headless_runner.py"
OUTPUT_FILE = "search_results.json"
BEST_PARAMS_FILE = "best_parameters.json"

# Parameter Grid
# "Mainly focus on these parameters... finer grid"
# ADD_NEURON = 0.15
# ADD_WEIGHT = 0.6
# CHANGE_WEIGHT = 0.8

param_grid = {
    "ADD_NEURON": [0.05, 0.1, 0.15, 0.2, 0.25],
    "ADD_WEIGHT": [0.4, 0.5, 0.6, 0.7, 0.8],
    "CHANGE_WEIGHT": [0.6, 0.7, 0.8, 0.9],
    # Sparse grid for others
    "PRED_INIT_MUT": [40], # Default
    "PREY_INIT_MUT": [12], # Default
}

def generate_combinations(grid):
    keys = grid.keys()
    values = grid.values()
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))

def run_single_simulation(params):
    cmd = [
        PYTHON_EXECUTABLE,
        HEADLESS_RUNNER,
        "--params", json.dumps(params),
        "--max_steps", "2000" # Limit to prevent infinite loops, but high enough to test stability
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = json.loads(result.stdout.strip())
        return output
    except subprocess.CalledProcessError as e:
        return {"error": str(e), "stderr": e.stderr, "parameters": params}
    except json.JSONDecodeError as e:
        return {"error": "JSON decode error", "stdout": result.stdout, "parameters": params}

def main():
    combinations = list(generate_combinations(param_grid))
    print(f"Total combinations to test: {len(combinations)}")
    
    results = []
    start_time = time.time()
    
    # Use fewer processes than CPU count to avoid overloading if simulations are heavy
    # But these are lightweight (headless), so CPU count is probably fine.
    pool_size = max(1, multiprocessing.cpu_count() - 1)
    print(f"Running with {pool_size} processes...")
    
    with multiprocessing.Pool(pool_size) as pool:
        # Map returns results in order
        for i, result in enumerate(pool.imap_unordered(run_single_simulation, combinations)):
            results.append(result)
            if i % 10 == 0:
                print(f"Processed {i}/{len(combinations)} simulations...")
    
    end_time = time.time()
    print(f"Search completed in {end_time - start_time:.2f} seconds.")
    
    # Sort results
    # 1. Steps (Higher is better)
    # 2. Survived (True > False) - though if steps are equal (max_steps), survived doesn't matter much unless we want to know if it crashed vs finished.
    # Actually, if steps == max_steps, it survived.
    # If steps < max_steps, it died (survived=False).
    # So sorting by steps descending is sufficient.
    
    valid_results = [r for r in results if "error" not in r]
    error_results = [r for r in results if "error" in r]
    
    print(f"Valid results: {len(valid_results)}")
    print(f"Errors: {len(error_results)}")
    
    if error_results:
        print("Sample error:", error_results[0])
    
    valid_results.sort(key=lambda x: x.get("steps", 0), reverse=True)
    
    # Save all results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
        
    if valid_results:
        best = valid_results[0]
        print("\nBest Result:")
        print(json.dumps(best, indent=2))
        
        with open(BEST_PARAMS_FILE, "w") as f:
            json.dump(best["parameters"], f, indent=2)
            
        print(f"\nBest parameters saved to {BEST_PARAMS_FILE}")
    else:
        print("\nNo valid results found.")

if __name__ == "__main__":
    main()
