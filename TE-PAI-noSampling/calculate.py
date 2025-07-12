print("calculate.py loaded")
import warnings
import os

# match on the exact warning text:
warnings.filterwarnings(
    "ignore",
    message=r"Couldn't import `kahypar` - skipping from default hyper optimizer and using basic `labels` method instead\."
)

# (or, match by module name)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"cotengra\.hyperoptimizers\.hyper"
)

import json
import re
from circuitGeneratorPool import generate
from circuitSimulatorMPS import parse, trotter, showComplexity, trotterThenTEPAI
import multiprocessing
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
print("Current working directory:", os.getcwd())

def getqT(path):
    pattern = re.compile(r"-q-(?P<q>[^-]+).*?-T-(?P<T>[^-]+)$")
    m = pattern.search(path)
    if m:
        q_val = m.group("q")
        T_val = m.group("T")
        return q_val, T_val
    else:
        raise ValueError("Couldn't find q and T in path")

def main():
    print("Starting TE-PAI simulation...\n")
    # Load configuration from JSON file with correct encoding
    with open('TE-PAI-noSampling/calculateConfig.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Process "generate" configurations
    print("Starting generation phase...\n")
    for key, params in config["generate"].items():
        print(f"[GENERATION {key}] Starting with parameters:")
        print(json.dumps(params, indent=4))

        # Convert dict values to array (ensure the order matches generate() expectations)
        param_array = list(params.values())

        generate(param_array)

        print(f"[GENERATION {key}] Finished.\n")

    # Process "simulate" configurations
    print("Starting simulation phase...\n")
    for key, path in config["simulate"].items():
        print(f"[SIMULATION {key}] Simulating for path: {path}")
        costs = parse(
            path,
            isJSON=True,
            draw=False,
            saveAndPlot=False,
            optimize=False,
            flip=True
        )
        showComplexity(costs, 1, len(costs), path)

        q_val, T_val = getqT(path)
        trotter(100, 10, float(T_val), int(q_val), compare=False,save=True)

        print(f"[SIMULATION {key}] Finished.\n")

     # Process "simulate" configurations
    
    print("Starting Lie phase...\n")
    for key, params in config["lie"].items():
        print(f"[LIE {key}] generating for path: {params}")
        q = params["q"]
        T = params["T"]
        N = params["N"]
        trotter(N=N,
        n_snapshot=10, 
        T=T, 
        q=q, 
        compare=False, 
        save=True, 
        draw=False, 
        flip=True)

    print("Starting trotterThenTEPAI phase...\n")
    for key, params in config["trotterThenTEPAI"].items():
        print(f"[TROTTER THEN TEPAI {key}] Starting with parameters:")
        print(json.dumps(params, indent=4))
        path = params["path"]
        trotterN = params["trotterN"]
        trottern = params["trottern"]
        trotterT = params["trotterT"]
        flip = params["flip"]
        confirm = params["confirm"]

        # Convert dict values to array (ensure the order matches generate() expectations)
        param_array = list(params.values())

        trotterThenTEPAI(trotterN=trotterN,
            trottern=trottern,
            trotterT=trotterT,
            folder=path,
            flip=flip,
            confirm=confirm
        )

        print(f"[TROTTER THEN TEPAI {key}] Finished.\n")

    print("All tasks completed.")

if __name__ == "__main__":
    print("Starting main process...")
    multiprocessing.set_start_method("spawn", force=True)
    main()