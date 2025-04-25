import json
import re
from circuitGeneratorPool import generate
from circuitSimulatorMPS import parse, trotter
import multiprocessing
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
multiprocessing.set_start_method("spawn", force=True)

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
        parse(
            path,
            isJSON=True,
            draw=False,
            saveAndPlot=False,
            optimize=False
        )

        q_val, T_val = getqT(path)
        trotter(100, 10, float(T_val), int(q_val), compare=False,save=True)

        print(f"[SIMULATION {key}] Finished.\n")

    print("All tasks completed.")

if __name__ == "__main__":
    main()