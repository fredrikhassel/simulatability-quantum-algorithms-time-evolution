import json
from circuitGeneratorPool import generate
from circuitSimulatorMPS import parse
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
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

        print(f"[SIMULATION {key}] Finished.\n")

    print("All tasks completed.")

if __name__ == "__main__":
    main()