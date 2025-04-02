import json

# Step 1: Define some data
data = {
    "name": "Fredrik",
    "project": "Quantum Algorithm Simulatability",
    "success": True,
    "parameters": {
        "timesteps": 100,
        "epsilon": 0.01
    }
}

# Step 2: Write to a JSON file
with open("data.json", "w") as f:
    json.dump(data, f, indent=4)

print("✅ JSON file written as 'data.json'")

# Step 3: Read it back in
with open("data.json", "r") as f:
    loaded_data = json.load(f)

print("📂 Loaded data from file:")
print(loaded_data)
