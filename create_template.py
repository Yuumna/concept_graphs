import os
import json
import numpy as np

# Base directory containing the CLEVR JSON files
base_dir = "input/single-body_2d_3classes/train/"  # Change this to your directory

# Load the properties file
with open("properties_single-body_2d_3classes.json", "r") as f:
    props = json.load(f)

# Config dictionary for train and test splits
configs = {
    "H32-train1": {
        "train": ["000", "001", "100", "010"],
        "test":  ["011", "110", "101", "111"]
    }
}

# Combine all configurations into one list
all_configs = configs["H32-train1"]["train"] + configs["H32-train1"]["test"]

# Dictionary to store results for each config
results_by_config = {}

# Loop over each config
for config in all_configs:
    # Extract the target color using the second digit of the config
    target_color = np.array(props["colors"][config[1]])
    
    # List to hold tuples of (filename, distance)
    config_files = []
    
    # Loop through each file in the base directory
    for file in os.listdir(base_dir):
        if file.startswith("CLEVR_") and file.endswith(".json"):
            # Extract the config part from the filename
            # Assuming filename format: CLEVR_<config>_<id>.json
            parts = file.split("_")
            if len(parts) < 2:
                continue  # Skip any file that doesn't match the expected pattern
            
            file_config = parts[1]
            if file_config == config:
                file_path = os.path.join(base_dir, file)
                with open(file_path, "r") as jf:
                    data = json.load(jf)
                    # data[1] is assumed to be the RGBA list; take the first 3 values as RGB
                    file_color = np.array(data[1][:3])
                    
                    # Compute Euclidean distance between the file's RGB and target RGB
                    distance = np.linalg.norm(file_color - target_color)
                    config_files.append((file, distance))
    
    # Sort files for this config based on the computed distance (smallest first)
    config_files.sort(key=lambda x: x[1])
    # pick first 10
    results_by_config[config] = config_files[:10]

# Display the results for each config 
for config, files in results_by_config.items():
    print(f"Config {config} (target color: {np.array(props['colors'][config[1]])}):")
    for file, distance in files:
        print(f"  {file} - Distance: {distance:.4f}")
    print()  # Empty line for better separation between configs

# create a dir called template with the closest files json and png
os.makedirs("template", exist_ok=True)
for config, files in results_by_config.items():
    for file, _ in files:
        # Copy the JSON file to the template directory
        json_file_path = os.path.join(base_dir, file)
        png_file_path = json_file_path.replace(".json", ".png")
        
        # Copy JSON
        os.system(f"cp {json_file_path} template/{file}")
        
        # Copy PNG if it exists
        if os.path.exists(png_file_path):
            os.system(f"cp {png_file_path} template/{file.replace('.json', '.png')}")
