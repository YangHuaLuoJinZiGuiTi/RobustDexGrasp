import os
import numpy as np

base_path = 'large_scale_light'
output_file = 'large_scale_light.npy'

# Ensure the base path exists
if not os.path.exists(base_path):
    raise FileNotFoundError(f"The base path '{base_path}' does not exist.")

# List all entries in the base path
ids = []
for entry in os.listdir(base_path):
    full_path = os.path.join(base_path, entry)
    if os.path.isdir(full_path):
        ids.append(entry)
        print(f"Found ID: {entry}")

# Save the list of IDs into a .npy file
np.save(output_file, np.array(ids))
print(f"Extracted {len(ids)} IDs and saved to '{output_file}'")

