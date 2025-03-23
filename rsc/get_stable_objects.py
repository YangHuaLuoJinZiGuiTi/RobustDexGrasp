import os
import shutil
import numpy as np

# Load the .npy file
npy_file_path = "stable_ids/surdf_group27_m.npy"
folder_names = np.load(npy_file_path)
print(f"Loaded {len(folder_names)} folder names from {npy_file_path}")

# Define source and destination base paths
source_base = "surdf_group27_m"
destination_base = "surdf_group27_m_stable"

# Ensure the destination base folder exists
os.makedirs(destination_base, exist_ok=True)

# Iterate over folder names and copy only the specified ones
for folder_name in folder_names:
    source_path = os.path.join(source_base, folder_name)
    destination_path = os.path.join(destination_base, folder_name)

    if os.path.isdir(source_path):  # Check if source folder exists and is a directory
        shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        # print(f"Copied {source_path} to {destination_path}")
    else:
        print(f"Folder not found and skipped: {source_path}")
