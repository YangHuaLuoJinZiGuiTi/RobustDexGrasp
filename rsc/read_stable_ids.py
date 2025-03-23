import os
import numpy as np

# Define paths
source_folder = "stable_states"
destination_folder = "stable_ids"

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

grouped_ids = {}

# Loop through all group folders in the source folder
for group_name in os.listdir(source_folder):
    group_path = os.path.join(source_folder, group_name)

    # Ensure the current item is a directory
    if os.path.isdir(group_path):
        ids = []  # List to store IDs

        # Loop through all .npy files in the group folder
        for file_name in os.listdir(group_path):
            if file_name.endswith(".npy"):
                # Extract the ID from the file name
                id_part = os.path.splitext(file_name)[0]  # Remove the .npy extension
                ids.append(id_part)

        # Save the IDs as a .npy file in the destination folder
        np.save(os.path.join(destination_folder, f"{group_name}.npy"), np.array(ids))
        grouped_ids[group_name] = ids

for i in range(1, 36):
    group_name = f'surdf_group{i}_s'
    print(f"length of {group_name}: {len(grouped_ids[group_name])}")
    group_name = f'surdf_group{i}_m'
    print(f"length of {group_name}: {len(grouped_ids[group_name])}")
    group_name = f'surdf_group{i}_l'
    print(f"length of {group_name}: {len(grouped_ids[group_name])}")

total_obj_ids = sum(len(ids) for ids in grouped_ids.values())
print(f'Total obj_ids found: {total_obj_ids}')
print("ID extraction and saving completed.")
