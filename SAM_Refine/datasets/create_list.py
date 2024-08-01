import os
import glob

# Specify the folder containing the .npz files
folder_path = '../SAM_FLARE'

# Get a list of all .npz files in the folder
npz_files = glob.glob(os.path.join(folder_path, '*.npz'))

# Create a list to store the modified filenames (without suffix)
modified_filenames = []

# Iterate through the .npz files and remove the suffix
for npz_file in npz_files:
    file_name = os.path.basename(npz_file)
    name_without_suffix = os.path.splitext(file_name)[0]
    modified_filenames.append(name_without_suffix)

# Define the path for the output text file
output_file = '../lists/lists_Synapse/train.txt'

# Write the modified filenames to the text file
with open(output_file, 'w') as f:
    f.write('\n'.join(modified_filenames))

print(f"Modified filenames saved to {output_file}")
