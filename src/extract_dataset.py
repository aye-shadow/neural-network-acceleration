import kagglehub
import os
import shutil
import glob

# Define paths
root_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')  # Root-level data folder
src_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'data')  # src/data folder

# Create directories if they don't exist
os.makedirs(root_data_dir, exist_ok=True)
os.makedirs(src_data_dir, exist_ok=True)

print("Path to dataset files (root):", root_data_dir)
print("Path to required files (src):", src_data_dir)

# Download latest version into the default directory
print("Downloading MNIST dataset from Kaggle...")
path = kagglehub.dataset_download("hojjatk/mnist-dataset")  # Remove download_path
print("Path to dataset files (downloaded here):", path)

# Move the dataset to the root data directory
if os.path.exists(path):
    for file in os.listdir(path):
        src_file = os.path.join(path, file)
        dst_file = os.path.join(root_data_dir, file)
        print(f"Moving {src_file} -> {dst_file}")
        shutil.move(src_file, dst_file)
    print("Dataset moved to root data directory.")
else:
    print("Error: Downloaded dataset path does not exist.")

# List all files in the downloaded directory to debug
print("Files found in downloaded directory:")
for file in glob.glob(os.path.join(path, '**'), recursive=True):
    if os.path.isfile(file):
        print(f"- {file}")

# Move required files to src/data directory
required_files = [
    't10k-images.idx3-ubyte',
    't10k-labels.idx1-ubyte',
    'train-images.idx3-ubyte',
    'train-labels.idx1-ubyte'
]

# Try to find the files in the downloaded directory
for file in required_files:
    # Look for the file pattern inside the root data directory
    matching_files = glob.glob(os.path.join(root_data_dir, '**', file), recursive=True)
    
    if matching_files:
        src_path = matching_files[0]  # Take the first match
        dst_path = os.path.join(src_data_dir, file)
        print(f"Moving {src_path} -> {dst_path}")
        
        try:
            # Use move to transfer the files to src/data
            shutil.move(src_path, dst_path)
            print(f"Successfully moved {file}")
        except Exception as e:
            print(f"Error moving {file}: {e}")
    else:
        print(f"Warning: Could not find {file} in the downloaded dataset")

print("Dataset extraction completed")
print("Files in src/data directory:")
for file in os.listdir(src_data_dir):
    print(f"- {file}")