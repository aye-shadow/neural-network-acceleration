import kagglehub
import os
import shutil
import glob

# Create data directory if it doesn't exist
# Change data_dir to be one directory level up from the script location
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'data')
os.makedirs(data_dir, exist_ok=True)
print("Path to dataset files (local):", data_dir)

# Download latest version
print("Downloading MNIST dataset from Kaggle...")
path = kagglehub.dataset_download("hojjatk/mnist-dataset")
print("Path to dataset files (downloaded here):", path)

# List all files in the downloaded directory to debug
print("Files found in downloaded directory:")
for file in glob.glob(os.path.join(path, '**'), recursive=True):
    if os.path.isfile(file):
        print(f"- {file}")

# Move required files to data directory
required_files = [
    't10k-images.idx3-ubyte',
    't10k-labels.idx1-ubyte',
    'train-images.idx3-ubyte',
    'train-labels.idx1-ubyte'
]

# Try to find the files in the downloaded directory
for file in required_files:
    # Look for the file pattern across all subdirectories
    matching_files = glob.glob(os.path.join(path, '**', file), recursive=True)
    
    if matching_files:
        src_path = matching_files[0]  # Take the first match
        dst_path = os.path.join(data_dir, file)
        print(f"Copying {src_path} -> {dst_path}")
        
        try:
            # Use copy2 instead of move to avoid cross-filesystem issues
            # For context: move was giving me issues on WSL, but not on Windows
            shutil.copy2(src_path, dst_path)
            print(f"Successfully copied {file}")
        except Exception as e:
            print(f"Error copying {file}: {e}")
    else:
        print(f"Warning: Could not find {file} in the downloaded dataset")

print("Dataset extraction completed")
print("Files in data directory:")
for file in os.listdir(data_dir):
    print(f"- {file}")