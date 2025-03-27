import kagglehub
import os
import shutil

# Create data directory if it doesn't exist
# Change data_dir to be one directory level up from the script location
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(data_dir, exist_ok=True)
print("Path to dataset files (local):", data_dir)

# Download latest version
print("Downloading MNIST dataset from Kaggle...")
path = kagglehub.dataset_download("hojjatk/mnist-dataset")
print("Path to dataset files (downloaded here):", path)

# Move required files to data directory
required_files = [
    't10k-images.idx3-ubyte',
    't10k-labels.idx1-ubyte',
    'train-images.idx3-ubyte',
    'train-labels.idx1-ubyte'
]

for file in required_files:
    src_path = os.path.join(path, file)
    dst_path = os.path.join(data_dir, file)
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)

print("Dataset extracted successfully to:", data_dir)
print("Files in data directory:")
for file in os.listdir(data_dir):
    print(f"- {file}")