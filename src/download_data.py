import kagglehub
import shutil
import os

print("Downloading dataset from Kaggle...")
cache_path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")
print(f"Downloaded to cache: {cache_path}")

local_data_dir = "./data"

# Clear existing data folder if it exists, then copy the new data
if os.path.exists(local_data_dir):
    shutil.rmtree(local_data_dir)
    
shutil.copytree(cache_path, local_data_dir)
print(f"Dataset successfully moved to {local_data_dir} for DVC tracking.")