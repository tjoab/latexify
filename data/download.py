import os
import subprocess


#TODO: Remove the tar file
#TODO: Clean up directories
#TODO: Ensure works cross platform (dockerize?)

# full dataset: https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz
# excerpt dataset: https://storage.googleapis.com/mathwriting_data/mathwriting-2024-excerpt.tgz

def download_and_extract_dataset(dataset_url, dest_dir):
    
    os.makedirs(dest_dir, exist_ok=True)
    
    dataset_filename = os.path.basename(dataset_url)
    dest_path = os.path.join(dest_dir, dataset_filename)


    if not os.path.exists(dest_path):
        subprocess.run(["curl", "-L", dataset_url, "-o", dest_path], check=True)

    subprocess.run(["tar", "zxf", dest_path, "-C", dest_dir], check=True)

    data_dir = dest_path.strip('.tgz')
    return data_dir


#print(download_and_extract_dataset('https://storage.googleapis.com/mathwriting_data/mathwriting-2024-excerpt.tgz', './data'))

