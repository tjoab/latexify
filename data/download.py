import os
import subprocess

#TODO: Ensure works cross operating systems (dockerize)
def download_and_extract_dataset(dataset_url: str, dest_dir: str) -> str:
    """
    Downloads and extracts a .tgz dataset from the given URL into the specified directory.

    Parameters
    ----------
    dataset_url : str
        URL to the .tgz file.
    dest_dir : str
        Directory to download and extract the dataset into.

    Returns
    -------
    str
        path to the new data directory
    """
    
    # Ensure parent directory exists
    os.makedirs(dest_dir, exist_ok=True)

    tgz_name = os.path.basename(dataset_url)
    tgz_path = os.path.join(dest_dir, tgz_name)

    # This is the path to the extracted folder
    data_dir = tgz_path.removesuffix(".tgz")

    # Only downloads and extracts if we havent already done so
    if not os.path.exists(data_dir):
        subprocess.run(["curl", "-L", dataset_url, "-o", tgz_path], check=True)
        subprocess.run(["tar", "zxf", tgz_path, "-C", dest_dir], check=True)
        os.remove(tgz_path)

    return data_dir

'''
Example Usage
-------------
full_data_url = 'https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz'
partial_data_url = 'https://storage.googleapis.com/mathwriting_data/mathwriting-2024-excerpt.tgz'

target_dir = './data'

print(download_and_extract_dataset(partial_data_url, target_dir))
'''
