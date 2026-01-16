import os
import requests
import tarfile
from tqdm import tqdm


url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
filename = "imagenette2.tgz"
extract_path = "./data"  

def download_file(url, filename):
    print(f"Descargando {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_file(filename, path):
    print(f"Extrayendo {filename} a {path}...")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=path)
    print("¡Listo!")

if __name__ == "__main__":
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    
    # Download the data
    if not os.path.exists(filename):
        download_file(url, filename)
    else:
        print(f"The file {filename} already exist.")

    # Decompress it
    extract_file(filename, extract_path)
    
    print("\nDataset created.")
    print(f"Training directory: {os.path.join(extract_path, 'imagenette2', 'train')}")
    print(f"Validation directory:   {os.path.join(extract_path, 'imagenette2', 'val')}")