"""Python script to get the mnist data set.

Mostly copied from: https://huggingface.co/datasets/mnist/blob/main/mnist.py
"""

import os
import requests
import gzip
import shutil
from PIL import Image

_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_URLS = {
    "train_labels": "train-labels-idx1-ubyte.gz",
    "train_images": "train-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
}

urls_to_download = {key: _URL + fname for key, fname in _URLS.items()}

print(urls_to_download)
script_dir = os.path.dirname(os.path.abspath(__file__))
download_dir = os.path.join(script_dir, "..", "datasets", "mnist")
os.makedirs(download_dir, exist_ok=True)

for key, url in urls_to_download.items():
    # Download the file
    file_path = os.path.join(download_dir, key + ".gz")
    print(f"Downloading {key}...")
    response = requests.get(url, stream=True)
    with open(file_path, "wb") as file:
        shutil.copyfileobj(response.raw, file)

    # Unzip the file
    unzipped_file_path = os.path.join(download_dir, key)
    with gzip.open(file_path, "rb") as gz_file:
        with open(unzipped_file_path, "wb") as unzipped_file:
            shutil.copyfileobj(gz_file, unzipped_file)

    # Remove the gzipped file
    os.remove(file_path)

    if "images" in key:
        label_key = key.replace("images", "labels")
        label_file_path = os.path.join(download_dir, label_key)
        if "train" in key:
            out_dir = "train"
        else:
            out_dir = "test"
        out_dir = os.path.join(download_dir, out_dir)
        os.makedirs(out_dir, exist_ok=True)

        with open(unzipped_file_path, "rb") as image_file, \
             open(label_file_path, "rb") as label_file:
            magic_number = int.from_bytes(image_file.read(4), byteorder="big")
            num_images = int.from_bytes(image_file.read(4), byteorder="big")
            num_rows = int.from_bytes(image_file.read(4), byteorder="big")
            num_cols = int.from_bytes(image_file.read(4), byteorder="big")

            label_file.read(8)  # Skip the label file header

            for i in range(num_images):
                image_data = image_file.read(num_rows * num_cols)
                image = Image.new("L", (num_cols, num_rows))
                image.putdata(image_data)

                label = int.from_bytes(label_file.read(1), byteorder="big")

                image_filename = f"{label}_{i}.jpg"
                image.save(os.path.join(out_dir, image_filename))

        # Remove the original image and label files
        os.remove(unzipped_file_path)
        os.remove(label_file_path)

print("All files downloaded and unzipped successfully.")
