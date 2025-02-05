import os
import requests
import zipfile
from pathlib import Path
from modules.utils import prepare_directory

def get_data_from_github(placeholder: str):

    data_path = Path("data/")
    image_path = data_path / "goose_ducks_dataset"

    prepare_directory(data_path)
    prepare_directory(image_path)

    with open(data_path / f"{placeholder}.zip", "wb") as f:
        request = requests.get(f"https://github.com/gmorlo/goose_ducks_DL/raw/main/data/zip/{placeholder}.zip")
        # request = requests.get(f"https://github.com/gmorlo/goose_ducks_DL/raw/main/data/zip/goose_ducks_labeled.zip")

        print(f"Downloading {placeholder} data...")
        print(f"Response status code: {request.status_code}")
        print(f"Response headers: {request.headers}")
        f.write(request.content)


    with zipfile.ZipFile(data_path / f"{placeholder}.zip", "r") as zip_ref:
        print(f"Unzipping {placeholder} data...") 
        zip_ref.extractall(image_path)

    os.remove(data_path / f"{placeholder}.zip")