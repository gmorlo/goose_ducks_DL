import os
import zipfile
from pathlib import Path
from PIL import Image


def prepare_directory(directory: Path):
    if not directory.exists():
        print(f"Did not find {directory}, creating one...")
        directory.mkdir(parents=True, exist_ok=True)
    else:
        print(f"{directory} directory exists.")

def resize_image(image_path: Path, output_size=(64, 64)):
    with Image.open(image_path) as img:
        img = img.resize(output_size)
        img.save(image_path)

def process_images(directory: Path, output_size=(64, 64)):
    for file in directory.iterdir():
        if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
            resize_image(file, output_size)

def get_all_image_files(directory: Path):
    return [file for file in directory.rglob("*") if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]]

def zip_photos(zip_name: str):
    # Setup path to data folder
    data_path = Path("data/")
    data_path_for_zip = data_path / "zip"
    source_folder = data_path / "photos_to_zip"
    zip_path = data_path_for_zip / f"{zip_name}.zip"
    label_1_path = source_folder / "ducks"
    label_2_path = source_folder / "goose"
    
    # Prepare directories
    prepare_directory(data_path)
    prepare_directory(data_path_for_zip)
    prepare_directory(source_folder)
    prepare_directory(label_1_path)
    prepare_directory(label_2_path)
    
    files_to_zip = get_all_image_files(source_folder)
    
    if not label_2_path and label_1_path:
        print("Brak zdjęć do spakowania. Kod zatrzymany.")
        return
    
    print("Resizing images...")
    process_images(label_1_path, output_size=(64, 64))
    process_images(label_2_path, output_size=(64, 64))

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_zip:
            zipf.write(file, file.relative_to(source_folder))
    
    print(f"Directory created: {zip_path}")