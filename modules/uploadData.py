import os
import zipfile
import random
from pathlib import Path
from PIL import Image
from modules.utils import prepare_directory, remove_directory

def resize_image(image_path: Path, output_size=(64, 64)):
    with Image.open(image_path) as img:
        img = img.resize(output_size)
        img.save(image_path)

def process_images(directory: Path, output_size=(64, 64)):
    for file in directory.iterdir():
        if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
            resize_image(file, output_size)

def get_all_image_files(directory: Path, max_files=100):
    all_files = [file for file in directory.rglob("*") if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]]
    return random.sample(all_files, min(len(all_files), max_files))

def split_and_move_images(files, train_dir, test_dir, split_ratio=0.8):
    random.shuffle(files)
    split_index = int(len(files) * split_ratio)
    
    train_files = files[:split_index]
    test_files = files[split_index:]
    
    for file in train_files:
        category = file.parent.name
        target_folder = train_dir / category
        prepare_directory(target_folder)
        file.rename(target_folder / file.name)

    for file in test_files:
        category = file.parent.name
        target_folder = test_dir / category
        prepare_directory(target_folder)
        file.rename(target_folder / file.name)

def zip_directory(source_folder: Path, data_path_for_zip: Path, zip_name: str):
    zip_path = data_path_for_zip / f"{zip_name}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in source_folder.rglob("*"):
            zipf.write(file, file.relative_to(source_folder))
    print(f"Zipped to: {zip_path}")

def zip_data(zip_name: str):
    # Setup path to data folder
    data_path = Path("data/")
    data_path_for_zip = data_path / "zip"
    photos_source  = data_path / "photos_to_zip"
    output_folder  = data_path_for_zip / zip_name
    train_folder = output_folder / "train"
    test_folder = output_folder / "test"
    
    # Prepare directories
    prepare_directory(data_path)
    prepare_directory(data_path_for_zip)
    prepare_directory(output_folder)
    prepare_directory(train_folder)
    prepare_directory(test_folder)
    
    files_to_process = get_all_image_files(photos_source, max_files=100)
    
    if not files_to_process:
        print("Brak zdjęć do przetworzenia.")
        return

    split_and_move_images(files_to_process, train_folder, test_folder)

    process_images(train_folder, output_size=(64, 64))
    process_images(test_folder, output_size=(64, 64))

    zip_directory(output_folder, data_path_for_zip, zip_name)

    remove_directory(output_folder)
