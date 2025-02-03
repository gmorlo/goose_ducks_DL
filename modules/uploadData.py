import os
import zipfile
from pathlib import Path

def prepare_directory(directory: Path):
    if not directory.exists():
        print(f"Did not find {directory}, creating one...")
        directory.mkdir(parents=True, exist_ok=True)
    else:
        print(f"{directory} directory exists.")

def get_image_files(directory: Path):
    return [file for file in directory.iterdir() if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]]

def zip_photos(zip_name: str):
    # Setup path to data folder
    data_path = Path("data/")
    source_folder = data_path / "photos_to_zip"
    zip_path = data_path / f"{zip_name}.zip"
    
    # Prepare directories
    prepare_directory(data_path)
    prepare_directory(source_folder)
    
    files_to_zip = get_image_files(source_folder)
    
    if not files_to_zip:
        print("Brak zdjęć do spakowania. Kod zatrzymany.")
        return
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_zip:
            zipf.write(file, file.relative_to(source_folder))
    
    print(f"Directory created: {zip_path}")