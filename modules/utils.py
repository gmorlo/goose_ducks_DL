from pathlib import Path
import os


def prepare_directory(directory: Path):
    if not directory.exists():
        print(f"Did not find {directory}, creating one...")
        directory.mkdir(parents=True, exist_ok=True)
    else:
        print(f"{directory} directory exists.")

def remove_directory(directory: Path):
    for item in sorted(directory.rglob("*"), key=lambda x: -len(x.parts)):
        if item.is_file():
            os.remove(item)
        elif item.is_dir():
            try:
                os.rmdir(item)
            except OSError:
                pass  
    try:
        os.rmdir(directory)
    except OSError:
        pass