from pathlib import Path

def prepare_directory(directory: Path):
    if not directory.exists():
        print(f"Did not find {directory}, creating one...")
        directory.mkdir(parents=True, exist_ok=True)
    else:
        print(f"{directory} directory exists.")