from pathlib import Path
import os
import torch


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

def save_model(
        model: torch.nn.Module,
        target_directory: str,
        model_name: str,
):
    target_directory_path = Path(target_directory)
    target_directory_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name must end with .pth or .pt"
    model_save_path = target_directory_path / model_name

    print(f"Saving model to {model_save_path}")
    torch.save(
        obj=model.state_dict(), 
        f=model_save_path
        )