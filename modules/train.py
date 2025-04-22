import os
import torch
import data_setup, engine, model_builder.model_builder_TinyVGG as model_builder_TinyVGG, utils

from torchvision import transforms

def train(
    num_epochs: int,
    batch_size: int,
    hidden_units: int,
    learning_rate: float,
    train_dir: str,
    test_dir: str,
    model_name: str,
    transform: transforms.Compose,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    

    # NUM_EPOCHS = 5
    # BATCH_SIZE = 32
    # HIDDEN_UNITS = 10
    # LEARNING_RATE = 0.001
    # train_dir = "data/gd_dataset/test"
    # test_dir = "data/gd_dataset/train"

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir = train_dir,
        test_dir = test_dir,
        batch_size = batch_size,
        transform = transform
    )

    torch.manual_seed(42)
    model = model_name(
        input_shape = 3,
        hidden_units = hidden_units, 
        output_shape = len(class_names)
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                    lr=learning_rate)

    engine.train(
        model=model, 
        train_dataloader=train_dataloader, 
        test_dataloader=test_dataloader, 
        optimizer=optimizer,
        loss_fn=loss_fn, 
        epochs=num_epochs,
        device=device
    )

    model_path_name = f"{model_name}_{num_epochs}epochs_{batch_size}batch_size.pth"

    utils.save_model(
        model=model, 
        target_directory="models", 
        model_path_name=model_path_name,
    )