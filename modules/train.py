import os
import torch
import data_setup, engine, model_builder_TinyVGG, utils

from torchvision import transforms

NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Skalowanie obrazów do 64x64
    transforms.ToTensor(),  # Konwersja obrazu do tensora PyTorch
    # transforms.RandomHorizontalFlip(),  # Add horizontal flip
    # transforms.RandomRotation(10),  # Add random rotation
])

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dir = "data/goose_ducks_dataset/test"
test_dir = "data/goose_ducks_dataset/train"

if __name__ == "__main__":
    
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir = train_dir,
        test_dir = test_dir,
        batch_size=BATCH_SIZE,
        transform=transform
    )

    torch.manual_seed(42)
    model = model_builder_TinyVGG.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS, 
        output_shape=len(class_names)
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE)

    engine.train(
        model=model, 
        train_dataloader=train_dataloader, 
        test_dataloader=test_dataloader, 
        optimizer=optimizer,
        loss_fn=loss_fn, 
        epochs=NUM_EPOCHS,
        device=device
    )

    utils.save_model(
        model=model, 
        target_directory="models", 
        model_name="goose_ducks_model_01.pth",
    )