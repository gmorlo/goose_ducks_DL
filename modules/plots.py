import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import torch
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_accuracy": [...],
             "test_loss": [...],
             "test_accuracy": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_accuracy']
    test_accuracy = results['test_accuracy']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

def pred_and_plot_image(model: torch.nn.Module, 
                        image_path: str, 
                        class_names: List[str] = None, 
                        transform=None,
                        device: torch.device = device):
    """Makes a prediction on a target image and plots the image with its prediction."""
    
    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    
    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255. 
    
    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)
    
    # 4. Make sure the model is on the target device
    model.to(device)
    
    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)
    
        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))
        
    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    
    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(target_image.squeeze().permute(1, 2, 0)) # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else: 
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)

# def pred_and_plot_image(model: torch.nn.Module, 
#                         image_path: str, 
#                         class_names: List[str] = None, 
#                         transform=None,
#                         device: torch.device = torch.device("cpu")):
#     """Makes a prediction on a target image and plots the image with its prediction."""

#     # 1. Load in image and convert the tensor values to float32
#     target_image = torchvision.io.read_image(str(image_path)).float()

#     # 2. Check if the image has 1 channel (grayscale), convert it to 3 channels (RGB)
#     if target_image.shape[0] == 1:  # Grayscale image (C=1, H, W)
#         target_image = target_image.repeat(3, 1, 1)  # Convert to (3, H, W)

#     # 3. Normalize pixel values to [0, 1]
#     target_image = target_image / 255.0

#     # 4. Apply transformation if provided
#     if transform:
#         target_image = transform(target_image)

#     # 5. Ensure the model is on the correct device
#     model.to(device)

#     # 6. Prepare image for inference
#     model.eval()
#     with torch.inference_mode():
#         target_image = target_image.unsqueeze(0).to(device)  # Add batch dim and move to device
#         target_image_pred = model(target_image)

#     # 7. Convert logits -> probabilities using softmax
#     target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

#     # 8. Get the predicted class label
#     target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1).cpu().item()

#     # 9. Plot the image with the prediction
#     plt.imshow(target_image.squeeze().permute(1, 2, 0))  # Convert (C, H, W) → (H, W, C) for matplotlib
#     plt.title(f"Pred: {class_names[target_image_pred_label] if class_names else target_image_pred_label} "
#               f"| Prob: {target_image_pred_probs.max().cpu().item():.3f}")
#     plt.axis(False)
#     plt.show()