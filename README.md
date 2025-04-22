# Goose Ducks DL

A deep learning project to classify geese vs ducks in images using CNN (Convolutional Neural Network) models. This project is based on a course by Daniel Bourke and is currently **under construction** (in progress).

## Features

- **Image Classification**: Distinguish between images of geese and ducks using CNN-based models.  
- **Pre-trained Models**: Fine-tune two different pre-trained CNN models on the goose/duck dataset.  
- **Custom CNN Models**: Build and train two simple custom CNN models from scratch, and compare their performance with the pre-trained models on the same dataset.  
- **Logging System**: Implement a logging mechanism to record training results and metrics for each model.  
- **TensorBoard Monitoring**: Use TensorBoard to monitor training progress and visualize performance metrics during training.  
- **Model Deployment**: Plan and prepare the deployment of the best-performing model (e.g. as a web application or service) once training is complete.  

## Dataset

- The dataset consists of images of geese and ducks organized into separate class directories for training and testing.  
- A zip file of the dataset (`gd_dataset.zip`) is included in the project repository, along with a few sample images of geese and ducks for preview.  

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/gmorlo/goose_ducks_DL.git
   cd goose_ducks_DL
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Training Script:**

   ```bash
   python train.py
   ```

*Detailed setup instructions will be added as the project progresses.*

## Technologies Used

- **Python**: Core programming language used.  
- **PyTorch**: Deep learning framework for building and training the CNN models.  
- **Torchvision**: PyTorch library providing pre-trained models and image handling utilities.  
- **TensorBoard**: Tool for visualizing training metrics and model performance over time.  

## License

This project is licensed under the MIT License. Feel free to use or modify the code for personal and educational purposes.