# DEEP-LEARNING-PROJECT

🐶🐱 Image Classification with CNN (PyTorch)
This project demonstrates how to build, train, and evaluate a simple Convolutional Neural Network (CNN) to classify images of cats and dogs using PyTorch.

🛠️ Technologies Used
Python 3

PyTorch & TorchVision

PIL (Python Imaging Library)

Matplotlib (for visualization)

🧠 Model Architecture
A custom CNN model with:

Two Conv2d layers + ReLU + MaxPooling

One Flatten layer

Two Linear layers (fully connected)

Final output: 2 classes (cat, dog)
🔄 Workflow Summary
Preprocessing

Images resized to 128×128

Transformed to tensors

Random horizontal flip added to training data for augmentation

Training

Loss function: CrossEntropyLoss

Optimizer: Adam

Trained for 5 epochs

Accuracy and loss reported for both train and validation sets

Saving the Model

Trained model saved as: simple_cnn.pth

Inference

Loads test.jpg from root directory

Predicts whether the image is a cat or dog

Displays result using matplotlib
