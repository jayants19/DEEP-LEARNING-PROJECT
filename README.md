# DEEP-LEARNING-PROJECT

# 🐶🐱 Image Classification using CNN in PyTorch

This project performs binary image classification (cats vs dogs) using a custom **Convolutional Neural Network (CNN)** built with **PyTorch**.

---

## 🎯 Objective

Train a CNN model to:

- Classify input images as either **cat** or **dog**  
- Evaluate performance using validation data  
- Predict on unseen test images using the saved model

---

## 📁 Input Data

Input is read from a folder named: `dataset/`

The directory structure should be:


- `train/` is used to train the model   
- Images are resized to **128×128 pixels** during preprocessing

---

## 🧠 What This Project Does

1. **Preprocesses the Image Data**  
   - Resizes all images to 128×128  
   - Applies `ToTensor()` to convert images to tensors  
   - Adds data augmentation with `RandomHorizontalFlip()` for training images

2. **Builds a Simple CNN Model**  
   - 2 convolutional layers with ReLU and MaxPooling  
   - 2 fully connected layers  
   - Final output layer with 2 units (cat or dog)

3. **Trains the Model**  
   - Optimizer: `Adam`  
   - Loss Function: `CrossEntropyLoss`  
   - Trains for 5 epochs  
   - Prints accuracy and loss for both train and val after each epoch

4. **Saves the Model**  
   - After training, the model is saved as `simple_cnn.pth`

5. **Runs Inference on a Test Image**  
   - Reads `test.jpg` from the current folder  
   - Predicts whether it is a **cat** or a **dog**  
   - Displays the image with the predicted label using `matplotlib`

---




