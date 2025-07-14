import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# Set parameters
data_dir = 'dataset'  
batch_size = 8
num_epochs = 5
image_size = 128

# Define transform only for training
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Load only training data
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
class_names = train_dataset.classes

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (image_size // 4) * (image_size // 4), 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Update this if more than 2 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN().to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train loop (only for training)
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

# Save the model
torch.save(model.state_dict(), 'simple_cnn.pth')
print("\n✅ Model saved as 'simple_cnn.pth'")

# Predict a single test image
test_image_path = r'C:\Users\jayan\OneDrive\Desktop\jay\test.jpg'
if os.path.exists(test_image_path):
    img = Image.open(test_image_path).convert('RGB')
    img_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    img_tensor = img_transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]
        print(f"\n🖼️ Predicted class for test.jpg: {predicted_class}")

    plt.imshow(np.array(img))
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()
else:
    print("\n⚠️ test.jpg not found.")

