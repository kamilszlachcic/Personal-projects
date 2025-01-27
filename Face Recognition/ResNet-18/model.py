import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np


# Data loading and preparation
data = fetch_lfw_people(min_faces_per_person=50, resize=0.4)
images, labels = data.images, data.target
target_names = data.target_names

# Normalize images to the range [0, 1]
images = images / 255.0

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)

# Convert images to tensors and add the channel dimension (1 channel: grayscale)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # [batch_size, 1, height, width]
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

# Replicate the grayscale channel to 3 channels (RGB)
X_train_tensor = X_train_tensor.repeat(1, 3, 1, 1)
X_test_tensor = X_test_tensor.repeat(1, 3, 1, 1)

# Convert labels to tensors
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Check tensor shapes
print(f"X_train_tensor shape: {X_train_tensor.shape}")  # Expected: [batch_size, 3, height, width]
print(f"y_train_tensor shape: {y_train_tensor.shape}")  # Expected: [batch_size]

# Load pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)

# Modify the final fully connected layer to match the number of classes in the LFW dataset
model.fc = nn.Linear(model.fc.in_features, len(target_names))

# Move the model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Data augmentation
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, target_names):
        self.images = images
        self.labels = labels
        self.target_names = target_names
        self.augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Apply augmentation only for rare classes (with fewer than 20 samples)
        if np.sum(self.labels == label) <= 20:
            image = self.augment(image)  # Apply augmentation
        else:
            # Convert the image to tensor
            image = torch.tensor(image, dtype=torch.float32)

        # Replicate grayscale channel to RGB channels
        image = image.expand(3, -1, -1)  # Replicate grayscale to 3 channels RGB

        return image, torch.tensor(label, dtype=torch.long)

# Convert y_train labels to tensor
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# Define class weights
class_weights = [1 / torch.sum(y_train_tensor == i).item() for i in range(len(target_names))]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Define the loss function with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Optimizer and learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Prepare DataLoader for training and testing sets
train_loader = DataLoader(AugmentedDataset(X_train, y_train, target_names), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=False)

# Early stopping
best_loss = float('inf')
patience = 15
early_stop_counter = 0

# Training the model
for epoch in range(50):  # Increased the number of epochs
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Update the learning rate scheduler
    scheduler.step()

    # Calculate validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(test_loader)
    print(f"Epoch {epoch+1}, Training Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")

    # Early stopping condition
    if val_loss < best_loss:
        best_loss = val_loss
        early_stop_counter = 0
        best_model_state = model.state_dict()  # Save the best model state
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

# Load the best model
model.load_state_dict(best_model_state)

# Testing the model
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Print the classification report
print(classification_report(all_labels, all_preds, target_names=target_names))

# Save the trained model
torch.save({
    'model_state_dict': model.state_dict(),
    'class_mapping': target_names
}, "model.pth")
