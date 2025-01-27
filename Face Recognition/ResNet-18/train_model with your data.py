import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Load the LFW dataset (faces dataset)
data = fetch_lfw_people(min_faces_per_person=50, resize=0.4)
images, labels = data.images, data.target
target_names = data.target_names

# Normalize images to range [0, 1]
images = images / 255.0

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)

# Path to folder containing new images for training
new_data_path = "NewPhotos"

# Load the pre-trained model and class mappings
model = models.resnet18(num_classes=12)
checkpoint = torch.load("model.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
class_mapping = checkpoint['class_mapping']

# Configure the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Update class mappings to include new classes
new_classes = ["Kamil", "Malwina"]
new_target_names = list(class_mapping) + new_classes
new_label_mapping = {name: idx for idx, name in enumerate(new_target_names)}

# Modify the final layer of the model to match the new number of classes
model.fc = nn.Linear(model.fc.in_features, len(new_target_names))
model = model.to(device)

# Load new images and labels
new_images = []
new_labels = []

# Print the updated class names
print("New classes:", new_target_names)

# Create a label-to-index mapping for new classes
new_label_mapping = {name: i for i, name in enumerate(new_target_names)}

# Process new images and assign labels based on file names
for file_name in os.listdir(new_data_path):
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        label_name = file_name.split("_")[0]  # Extract the label from file name
        label_idx = new_label_mapping[label_name.capitalize()]  # Map label to index
        img = Image.open(os.path.join(new_data_path, file_name)).convert("RGB")
        img = img.resize((X_train.shape[1], X_train.shape[2]))  # Resize to match LFW dimensions
        img = np.array(img) / 255.0  # Normalize image
        new_images.append(img)
        new_labels.append(label_idx)

# Convert new images and labels to numpy arrays
new_images = np.array(new_images)
new_labels = np.array(new_labels)

# Split new data into training and testing sets
X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(new_images, new_labels, test_size=0.2, random_state=42)

# Define transformations for resizing and normalization
resize_transform = transforms.Compose([
    transforms.ToPILImage(),        # Convert numpy image to PIL image
    transforms.Resize((50, 37)),    # Resize to match training data dimensions
    transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale (RGB)
    transforms.ToTensor()           # Convert PIL image to tensor
])

# Process training and testing images with the defined transformations
X_new_train_resized = [resize_transform(img) for img in X_new_train]
X_new_train_tensor = torch.stack(X_new_train_resized)  # Combine into a tensor

X_new_test_resized = [resize_transform(img) for img in X_new_test]
X_new_test_tensor = torch.stack(X_new_test_resized)

# Convert new labels to tensors
y_new_train_tensor = torch.tensor(y_new_train, dtype=torch.long)
y_new_test_tensor = torch.tensor(y_new_test, dtype=torch.long)

# Define a dataset class with augmentation for rare classes
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

        # Apply augmentation only for rare classes
        if np.sum(self.labels == label) <= 20:
            image = self.augment(image)
        else:
            image = torch.tensor(image, dtype=torch.float32)

        # Expand grayscale to 3 channels for compatibility
        image = image.expand(3, -1, -1)

        return image, torch.tensor(label, dtype=torch.long)

# Combine original and new training/testing datasets
combined_train_loader = DataLoader(
    torch.utils.data.ConcatDataset([
        AugmentedDataset(X_train, y_train, target_names),
        TensorDataset(X_new_train_tensor, y_new_train_tensor)
    ]),
    batch_size=32,
    shuffle=True
)

combined_test_loader = DataLoader(
    torch.utils.data.ConcatDataset([
        AugmentedDataset(X_test, y_test, target_names),
        TensorDataset(X_new_test_tensor, y_new_test_tensor)
    ]),
    batch_size=32,
    shuffle=True
)

# Adjust the final layer to match the new number of classes
model.fc = nn.Linear(model.fc.in_features, len(new_target_names))
model = model.to(device)

# Combine original and new training labels
combined_y_train = torch.cat([torch.tensor(y_train, dtype=torch.long), y_new_train_tensor])

# Define class weights to handle class imbalance
class_weights = [1 / torch.sum(combined_y_train == i).item() if torch.sum(combined_y_train == i).item() != 0 else 1 for i in range(len(new_target_names))]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Define loss function with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Define optimizer and learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Early stopping parameters
epoch_num = 50
best_loss = float('inf')
patience = 15
early_stop_counter = 0

# Training loop
for epoch in range(epoch_num):
    model.train()
    running_loss = 0.0
    for images, labels in combined_train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    # Validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in combined_test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(combined_test_loader)
    print(f"Epoch {epoch+1}, Training Loss: {running_loss / len(combined_test_loader):.4f}, Validation Loss: {val_loss:.4f}")

    # Early stopping condition
    if val_loss < best_loss:
        best_loss = val_loss
        early_stop_counter = 0
        best_model_state = model.state_dict()
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

# Load the best model state
model.load_state_dict(best_model_state)

# Testing the model
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in combined_test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Print classification report
print(classification_report(all_labels, all_preds, target_names=new_target_names))

# Save the trained model based on user input
print("Do you want to save the trained model? (yes/no)")
user_input = input().strip().lower()
if user_input == "yes":
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_mapping': new_label_mapping
    }, 'model_updated.pth')
    print("Model saved as 'model_updated.pth'.")
else:
    print("Model not saved.")
