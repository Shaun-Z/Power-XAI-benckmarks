from PIL import Image
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models

import matplotlib.pyplot as plt
from datasets import MixedImgDataset


# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset
dataset = MixedImgDataset(root_dir='./data/IFWD', transform=transform)

# Check dataset
print(f"Dataset size: {len(dataset)}")
print(f"Classes: {dataset.classes}")
print(f"Class to index mapping: {dataset.class_to_idx}")

from torch.utils.data import random_split
batch_size = 4
# Define split ratio (e.g., 80% training, 20% testing)
train_ratio = 0.8
test_ratio = 1 - train_ratio

# Calculate lengths
dataset_size = len(dataset)
train_size = int(train_ratio * dataset_size)
test_size = dataset_size - train_size

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create separate dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Print information about the splits
print(f"Total dataset size: {dataset_size}")
print(f"Training set size: {len(train_dataset)}")
print(f"Testing set size: {len(test_dataset)}")

from adtfm import ADTFM_AT_Model_Image
from torch import nn

# 设置设备，优先使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ADTFM_AT_Model_Image(
    in_channels=3,
    cnn_out_channels=128,
    hidden_size=256,
    num_layers=2,
    num_classes=4
).to(device)

# 超参数设置
epochs = 20
batch_size = 4
learning_rate = 1e-4

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        # Skip batches that are too small
        if images.size(0) < 2:
            continue
            
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_dataloader)}], '
                  f'Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')
    
    # Validation
    if (epoch + 1) % 5 == 0:
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        print(f'Validation - Epoch [{epoch+1}/{epochs}], Loss: {val_loss/len(test_dataloader):.4f}, '
              f'Accuracy: {100 * val_correct / val_total:.2f}%')

torch.save(model.state_dict(), f'adt_model4.pth')

print('Training completed.')