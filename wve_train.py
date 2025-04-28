# ================================
# 导入依赖
# ================================
from PIL import Image
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt

from datasets import MixedImgDataset
from wve import WaveformVectorEmbedding

# ================================
# 设置设备
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# 定义 transforms
# ================================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ================================
# 加载数据集
# ================================
dataset = MixedImgDataset(root_dir='./data/MixedImg', transform=transform)

# 查看数据集情况
print(f"Dataset size: {len(dataset)}")
print(f"Classes: {dataset.classes}")
print(f"Class to index mapping: {dataset.class_to_idx}")

# 划分训练集、测试集
batch_size = 4
train_ratio = 0.8
test_ratio = 1 - train_ratio

dataset_size = len(dataset)
train_size = int(train_ratio * dataset_size)
test_size = dataset_size - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print(f"Training set size: {len(train_dataset)}")
print(f"Testing set size: {len(test_dataset)}")

# ================================
# 初始化模型
# ================================
model = WaveformVectorEmbedding(vocab_size=3, num_fft_components=256).to(device)

# ================================
# 定义超参数
# ================================
epochs = 80
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# ================================
# 训练循环
# ================================
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        if images.size(0) < 2:
            continue

        images = images.to(device)
        labels = labels.to(device)

        # =============== 前向传播 ===============
        optimizer.zero_grad()
        outputs = model(images)

        # 计算 loss（使用 nn.CrossEntropyLoss）
        loss = criterion(outputs, labels)

        # =============== 反向传播和优化 ===============
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_dataloader)}], '
                  f'Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')

    # =======================
    # 验证集评估
    # =======================
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

# ================================
# 保存模型
# ================================
torch.save(model.state_dict(), 'wve_model.pth')
print('Training completed.')