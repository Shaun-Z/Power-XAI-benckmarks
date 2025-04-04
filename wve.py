import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#########################
# 波形数字化相关函数
#########################

def digitize_waveform(waveform):
    """
    对输入的波形（一周期信号，1D tensor）进行 FFT，
    提取直流分量、基波及第2~7谐波的幅值，并根据预设阈值映射为字母。
    对于直流分量和基波，采用间隔 0.1；对于2~7谐波，采用间隔 0.01。
    返回一个长度为8的字母列表。
    """
    # 计算 FFT（这里假设输入为实数信号）
    fft_result = torch.fft.fft(waveform)
    # 提取前8个频率成分的幅值：索引 0：直流；1：基波；2~7：谐波
    magnitudes = torch.abs(fft_result[:8])
    letters = []
    # 对直流和基波使用间隔 0.1
    for i in range(2):
        mag = magnitudes[i].item()
        if mag <= 0.1:
            letter = 'A'
        elif mag <= 0.2:
            letter = 'B'
        elif mag <= 0.3:
            letter = 'C'
        elif mag <= 0.4:
            letter = 'D'
        elif mag <= 0.5:
            letter = 'E'
        elif mag <= 0.6:
            letter = 'F'
        elif mag <= 0.7:
            letter = 'G'
        elif mag <= 0.8:
            letter = 'H'
        else:
            letter = 'H'
        letters.append(letter)
    # 对第2~7谐波使用间隔 0.01
    for i in range(2, 8):
        mag = magnitudes[i].item()
        if mag <= 0.01:
            letter = 'A'
        elif mag <= 0.02:
            letter = 'B'
        elif mag <= 0.03:
            letter = 'C'
        elif mag <= 0.04:
            letter = 'D'
        elif mag <= 0.05:
            letter = 'E'
        elif mag <= 0.06:
            letter = 'F'
        elif mag <= 0.07:
            letter = 'G'
        elif mag <= 0.08:
            letter = 'H'
        else:
            letter = 'H'
        letters.append(letter)
    return letters

def letters_to_onehot(letters):
    """
    将字母列表转换为 one-hot 向量。
    这里采用对 8 个字母（A～H）进行计数，再归一化，得到一个 8 维向量。
    例如：[C, F, C, H, D, B, A, B] --> [1, 2, 2, 1, 0, 1, 0, 1] / 8
    """
    mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
    vec = torch.zeros(8)
    for letter in letters:
        idx = mapping.get(letter, 0)
        vec[idx] += 1
    vec = vec / len(letters)
    return vec  # shape: (8,)

#########################
# 波形向量嵌入模型
#########################

# class WaveformVectorEmbedding(nn.Module):
#     def __init__(self, vocab_size=8):
#         """
#         构造两层神经网络嵌入模型。
#         输入层与输出层均为 vocab_size（默认8），中间无非线性激活（f(x)=x），
#         输出层后采用 softmax 归一化以获得概率分布形式的波形向量。
#         """
#         super(WaveformVectorEmbedding, self).__init__()
#         # 第一层（线性变换，无激活）
#         self.fc = nn.Linear(vocab_size, vocab_size, bias=False)
#         # 输出层（后接 softmax）
#         self.out = nn.Linear(vocab_size, vocab_size, bias=False)
    
#     def forward(self, x):
#         """
#         x: (batch, vocab_size) —— 每个样本为一个 one-hot 向量（8 维）
#         返回：概率分布形式的向量嵌入，形状 (batch, vocab_size)
#         """
#         hidden = self.fc(x)  # f(x)=x
#         logits = self.out(hidden)
#         # 为防止溢出，先减去每个样本的最大值
#         M, _ = torch.max(logits, dim=1, keepdim=True)
#         logits_shifted = logits - M
#         probs = torch.softmax(logits_shifted, dim=1)
#         return probs

class WaveformVectorEmbedding(nn.Module):
    def __init__(self, vocab_size=8):
        super(WaveformVectorEmbedding, self).__init__()
        # Calculate input size for fully connected layer:
        # For 256x256 RGB images (3 channels)
        input_size = 3 * 256 * 256
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate size after feature extraction
        # After 3 MaxPool2d with stride 2, spatial dimensions are reduced by factor of 2^3 = 8
        feature_size = 256 * (256 // 8) * (256 // 8)
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Explicitly flatten the features
            nn.Linear(feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, vocab_size)
        )

    def forward(self, x):
        # Extract features
        x = self.features(x)
        # Classify
        x = self.classifier(x)
        return x

def waveform_embedding_loss(probs, targets, model, reg_weight=1e-3):
    """
    自定义损失函数：
    使用交叉熵损失（对每个样本计算 -sum(target * log(prob)) 的均值），
    并对模型参数（fc 和 out 权重）添加 L2 正则化。
    """
    # 交叉熵损失（假设 targets 是 one-hot 编码）
    ce_loss = -torch.sum(targets * torch.log(probs + 1e-8), dim=1).mean()
    # L2 正则化：对 fc 和 out 权重进行
    reg_loss = 0.5 * (torch.sum(model.fc.weight ** 2) + torch.sum(model.out.weight ** 2)) / model.fc.weight.numel()
    return ce_loss + reg_weight * reg_loss
