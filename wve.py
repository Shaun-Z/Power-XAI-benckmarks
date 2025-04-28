import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#########################
# 波形数字化相关函数
#########################

def extract_fft_features(waveform, num_fft_components=64):
    """
    输入一维波形信号，计算其 FFT，提取前 num_fft_components 个频率成分的幅值，
    并进行归一化处理（除以最大幅值）。
    返回形状为 (num_fft_components,) 的张量。
    """
    fft_result = torch.fft.fft(waveform)
    magnitudes = torch.abs(fft_result)[:num_fft_components]
    max_val = magnitudes.max()
    if max_val > 0:
        magnitudes = magnitudes / max_val
    return magnitudes

#########################
# 波形向量嵌入模型
#########################

class WaveformVectorEmbedding(nn.Module):
    def __init__(self, vocab_size=8, num_fft_components=64):
        super(WaveformVectorEmbedding, self).__init__()
        self.num_fft_components = num_fft_components
        self.fc1 = nn.Linear(num_fft_components, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, vocab_size)

    def forward(self, x):
        """
        x: (batch_size, channels, height, width) image tensor
        对每张图像提取第一个通道，展平为 1D 波形，
        计算归一化 FFT 特征，批量处理后通过三层 MLP。
        返回形状为 (batch_size, vocab_size) 的张量。
        """
        batch_size = x.shape[0]
        # 提取第一个通道，展平为 (batch_size, height*width)
        waveforms = x[:].reshape(batch_size, -1)
        # 计算 FFT 特征，批量处理
        fft_features = torch.fft.fft(waveforms)
        magnitudes = torch.abs(fft_features)[:, :self.num_fft_components]
        max_vals = magnitudes.max(dim=1, keepdim=True)[0]
        max_vals[max_vals == 0] = 1.0  # 避免除零
        normalized_magnitudes = magnitudes / max_vals
        # MLP
        out = F.relu(self.fc1(normalized_magnitudes))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

def waveform_embedding_loss(probs, targets, model, reg_weight=1e-3):
    """
    自定义损失函数：
    使用交叉熵损失（对每个样本计算 -sum(target * log(prob)) 的均值），
    并对模型参数添加 L2 正则化。
    """
    probs = torch.softmax(probs, dim=1)
    ce_loss = -torch.sum(targets * torch.log(probs + 1e-8), dim=1).mean()
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2)
    reg_loss = 0.5 * reg_loss / sum(p.numel() for p in model.parameters())
    return ce_loss + reg_weight * reg_loss
