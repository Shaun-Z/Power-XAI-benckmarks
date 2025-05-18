import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# MSNN 模型：仿照文章实现
# ----------------------------
class MSNN_Model(nn.Module):
    def __init__(self, in_channels, num_classes, time_kernel_size=3, freq_kernel_size=3, local_kernel_size=3, global_kernel_size=5, use_time=True, use_freq=True, use_local=True, use_global=True):
        super(MSNN_Model, self).__init__()
        # CWT 层这里用简单的卷积模拟，便于实现
        self.cwt_sim = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        
        # 多通道卷积分支
        self.time_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, time_kernel_size), padding=(0, time_kernel_size//2)),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.freq_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(freq_kernel_size, 1), padding=(freq_kernel_size//2, 0)),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.local_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(local_kernel_size, local_kernel_size), padding=local_kernel_size//2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.global_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(global_kernel_size, global_kernel_size), padding=global_kernel_size//2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.use_time = use_time
        self.use_freq = use_freq
        self.use_local = use_local
        self.use_global = use_global
        
        # 特征融合与分类
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * sum([use_time, use_freq, use_local, use_global]), num_classes)
    
    def forward(self, x):
        """
        x: (batch, in_channels, height, width)
        输出: (batch, num_classes)
        """
        x_cwt = self.cwt_sim(x)  # (batch, 1, H, W)
        
        feats_list = []
        if self.use_time:
            t_feat = self.time_branch(x_cwt)
            feats_list.append(t_feat)
        if self.use_freq:
            f_feat = self.freq_branch(x_cwt)
            feats_list.append(f_feat)
        if self.use_local:
            l_feat = self.local_branch(x_cwt)
            feats_list.append(l_feat)
        if self.use_global:
            g_feat = self.global_branch(x_cwt)
            feats_list.append(g_feat)
        
        feats = torch.cat(feats_list, dim=1)  # (batch, 16*active_branches, H', W')
        pooled = self.global_pool(feats)  # (batch, 16*active_branches, 1, 1)
        flat = self.flatten(pooled)       # (batch, 16*active_branches)
        logits = self.fc(flat)            # (batch, num_classes)
        return logits

# ----------------------------
# 示例：MSNN 模型实例化
# ----------------------------
if __name__ == '__main__':
    # 假设输入为单通道 64x64 图像
    msnn_model = MSNN_Model(
        in_channels=1,
        num_classes=3,
        time_kernel_size=3,
        freq_kernel_size=3,
        local_kernel_size=3,
        global_kernel_size=5,
        use_time=True,
        use_freq=True,
        use_local=True,
        use_global=True
    )
    print(msnn_model)
    
    # 构造一个随机输入：(batch, channel, height, width)
    dummy_input = torch.randn(8, 1, 64, 64)
    logits = msnn_model(dummy_input)
    print("输出 logits 形状：", logits.shape)  # 应输出 (8, 3)