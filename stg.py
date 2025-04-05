import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################
# 1. Temporal Graph Convolution Layer
##############################################
class TemporalGraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalGraphConvolution, self).__init__()
        # Intra-cycle convolution: 沿采样点（宽度 N）进行卷积
        self.intra_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        # Inter-cycle convolution: 沿周期（高度 T）进行卷积
        self.inter_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        # 融合两路卷积输出：输入通道为 out_channels*2，输出 out_channels
        self.combine = nn.Linear(out_channels * 2, out_channels)
    
    def forward(self, x):
        # 输入 x: (batch, T, N, in_channels)，其中 T=height, N=width
        batch, T, N, C = x.shape
        
        # Intra-cycle convolution：沿 N 方向卷积（每个周期内部）
        x_intra = x.view(batch * T, N, C).permute(0, 2, 1)  # (batch*T, C, N)
        out_intra = self.intra_conv(x_intra)  # (batch*T, out_channels, N)
        out_intra = out_intra.permute(0, 2, 1).view(batch, T, N, -1)
        
        # Inter-cycle convolution：沿 T 方向卷积（跨周期，同一采样点）
        x_inter = x.permute(0, 2, 1, 3).contiguous()  # (batch, N, T, C)
        x_inter = x_inter.view(batch * N, T, C).permute(0, 2, 1)  # (batch*N, C, T)
        out_inter = self.inter_conv(x_inter)  # (batch*N, out_channels, T)
        out_inter = out_inter.permute(0, 2, 1).view(batch, N, T, -1).permute(0, 2, 1, 3)
        
        # 拼接两路特征并融合
        out = torch.cat([out_intra, out_inter], dim=-1)  # (batch, T, N, out_channels*2)
        out = self.combine(out)  # (batch, T, N, out_channels)
        return out

##############################################
# 2. Attention Pooling Module
##############################################
class AttentionPool(nn.Module):
    def __init__(self, in_channels):
        super(AttentionPool, self).__init__()
        # 利用 1x1 卷积生成注意力图
        self.attn_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
    
    def forward(self, x):
        # 输入 x: (batch, channels, T, N)
        attn = self.attn_conv(x)  # (batch, 1, T, N)
        attn = attn.view(x.size(0), -1)  # (batch, T*N)
        attn = torch.softmax(attn, dim=1).unsqueeze(1)  # (batch, 1, T*N)
        x_flat = x.view(x.size(0), x.size(1), -1)  # (batch, channels, T*N)
        pooled = torch.bmm(x_flat, attn.transpose(1,2)).squeeze(2)  # (batch, channels)
        return pooled

##############################################
# 3. Graph Embedding Network with Attention Pooling
##############################################
class GraphEmbeddingNet_Attention(nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_dim, num_layers=2):
        """
        in_channels: 输入的通道数（例如图像的 channel）  
        hidden_channels: 每层 TemporalGraphConvolution 的输出通道数  
        embedding_dim: 最终嵌入向量维度  
        num_layers: 堆叠的 TemporalGraphConvolution 层数  
        """
        super(GraphEmbeddingNet_Attention, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(TemporalGraphConvolution(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.layers.append(TemporalGraphConvolution(hidden_channels, hidden_channels))
        # 使用注意力池化代替简单的平均池化
        self.attn_pool = AttentionPool(hidden_channels)
        self.fc = nn.Linear(hidden_channels, embedding_dim)
    
    def forward(self, x):
        # x: (batch, T, N, in_channels)
        for layer in self.layers:
            x = layer(x)  # (batch, T, N, hidden_channels)
        # 转换为 (batch, hidden_channels, T, N)
        x = x.permute(0, 3, 1, 2)
        pooled = self.attn_pool(x)  # (batch, hidden_channels)
        embedding = self.fc(pooled)  # (batch, embedding_dim)
        return embedding

##############################################
# 4. Single Input Temporal Graph Classifier
##############################################
class SingleTemporalGraphClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_dim, num_classes, num_layers=2):
        """
        单输入模型：输入形状为 (batch, channel, height, width) 的图像，
        输出类别 logits。
        in_channels: 输入图像通道数  
        hidden_channels: 图卷积层输出通道数  
        embedding_dim: 嵌入向量维度  
        num_classes: 分类类别数  
        num_layers: 堆叠的 TemporalGraphConvolution 层数  
        """
        super(SingleTemporalGraphClassifier, self).__init__()
        self.embedding_net = GraphEmbeddingNet_Attention(in_channels, hidden_channels, embedding_dim, num_layers)
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        """
        x: (batch, channel, height, width)
        视 height 为周期数 T，width 为每周期采样点数 N，将 x 转换为 (batch, T, N, channel)
        然后提取嵌入，再输出类别 logits。
        """
        # 转换： (batch, channel, height, width) -> (batch, height, width, channel)
        x = x.permute(0, 2, 3, 1).contiguous()
        embedding = self.embedding_net(x)  # (batch, embedding_dim)
        logits = self.classifier(embedding)  # (batch, num_classes)
        return logits

##############################################
# 5. 示例：训练流程
##############################################
if __name__ == '__main__':
    # 假设输入图像数据为 (batch, channel, height, width)
    batch_size = 8
    channel = 1      # 例如单通道波形图像
    height = 16      # 视为周期数 T
    width = 128      # 视为每周期采样点数 N
    in_channels = channel
    hidden_channels = 32
    embedding_dim = 64
    num_classes = 4  # 假设有 4 个类别
    
    # 构造输入图像
    x = torch.randn(batch_size, channel, height, width)
    
    # 实例化模型
    model = SingleTemporalGraphClassifier(in_channels, hidden_channels, embedding_dim, num_classes, num_layers=2)
    
    # 前向传播得到 logits
    logits = model(x)
    print("Logits shape:", logits.shape)  # 应输出 (batch_size, num_classes)
    
    # 假设有目标标签
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 简单训练一步示例
    optimizer.zero_grad()
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    
    print("Loss:", loss.item())