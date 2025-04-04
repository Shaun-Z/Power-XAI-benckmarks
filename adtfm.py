import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# 自适应时频记忆单元 (ADTFMCell)
# ----------------------------
class ADTFMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ADTFMCell, self).__init__()
        self.hidden_size = hidden_size
        # 标准 LSTM 门
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size)
        
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)
        
        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size)
        
        self.W_g = nn.Linear(input_size, hidden_size)
        self.U_g = nn.Linear(hidden_size, hidden_size)
        
        # 自适应小波变换参数（用于输入门输出的调制）
        self.W_a = nn.Linear(hidden_size, hidden_size)  # 学习尺度参数
        self.W_b = nn.Linear(hidden_size, hidden_size)  # 学习平移参数
        self.omega0 = 5.0  # 小波中心频率
        
        # 联合遗忘门的简化实现（将三个部分简单相乘）
        self.W_f_ste = nn.Linear(input_size, hidden_size)
        self.U_f_ste = nn.Linear(hidden_size, hidden_size)
        
        self.W_f_tim = nn.Linear(input_size, hidden_size)
        self.U_f_tim = nn.Linear(hidden_size, hidden_size)
        
        self.W_f_fre = nn.Linear(input_size, hidden_size)
        self.U_f_fre = nn.Linear(hidden_size, hidden_size)
        
        # 用于将 cell 状态映射到隐藏状态的线性层
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        
        # 初始化参数，防止梯度爆炸/消失
        self._init_weights()
    
    def _init_weights(self):
        # 使用Xavier初始化线性层的权重，有助于稳定训练
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, h_prev, c_prev):
        # x: (batch, input_size)
        # h_prev, c_prev: (batch, hidden_size)
        # LSTM 门计算
        i = torch.sigmoid(self.W_i(x) + self.U_i(h_prev))
        f_std = torch.sigmoid(self.W_f(x) + self.U_f(h_prev))
        o = torch.sigmoid(self.W_o(x) + self.U_o(h_prev))
        g = torch.tanh(self.W_g(x) + self.U_g(h_prev))
        
        # 输入门输出
        ig = i * g
        
        # 自适应小波变换分支（添加数值稳定性）
        a = torch.clamp(torch.tanh(self.W_a(ig)), min=0.1, max=0.9)  # 限制尺度参数范围，避免极端值
        b = torch.clamp(torch.tanh(self.W_b(ig)), min=-0.9, max=0.9)  # 限制平移参数范围
        
        t_val = 0.0  # 此处取常数，如有需要可传入实际时间信息
        
        # 简化的计算，避免使用复数
        # 使用实部近似，避免复数计算可能导致的NaN
        term1 = torch.cos(self.omega0 * a * (t_val + b))
        term2 = torch.exp(- (1.0 / (a + 1e-8)) * ((t_val + b) ** 2))
        amp = torch.abs(term1 * term2)  # 提取幅值
        
        # 联合遗忘门（添加数值稳定性）
        f_ste = torch.sigmoid(self.W_f_ste(x) + self.U_f_ste(h_prev))
        f_tim = torch.sigmoid(self.W_f_tim(x) + self.U_f_tim(h_prev))
        f_fre = torch.sigmoid(self.W_f_fre(x) + self.U_f_fre(h_prev))
        F_joint = f_ste * f_tim * f_fre
        
        # 更新 cell 状态：添加梯度裁剪以增强稳定性
        c_new = F_joint * c_prev + ig * amp
        # 防止梯度爆炸
        c_new = torch.clamp(c_new, min=-10.0, max=10.0)
        
        # 计算新的隐藏状态
        h_new = o * torch.tanh(self.fc_out(c_new))
        return h_new, c_new

# ----------------------------
# ADTFM 基于 RNN
# ----------------------------
class ADTFM_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(ADTFM_RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # 堆叠多个 ADTFMCell
        self.cells = nn.ModuleList([
            ADTFMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        outputs = []
        for t in range(seq_len):
            inp = x[:, t, :]  # 当前时间步输入
            for i, cell in enumerate(self.cells):
                h[i], c[i] = cell(inp, h[i], c[i])
                inp = h[i]
            outputs.append(h[-1].unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden_size)
        return outputs

# ----------------------------
# 注意力模块
# ----------------------------
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_fc = nn.Linear(hidden_size, hidden_size)
        # 上下文向量，训练时学习
        self.context_vector = nn.Parameter(torch.randn(hidden_size))
    
    def forward(self, rnn_outputs):
        # rnn_outputs: (batch, seq_len, hidden_size)
        u = torch.tanh(self.attention_fc(rnn_outputs))  # (batch, seq_len, hidden_size)
        attn_scores = torch.matmul(u, self.context_vector)  # (batch, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        s = torch.sum(rnn_outputs * attn_weights, dim=1)  # (batch, hidden_size)
        return s

# ----------------------------
# 针对图像输入的 ADTFM‑AT 模型
# ----------------------------
class ADTFM_AT_Model_Image(nn.Module):
    def __init__(self, in_channels, cnn_out_channels, hidden_size, num_layers, num_classes):
        """
        in_channels: 输入图像的通道数（例如 1 或 3）
        cnn_out_channels: CNN 特征提取器输出通道数
        hidden_size: ADTFM 单元的隐藏状态维度
        num_layers: 堆叠 ADTFM 单元的层数
        num_classes: 分类的类别数
        """
        super(ADTFM_AT_Model_Image, self).__init__()
        # CNN 特征提取器，将图像转换为特征图
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 尺寸减半
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 尺寸再减半
            nn.Conv2d(64, cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ReLU()
        )
        # 自适应池化：将特征图的高度固定为 8，宽度保持动态（作为时间步数）
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, None))
        # 经过池化后，每个时间步的特征维度为：cnn_out_channels * 8
        self.seq_feature_dim = cnn_out_channels * 8
        self.rnn = ADTFM_RNN(input_size=self.seq_feature_dim, hidden_size=hidden_size, num_layers=num_layers)
        self.attention = Attention(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        x: (batch, in_channels, height, width)
        经过 CNN 和自适应池化后，将特征图转换为 (batch, seq_len, feature_dim) 的序列，
        再经 RNN、注意力模块和全连接层输出分类 logits。
        """
        features = self.cnn(x)  # (batch, cnn_out_channels, H', W')
        features = self.adaptive_pool(features)  # (batch, cnn_out_channels, 8, W_seq)
        B, C, H_fixed, W_seq = features.size()
        # 将高度和通道维度合并，将宽度作为时间步
        x_seq = features.reshape(B, C * H_fixed, W_seq)  # (batch, C*H_fixed, W_seq)
        x_seq = x_seq.permute(0, 2, 1)  # (batch, W_seq, C*H_fixed)
        rnn_out = self.rnn(x_seq)             # (batch, seq_len, hidden_size)
        attn_out = self.attention(rnn_out)      # (batch, hidden_size)
        logits = self.classifier(attn_out)      # (batch, num_classes)
        return logits

# ----------------------------
# 示例：模型实例化
# ----------------------------
if __name__ == '__main__':
    # 假设输入图像为 RGB 格式，尺寸为 64x64
    model = ADTFM_AT_Model_Image(
        in_channels=3,
        cnn_out_channels=128,
        hidden_size=256,
        num_layers=2,
        num_classes=3  # 例如故障分类有 4 个类别
    )
    print(model)
    
    # 构造一个随机输入：(batch, channel, height, width)
    dummy_input = torch.randn(8, 3, 64, 64)
    logits = model(dummy_input)
    print("输出 logits 形状：", logits.shape)  # 应输出 (8, 4)