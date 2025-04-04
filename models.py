import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T

# ---------------------
# 辅助函数与工具
# ---------------------
def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance
            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def loss_fn(x, y):
    # 计算归一化后欧氏距离（cosine similarity loss）
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# ---------------------
# 数据增强工具
# ---------------------
class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# ---------------------
# 指数滑动平均（EMA）
# ---------------------
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# ---------------------
# MLP 模块（用于投影器和预测器）
# ---------------------
class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------
# NetWrapper：包装基础网络，截取隐藏层输出并生成投影向量
# ---------------------
class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if isinstance(self.layer, str):
            modules = dict(self.net.named_modules())
            return modules.get(self.layer, None)
        elif isinstance(self.layer, int):
            children = list(self.net.children())
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'隐藏层 ({self.layer}) 未找到'
        layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        # 如果指定为最后一层，则直接返回网络输出
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()
        assert hidden is not None, f'隐藏层 {self.layer} 未输出结果'
        return hidden

    def forward(self, x, return_projection=True):
        representation = self.get_representation(x)
        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation

# ---------------------
# BYOL 模型
# ---------------------
class BYOL(nn.Module):
    def __init__(self, net, hidden_layer=-2, projection_size=3, projection_hidden_size=64):
        super().__init__()
        # 使用传入的 net（假设它已经是一个 MLP），包装为在线编码器
        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer)

    def forward(self, x, return_projection=True):
        """
        简化的前向计算：直接返回在线编码器的投影和表示。
        """
        projection, representation = self.online_encoder(x, return_projection=return_projection)
        return projection, representation

# ---------------------
# 示例：模型实例化
# ---------------------
if __name__ == '__main__':
    # 例如，这里构造一个简单的基础网络（可以替换为 ResNet 等）
    base_net = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 128)
    )

    model = BYOL(net=base_net)
    print(model)