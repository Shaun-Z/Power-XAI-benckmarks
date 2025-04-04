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
    def __init__(
        self,
        net,
        image_size,
        hidden_layer=-2,
        projection_size=16,
        projection_hidden_size=64,
        augment_fn=None,
        augment_fn2=None,
        moving_average_decay=0.99,
        use_momentum=True
    ):
        super().__init__()
        self.net = net

        # 默认数据增强（利用 torchvision.transforms）
        DEFAULT_AUG1 = T.Compose([
            T.RandomHorizontalFlip(p=0.2),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])
            ),
        ])
        DEFAULT_AUG2 = T.Compose([
            T.RandomVerticalFlip(p=0.2),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])
            ),
        ])
        self.augment1 = default(augment_fn, DEFAULT_AUG1)
        self.augment2 = default(augment_fn2, DEFAULT_AUG2)

        # 在线编码器：包装基础网络并截取隐藏层输出
        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        # 在线预测器：通过 MLP 对在线编码器的投影进行预测
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # 将模型移动到基础网络所在的设备上
        device = get_module_device(net)
        self.to(device)
        # 通过传入虚拟数据初始化相关参数
        self.forward(torch.randn(4, 3, image_size, image_size, device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, '已关闭目标编码器的动量更新'
        assert self.target_encoder is not None, '目标编码器尚未创建'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, x, return_embedding=False, return_projection=True):
        # 训练时 batch 大小不能为 1（因 BatchNorm 限制）
        assert not (self.training and x.shape[0] == 1), '训练时 batch 大小必须大于1'
        if return_embedding:
            return self.online_encoder(x, return_projection=return_projection)

        # 对输入进行两种不同的数据增强，生成两视图
        image_one, image_two = self.augment1(x), self.augment2(x)

        online_proj_one, _ = self.online_encoder(image_one)
        online_proj_two, _ = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one, _ = target_encoder(image_one)
            target_proj_two, _ = target_encoder(image_two)
            target_proj_one.detach_()
            target_proj_two.detach_()

        # 计算 BYOL 损失（对称结构）
        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())
        loss = loss_one + loss_two
        return loss.mean()

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

    image_size = 224
    model = BYOL(net=base_net, image_size=image_size)
    print(model)