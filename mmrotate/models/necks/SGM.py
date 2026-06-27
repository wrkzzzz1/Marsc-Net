import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.models.builder import ROTATED_NECKS

@ROTATED_NECKS.register_module()
class SGM(nn.Module):
    def __init__(self, in_channels=256, reduction=4):  # ← 减小reduction提升表达能力
        super().__init__()
        self.saliency_convs = nn.ModuleList([
            nn.Sequential(
                # 改用3x3卷积捕获旋转特征 ↓
                nn.Conv2d(in_channels, in_channels//reduction, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels//reduction),
                nn.ReLU(),
                # 空间注意力增强 ↓
                nn.Conv2d(in_channels//reduction, 1, 3, padding=1),
                nn.Sigmoid()
            ) for _ in range(5)
        ])
        
        # 仅对P3-P5添加轻量级精炼 ↓
        self.refiners = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, 1) if i<3 else nn.Identity() 
            for i in range(5)
        ])

    def forward(self, inputs):
        return [
            x + self.refiners[i](x) * self.saliency_convs[i](x)  # 残差增强
            for i, x in enumerate(inputs)
        ]