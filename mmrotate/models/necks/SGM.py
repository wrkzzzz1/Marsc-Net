import torch.nn as nn
from mmrotate.models.builder import ROTATED_NECKS


@ROTATED_NECKS.register_module()
class SGM(nn.Module):
    def __init__(self, in_channels=256, reduction=4):
        super(SGM, self).__init__()

        hidden_channels = in_channels // reduction

        # Generate level-specific spatial saliency maps for P2-P6
        self.saliency_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    hidden_channels,
                    1,
                    kernel_size=3,
                    padding=1
                ),
                nn.Sigmoid()
            )
            for _ in range(5)
        ])

        # P2-P4 use 1x1 convolution for lightweight feature refinement;
        # P5-P6 use identity mapping to preserve high-level semantic features.
        self.refiners = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
            if i < 3 else nn.Identity()
            for i in range(5)
        ])

    def forward(self, inputs):
        assert len(inputs) == 5, 'SGM expects five input feature maps: P2-P6.'

        outputs = []
        for i, x in enumerate(inputs):
            saliency = self.saliency_convs[i](x)
            refined = self.refiners[i](x)
            out = x + refined * saliency
            outputs.append(out)

        return tuple(outputs)
