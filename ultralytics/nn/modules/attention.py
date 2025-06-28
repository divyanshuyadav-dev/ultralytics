import torch
import torch.nn as nn

class SCAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SCAM, self).__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(f"[SCAM] Input shape: {x.shape}")
        # Channel Attention
        ca = self.channel_att(x)
        x = x * ca

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_att(sa_input)
        x = x * sa

        # print(f"[SCAM] Output shape: {x.shape}")
        return x
