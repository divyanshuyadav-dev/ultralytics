import torch
from torch import nn
from .conv import Conv  # for 1×1 reductions, etc.

class AttentionPruningBlock(nn.Module):
    """
    Combined Spatial‐Channel Attention + Part‐aware Attention + Channel Pruning.
    """
    def __init__(self, c: int):
        super().__init__()
        # 1) Spatial‐Channel Attention
        self.scam = SpatialChannelAttention(c)
        # 2) Part‐aware Attention (you’ll implement this next)
        self.pam = PartAwareAttention(c)
        # 3) Reduce concatenated channels back to c
        self.reduce = Conv(c * 2, c, k=1, s=1)
        # 4) Simple learnable channel pruning mask
        self.prune = nn.Parameter(torch.ones(c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.scam(x)
        x2 = self.pam(x)
        x_cat = torch.cat([x1, x2], dim=1)
        x_red = self.reduce(x_cat)
        # apply pruning mask
        pruned = x_red * self.prune.view(1, -1, 1, 1)
        #element-wise multiplication with input
        out = pruned * x
        return out

class SpatialChannelAttention(nn.Module):
    """
    Lightweight Spatial + Channel Attention Module.
    """
    def __init__(self, channels):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # ----- Channel Attention -----
        ca = self.avg_pool(x)
        ca = self.fc(ca)
        x = x * ca  # broadcast along H, W

        # ----- Spatial Attention -----
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        sa = torch.cat([max_out, avg_out], dim=1)
        sa = self.spatial(sa)
        return x * sa

class PartAwareAttention(nn.Module):
    """
    Learns part-specific attention maps (unsupervised).
    - Applies 1x1 conv to generate N part masks.
    - Multiplies each with the input feature map (elementwise).
    - Aggregates (via sum or mean).
    """
    def __init__(self, channels, num_parts=4):
        super().__init__()
        self.num_parts = num_parts
        self.attn_conv = nn.Conv2d(channels, num_parts, kernel_size=1)
        self.combine = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        # [B, N, H, W] attention masks
        attn = self.attn_conv(x)
        attn = torch.softmax(attn, dim=1)  # normalize across part dimension

        # Multiply input feature with each attention map
        part_features = []
        for i in range(self.num_parts):
            attn_map = attn[:, i:i+1, :, :]           # [B,1,H,W]
            weighted = x * attn_map                   # [B,C,H,W]
            part_features.append(weighted)

        out = torch.stack(part_features, dim=0).sum(0)  # [B,C,H,W]
        return self.combine(out)  # optional channel mixing
