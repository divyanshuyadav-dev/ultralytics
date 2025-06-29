import torch
import torch.nn as nn
import torch.nn.functional as F

# GroupedSpatialAttention
class GroupedSpatialAttention(nn.Module):
    """
    Spatial attention module with grouped convolutions to produce diverse attention maps.
    """
    def __init__(self, channels, groups=4, num_maps=4):
        super().__init__()
        self.groups = groups
        self.num_maps = num_maps

        self.pool = nn.AdaptiveAvgPool2d(1)  # could also add max pool optionally
        self.conv = nn.Conv2d(channels, groups * num_maps, kernel_size=1, groups=groups)
        self.bn = nn.BatchNorm2d(groups * num_maps)
        self.activation = nn.Sigmoid()  # we can switch to softmax along map axis if needed

    def forward(self, x):
        B, C, H, W = x.shape

        # Apply 1x1 group conv to generate multiple maps
        attn_maps = self.conv(x)       # Shape: [B, G*K, H, W]
        attn_maps = self.bn(attn_maps)
        attn_maps = self.activation(attn_maps)

        # Reshape into [B, G, K, H, W]
        attn_maps = attn_maps.view(B, self.groups, self.num_maps, H, W)

        # Optionally: normalize maps within each group (per the AGDL loss idea)
        attn_maps = F.softmax(attn_maps, dim=2)  # Along K maps

        # Aggregate maps (mean across K)
        attn_out = attn_maps.mean(dim=2)  # Shape: [B, G, H, W]

        # Expand group dimension to channel dimension
        attn_out = attn_out.view(B, self.groups, 1, H, W).repeat(1, 1, C // self.groups, 1, 1)
        attn_out = attn_out.view(B, C, H, W)

        # Apply attention to input
        return x * attn_out

# ChannelAttention1D
class ChannelAttention1D(nn.Module):
    """
    Channel attention using 1D convolution across channels.
    Dynamically adjusts kernel size based on input channels.
    """
    def __init__(self, channels):
        super().__init__()
        kernel_size = self.get_kernel_size(channels)
        padding = (kernel_size - 1) // 2  # same padding
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # [B, C, 1, 1]
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def get_kernel_size(self, c):
        # Dynamic kernel size, as proposed
        return int(abs((torch.log2(torch.tensor(c, dtype=torch.float32)).item()) + 1)) | 1  # Ensure it's odd

    def forward(self, x):
        B, C, H, W = x.shape
        pooled = self.avg_pool(x)        # [B, C, 1, 1]
        pooled = pooled.view(B, 1, C)    # [B, 1, C] - to match Conv1d
        attn = self.conv(pooled)         # [B, 1, C]
        attn = self.sigmoid(attn)        # [B, 1, C]
        attn = attn.view(B, C, 1, 1)     # [B, C, 1, 1]
        return x * attn                  # scale input



class ADA(nn.Module):
    """
    Augment Group Diverse Attention (ADA) module from CPADA.
    Combines spatial and channel attention sequentially.
    """
    def __init__(self, channels, groups=4, num_maps=4):
        super().__init__()
        self.spatial = GroupedSpatialAttention(channels, groups, num_maps)
        self.channel = ChannelAttention1D(channels)

    def forward(self, x):
        x = self.spatial(x)
        x = self.channel(x)
        return x
