import torch
from ultralytics.nn.modules.ada import GroupedSpatialAttention

# Define dummy input
batch_size = 2
channels = 64
height = 64
width = 64

# Create dummy input tensor
x = torch.randn(batch_size, channels, height, width)

# Initialize the module
groups = 4
num_maps = 4
gsa = GroupedSpatialAttention(channels, groups=groups, num_maps=num_maps)

# Forward pass
output = gsa(x)

# Sanity checks
print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")

# Optional: Check attention effect
diff = (output - x).abs().mean().item()
print(f"Average absolute difference between input and output: {diff:.4f}")

# Optional: Check attention range
print(f"Min/max output: {output.min().item():.4f} / {output.max().item():.4f}")



from ultralytics.nn.modules.ada import ChannelAttention1D

x = torch.randn(2, 64, 64, 64)
ca = ChannelAttention1D(64)
out = ca(x)

print(f"Output shape: {out.shape}")
print(f"Mean abs diff from input: {(out - x).abs().mean():.4f}")





from ultralytics.nn.modules.ada import ADA

x = torch.randn(2, 64, 64, 64)
ada = ADA(64, groups=4, num_maps=4)
out = ada(x)

print(f"ADA Output shape: {out.shape}")
print(f"Mean abs diff: {(out - x).abs().mean():.4f}")
