from ultralytics import YOLO
from ultralytics.nn.modules.attention_pruning_block import AttentionPruningBlock

model = YOLO("yolov8n.yaml")
# print(model)

# Freeze all layers except SCAM layers
for name, param in model.model.named_parameters():
    # Freeze all
    param.requires_grad = False

# Now unfreeze SCAM layers
for m in model.model.modules():
    if isinstance(m, AttentionPruningBlock):
        for param in m.parameters():
            param.requires_grad = True


# Important: use the raw model.model, not the YOLO wrapper
core_model = model.model
core_model.eval()

# Use input tensor on CPU
input_tensor = torch.randn(1, 3, 640, 640)

# Use fvcore to calculate FLOPs and Params
flops = FlopCountAnalysis(core_model, input_tensor)
print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")


# Trainable parameters
trainable_params = sum(p.numel() for p in core_model.parameters())
print(f"Trainable Parameters: {trainable_params / 1e6:.2f} Million")


# # Count trainable parameters
# trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
# print(f"Trainable params: {trainable_params}")
# # Print trainable parameters
# for name, param in model.model.named_parameters():
#     if param.requires_grad:
#         print(f"Trainable: {name}")
