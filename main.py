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



# # Count trainable parameters
# trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
# print(f"Trainable params: {trainable_params}")
# # Print trainable parameters
# for name, param in model.model.named_parameters():
#     if param.requires_grad:
#         print(f"Trainable: {name}")
