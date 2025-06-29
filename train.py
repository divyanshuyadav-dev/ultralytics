from ultralytics import YOLO
from ultralytics.nn.modules.attention_pruning_block import AttentionPruningBlock
model = YOLO("yolov8n.yaml")
# Load the pre-trained weights
model.load("yolov8n.pt")  # This will load weights where possible


# Freeze all layers except SCAM layers
for name, param in model.model.named_parameters():
    # Freeze all
    param.requires_grad = False

# Now unfreeze SCAM layers
for m in model.model.modules():
    if isinstance(m, AttentionPruningBlock):
        for param in m.parameters():
            param.requires_grad = True

# print(model)

# # Count trainable parameters
# trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
# print(f"Trainable params: {trainable_params}")
# # Print trainable parameters
# for name, param in model.model.named_parameters():
#     if param.requires_grad:
#         print(f"Trainable: {name}")


# Load the pre-trained weights
model.train(
    data='ultralytics/datasets/pig_face/data.yaml',
    # epochs=50,
    # imgsz=640,
    # batch=16,
    epochs=10,             # reduce from 50
    imgsz=416,             # reduce from 640
    batch=4,               # reduce from 16
    workers=0,             # force CPU safe behavior
    name='scam_finetune_pigface',
    project='runs/train'
)