from ultralytics import YOLO
from ultralytics.nn.modules.attention import SCAM

model = YOLO("yolov8n.yaml")


# Freeze all layers except SCAM layers
for name, param in model.model.named_parameters():
    # Freeze all
    param.requires_grad = False

# Now unfreeze SCAM layers
for m in model.model.modules():
    if isinstance(m, SCAM):
        for param in m.parameters():
            param.requires_grad = True

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable_params}")
# Print trainable parameters
for name, param in model.model.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name}")


# Load the pre-trained weights
model.train(
    data='ultralytics/datasets/pig_face/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='scam_finetune_pigface',
    project='runs/train'
)
