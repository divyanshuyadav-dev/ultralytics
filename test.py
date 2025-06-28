from ultralytics import YOLO
from ultralytics.nn.modules.attention import SCAM
import csv

# model = YOLO("yolov8n.yaml")

# # Predict on one image

# # Load your trained model
# model = YOLO("runs/train/scam_finetune_pigface6/weights/best.pt")
model = YOLO("yolov8n.pt")

# # Predict on one image
# results = model("ultralytics/datasets/pig_face/valid/images/5_frame_0010_jpg.rf.69928a1153aaa0e00f360d89f55b7da5.jpg", save=True, conf=0.25)

# # Output will be saved in runs/detect/predict/

# # Predict on a directory of images
# model.predict(
#     source="ultralytics/datasets/pig_face/valid/images",
#     save=True,
#     imgsz=416,
#     conf=0.25,
#     name="pigface_test"
# )
# # Output will be saved in runs/detect/pigface_test/



# 2. Evaluate on dataset (val/test split defined in data.yaml)
# -----------------------------
metrics = model.val(data="ultralytics/datasets/pig_face/data.yaml")


# -----------------------------
# 2.5 Save results
# -----------------------------
csv_path = "runs/detect/val/eval_results.csv"

with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)

    # Write headline row
    writer.writerow(["metric", "value"])

    # Core results (mAP, precision, recall, fitness)
    for key, val in metrics.results_dict.items():
        writer.writerow([key, float(val)])

    # AP class mapping (class IDs that got mAP)
    writer.writerow([])
    writer.writerow(["AP class index", ", ".join(map(str, metrics.ap_class_index.tolist()))])

    # Class names
    writer.writerow([])
    writer.writerow(["Classes"])
    for cls_id, name in metrics.names.items():
        writer.writerow([cls_id, name])

    # Speeds
    writer.writerow([])
    writer.writerow(["Speed Type", "ms per image"])
    for k, v in metrics.speed.items():
        writer.writerow([k, round(v, 2)])
    print(f"results saved to {csv_path}")
