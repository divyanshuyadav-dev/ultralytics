from ultralytics import YOLO

# # Load your trained model
model = YOLO("runs/train/<?>/weights/best.pt")

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
metrics = model.val(data="ultralytics/datasets/pig_face_abhi/data.yaml")

# Get parameter count and FLOPs
params = model.info(verbose=False)["params"]  # total parameters
flops = model.info(verbose=False)["flops"]    # GFLOPs

# Save metrics to CSV
import csv
csv_path = "runs/detect/val/eval_results.csv"
with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    for key, val in metrics.results_dict.items():
        writer.writerow([key, float(val)])
    writer.writerow(["params", params])
    writer.writerow(["flops(G)", flops])
print(f"results saved to {csv_path}")