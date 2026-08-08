import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# ---------------- CONFIG ----------------
MODEL1_PATH = "models/yolov8n.pt"          # General YOLO (COCO)
MODEL2_PATH = "models/waste_yolov8.pt"     # Waste-specific YOLO
CLASSIFIER_PATH = "models/waste_classifier.h5"
SAVE_DIR = "outputs"

CONF = 0.25
IOU = 0.45
CROP_SIZE = (224, 224)
PAD = 10
# ----------------------------------------

print("🔄 Loading models...")
yolo_general = YOLO(MODEL1_PATH)
yolo_waste = YOLO(MODEL2_PATH)
classifier = load_model(CLASSIFIER_PATH)
print("✅ Models loaded successfully!\n")

class_map = {0: "biodegradable", 1: "non biodegradable", 2: "not waste"}

# ------------------------------------------------
def iou(box1, box2):
    """Compute Intersection over Union between two boxes"""
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    xi1, yi1 = max(x1, a1), max(y1, b1)
    xi2, yi2 = min(x2, a2), min(y2, b2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - inter_area
    return inter_area / union_area if union_area > 0 else 0

# ------------------------------------------------
def predict_class(image_path):
    """Use waste_classifier.h5 to predict class"""
    img = load_img(image_path, target_size=CROP_SIZE)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = classifier.predict(img, verbose=0)
    cls = np.argmax(preds)
    conf = float(np.max(preds))
    return class_map[cls], round(conf * 100, 2)

# ------------------------------------------------
def detect_and_classify(image_path):
    os.makedirs(SAVE_DIR, exist_ok=True)
    image = cv2.imread(image_path)

    # YOLO detections
    results_general = yolo_general(image_path, conf=CONF, iou=IOU, verbose=False)[0]
    results_waste = yolo_waste(image_path, conf=CONF, iou=IOU, verbose=False)[0]

    detections, waste_boxes = [], []

    # STEP 1: Collect waste model detections
    for i, box in enumerate(results_waste.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        label = results_waste.names[int(results_waste.boxes.cls[i])]
        waste_boxes.append((x1, y1, x2, y2))
        detections.append((x1, y1, x2, y2, label, "waste_yolo"))

    # STEP 2: Add general YOLO detections if not overlapping or if it's a vehicle
    for i, box in enumerate(results_general.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        label = results_general.names[int(results_general.boxes.cls[i])].lower()

        vehicle_keywords = ["car", "truck", "bus", "bike", "motorcycle", "van"]
        is_vehicle = any(v in label for v in vehicle_keywords)

        # keep vehicles or non-overlapping objects
        if is_vehicle or not any(iou((x1, y1, x2, y2), wb) > 0.85 for wb in waste_boxes):
            detections.append((x1, y1, x2, y2, label, "general_yolo"))

    # STEP 3: Remove waste detections overlapping vehicle boxes
    vehicle_boxes = [d[:4] for d in detections if d[5] == "general_yolo" and any(v in d[4] for v in ["car", "truck", "bus", "bike", "motorcycle", "van"])]
    filtered_detections = []
    for det in detections:
        box = det[:4]
        if det[5] == "waste_yolo" and any(iou(box, vb) > 0.6 for vb in vehicle_boxes):
            continue  # skip duplicate waste detection on vehicle
        filtered_detections.append(det)

    detections = filtered_detections

    # STEP 4: Classify and draw results
    summary, annotated = [], image.copy()
    print("\n📸 Detected objects:")

    for i, (x1, y1, x2, y2, label, source) in enumerate(detections):
        x1, y1 = max(0, x1 - PAD), max(0, y1 - PAD)
        x2, y2 = min(image.shape[1], x2 + PAD), min(image.shape[0], y2 + PAD)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]
        crop_path = os.path.join(SAVE_DIR, f"object_{i+1}.jpg")
        cv2.imwrite(crop_path, crop)

        cls_label, conf = predict_class(crop_path)

        # Heuristic for inorganic materials
        if any(word in label.lower() for word in ["glass", "metal", "plastic", "car", "truck"]):
            cls_label = "non biodegradable"

        summary.append({
            "object": i + 1,
            "yolo_label": label,
            "source": source,
            "classifier_result": cls_label,
            "confidence": conf
        })

        color = (0, 255, 0) if "bio" in cls_label else (0, 0, 255) if "non" in cls_label else (255, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{label} → {cls_label} ({conf:.1f}%)", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        print(f" - {label} ({source}) → {cls_label} ({conf}%)")

    # STEP 5: Save annotated image & summary
    annotated_path = os.path.join(SAVE_DIR, f"annotated_{os.path.basename(image_path)}")
    summary_path = os.path.join(SAVE_DIR, "summary_latest.json")

    cv2.imwrite(annotated_path, annotated)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n🖼️ Annotated saved to: {annotated_path}")
    print(f"📄 Summary saved to: {summary_path}")

# ------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python integrated_pipeline.py <image_path>")
    else:
        detect_and_classify(sys.argv[1])
