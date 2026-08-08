import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# ------------------- CONFIG -------------------
MODEL1_PATH = "models/yolov8n.pt"          # General YOLO
MODEL2_PATH = "models/waste_yolov8.pt"     # Waste-specific YOLO
CLASSIFIER_PATH = "models/waste_classifier.h5"
SAVE_DIR = "outputs"

GENERAL_CONF = 0.35
WASTE_CONF = 0.15
IOU = 0.45
CROP_SIZE = (224, 224)
# ------------------------------------------------

print("🔄 Loading models...")
yolo_general = YOLO(MODEL1_PATH)
yolo_waste = YOLO(MODEL2_PATH)
classifier = load_model(CLASSIFIER_PATH)
print("✅ Models loaded successfully!\n")

class_map = {0: "biodegradable", 1: "non biodegradable", 2: "not waste"}

# Only keep these general YOLO classes as "waste-like"
RELEVANT_CLASSES = {
    "bottle", "can", "cup", "plastic", "box", "paper", "bag", "fork", "spoon"
}


def iou(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    xi1, yi1 = max(x1, a1), max(y1, b1)
    xi2, yi2 = min(x2, a2), min(y2, b2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - inter
    return inter / union if union > 0 else 0


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def center_distance(box1, box2):
    cx1, cy1 = box_center(box1)
    cx2, cy2 = box_center(box2)
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def deduplicate_boxes(detections, img_shape):
    final = []
    diag = np.sqrt(img_shape[0] ** 2 + img_shape[1] ** 2)
    dist_thresh = 0.1 * diag  # 10% of diagonal

    for det in detections:
        keep = True
        for f in final:
            if det[4] == f[4] and det[5] == f[5]:
                if iou(det[:4], f[:4]) > 0.3 or center_distance(det[:4], f[:4]) < dist_thresh:
                    keep = False
                    break
        if keep:
            final.append(det)
    return final


def predict_class(image_path):
    img = load_img(image_path, target_size=CROP_SIZE)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = classifier.predict(img, verbose=0)
    cls = np.argmax(preds)
    conf = float(np.max(preds))
    return class_map[cls], round(conf * 100, 2)


def detect_and_classify(image_path):
    os.makedirs(SAVE_DIR, exist_ok=True)
    image = cv2.imread(image_path)

    results_waste = yolo_waste(image_path, conf=WASTE_CONF, iou=IOU, verbose=False)[0]
    results_gen = yolo_general(image_path, conf=GENERAL_CONF, iou=IOU, verbose=False)[0]

    detections = []
    waste_boxes = []

    # --- Primary detections (waste YOLO) ---
    for i, box in enumerate(results_waste.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        label = results_waste.names[int(results_waste.boxes.cls[i])]
        waste_boxes.append((x1, y1, x2, y2))
        detections.append((x1, y1, x2, y2, label, "waste_yolo"))

    # --- Fallback detections (general YOLO) ---
    for i, box in enumerate(results_gen.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        label = results_gen.names[int(results_gen.boxes.cls[i])]
        if label not in RELEVANT_CLASSES:  # skip people, cars, etc.
            continue
        overlap = any(iou((x1, y1, x2, y2), wb) > 0.3 for wb in waste_boxes)
        if not overlap:
            detections.append((x1, y1, x2, y2, label, "general_yolo"))

    detections = deduplicate_boxes(detections, image.shape)

    summary = []
    annotated = image.copy()
    print("\n📸 Detected objects:")

    for i, (x1, y1, x2, y2, label, source) in enumerate(detections):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]
        crop_path = os.path.join(SAVE_DIR, f"object_{i+1}.jpg")
        cv2.imwrite(crop_path, crop)

        cls_label, conf = predict_class(crop_path)
        summary.append({
            "object": i + 1,
            "yolo_label": label,
            "source": source,
            "classifier_result": cls_label,
            "confidence": conf
        })

        color = (
            (0, 255, 0) if "bio" in cls_label
            else (0, 0, 255) if "non" in cls_label
            else (255, 255, 0)
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{label} → {cls_label} ({conf:.1f}%)",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
        print(f" - {label} ({source}) → {cls_label} ({conf}%)")

    annotated_path = os.path.join(SAVE_DIR, f"annotated_{os.path.basename(image_path)}")
    summary_path = os.path.join(SAVE_DIR, "summary_latest.json")

    cv2.imwrite(annotated_path, annotated)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n🖼 Annotated saved to: {annotated_path}")
    print(f"📄 Summary saved to: {summary_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python integrated_pipeline.py <image_path>")
    else:
        detect_and_classify(sys.argv[1])