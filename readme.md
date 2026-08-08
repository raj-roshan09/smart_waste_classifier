# Smart Waste Classifier

## ♻️ Project Overview

Smart Waste Classifier is an AI-based computer vision project designed to detect objects in an image and classify them based on their waste category.

The system combines object detection and image classification to identify waste items and determine whether they are **biodegradable, non-biodegradable, or not waste**.

It uses multiple deep learning models to improve object detection and classification, making it useful for applications related to **smart waste management and automated waste segregation**.

---

## 🎯 Objective

* Detect waste-related objects from images
* Classify detected objects into different waste categories
* Automate the process of waste identification
* Combine object detection and image classification techniques
* Demonstrate the use of AI in smart waste management

---

## 🧠 Algorithms Used

## YOLOv8 + Deep Learning Classification

The project uses **YOLOv8** for detecting objects within an image and a **deep learning image classification model** for determining the waste category.

Two YOLO models are used in the pipeline: a general-purpose YOLO model and a waste-specific YOLO model. The detected objects are cropped from the original image and passed to a trained classifier.

The classifier categorizes each detected object into one of three classes:

* **Biodegradable**
* **Non-Biodegradable**
* **Not Waste**

The system also removes duplicate detections and generates confidence scores for the final classifications.

---

## ⚙️ Features

* ♻️ Automatic waste detection from images
* 🔍 Object detection using YOLOv8
* 🧠 AI-based waste classification
* 🌱 Biodegradable waste identification
* 🗑️ Non-biodegradable waste identification
* 🚫 Detection of non-waste objects
* 🎯 Confidence score for predictions
* 📦 Multiple object detection in a single image
* 🔄 Combined detection and classification pipeline
* 🖼️ Annotated output images
* 📄 JSON-based prediction summary

---

## 🛠️ Technologies Used

* Python
* YOLOv8
* TensorFlow
* Keras
* OpenCV
* NumPy
* JSON

---

## 📂 Project Structure

```text
smart_waste_classifier/
│── Models/
│   ├── yolov8n.pt
│   ├── waste_yolov8.pt
│   └── waste_classifier.h5
│
│── integrated_pipeline.py
│── model1_detection.py
│── train_classifier.py
│── readme.md
│── outputs/
```

---

## 🔄 Working Process

```text
Input Image
     ↓
Object Detection
     ↓
Waste Object Identification
     ↓
Object Cropping
     ↓
Image Classification
     ↓
Waste Category Prediction
     ↓
Confidence Score
     ↓
Annotated Image + JSON Summary
```

The integrated pipeline first detects potential objects using the waste-specific and general YOLO models. Relevant detections are then filtered and duplicate bounding boxes are removed.

Each detected object is cropped and passed to the trained image classifier, which predicts its category along with a confidence score. The system finally saves an annotated image and a JSON summary containing the prediction results.
