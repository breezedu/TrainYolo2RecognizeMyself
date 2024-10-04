
# YOLO Self Recognition Project

This project demonstrates how to train a YOLOv8 model to recognize yourself in real-time using Python, OpenCV, and the Ultralytics YOLOv8 framework.

## Steps Overview

### 1. Dataset Collection

Capture a set of images using your webcam that contains various pictures of yourself. You can use the following Python script to capture images every few seconds:

```python
save_dir = "captured_images"
cap = cv2.VideoCapture(0)
cv2.imwrite(f"{save_dir}/self_image_{i+1}.png", frame)
```

### 2. Image Labeling with LabelImg

Label the captured images using the **LabelImg** tool by drawing bounding boxes around yourself. Save the annotations in YOLO format (`.txt` files).

- Install LabelImg: `pip install labelImg`
- Start labeling your images.

### 3. Training YOLOv8 Model

Train a YOLOv8 model on your labeled dataset using the Ultralytics library.

- Install YOLOv8: `pip install ultralytics`
- Organize your dataset and train with:

```python
model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=50)
```

### 4. Real-Time Self-Recognition

After training, use the model to recognize yourself in real-time with your webcam:

```python
model = YOLO("best.pt")
cap = cv2.VideoCapture(0)
results = model.predict(source=frame)
cv2.imshow("Self Recognition", frame)
```
---

## Conclusion

This project guides you through capturing a dataset, labeling it, training a YOLOv8 model, and using it to recognize yourself in real-time video streams.
