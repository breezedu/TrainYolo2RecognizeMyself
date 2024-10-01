from ultralytics import YOLO

# Load the YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")  # 'n' means 'nano' version, suitable for fast training

# Train the model on your custom dataset
model.train(
    data="/mnt/data.yaml",  # Path to the dataset YAML file
    epochs=50,                            # Number of training epochs
    imgsz=640,                            # Image size for training
    batch=16,                             # Batch size
    name="jeff_yolo_self_recognition"     # Project name
) 
