##### 
## The first part, capture 300 images of Jeff (myself) using the laptop camera; 
import cv2
import os
import time

# Set the directory where the images will be saved
save_dir = "/mnt/TriangleAI_Presentation/TrainYoloWithImages"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize the camera (use 0 for the default camera)
cap = cv2.VideoCapture(1)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set the number of images to capture
num_images = 300
interval = 3  # seconds between captures

# Start capturing images
for i in range(num_images):
    ret, frame = cap.read()  # Capture frame
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Save the image
    img_name = os.path.join(save_dir, f"image_{i+1:03d}.png")
    cv2.imwrite(img_name, frame)
    print(f"Saved {img_name}")
    
    # Wait for 3 seconds before capturing the next image
    time.sleep(interval)

# Release the camera
cap.release()
cv2.destroyAllWindows()

print(f"Captured {num_images} images.")

