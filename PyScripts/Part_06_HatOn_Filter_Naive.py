import cv2
import numpy as np
from ultralytics import YOLO

model_jeff = YOLO("C:\\Users\\duj\\runs\\detect\\jeff_yolo_self_recognition5\\weights\\best.pt") 
#model_jeff = YOLO("C:\\Users\\duj\\runs\\detect\\jeff_yolo_self_recognition5\\weights\\last.pt") 
# Define a function to display the label
def draw_label(image, text, pos, color=(0, 255, 0), font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.6, thickness=2):
    cv2.putText(image, text, pos, font, scale, color, thickness, cv2.LINE_AA) 

# Load the hat image and resize it to an appropriate size 
# the quality of the input hat image matters a lot, the background in alpha channel 
# or a diferent background format will affect this function
hat_img = cv2.imread("C:\\Users\\\duj\\runs\\detect\\jeff_yolo_self_recognition5\\hat10.jpg", cv2.IMREAD_UNCHANGED)  
# Ensure hat image has an alpha channel (transparency) 

def remove_white_background(hat_img):
    # Convert hat image to grayscale
    gray = cv2.cvtColor(hat_img, cv2.COLOR_BGR2GRAY)

    # Create a binary mask where white (255) pixels are set to 0 (black) and non-white pixels are set to 1 (white)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Convert single channel mask to 3 channels (for blending purposes)
    mask_inv = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Extract the hat image (without the white background) using bitwise AND
    hat_no_bg = cv2.bitwise_and(hat_img, mask_inv)

    # Add an alpha channel (transparency) to the mask
    b, g, r = cv2.split(hat_no_bg)
    alpha = mask  # Use the mask as the alpha channel
    hat_with_alpha = cv2.merge([b, g, r, alpha])

    return hat_with_alpha

# Function to overlay hat on head with a manually created transparent background
def overlay_hat(frame, hat_img, x, y, w, h):
    # Resize the hat image based on the detected bounding box size (width and height)
    hat_width = w
    hat_height = int(hat_width * hat_img.shape[0] / hat_img.shape[1])  # Maintain aspect ratio
    resized_hat = cv2.resize(hat_img, (hat_width, hat_height), interpolation=cv2.INTER_AREA)

    # Split the resized hat image into its BGR and alpha channels
    if resized_hat.shape[2] == 4:  # Ensure the image has an alpha channel
        hat_bgr = resized_hat[:, :, :3]  # Color channels (BGR)
        hat_alpha = resized_hat[:, :, 3]  # Alpha channel (transparency)

    # Calculate the position to overlay the hat (place it slightly above the detected bounding box)
    y_offset = y - hat_height

    # Ensure the overlay doesn't go out of the frame
    if y_offset < 0:
        y_offset = 0

    # Check for boundaries to prevent the hat from going out of frame
    y1 = max(0, y_offset)
    y2 = min(frame.shape[0], y_offset + hat_height)
    x1 = max(0, x)
    x2 = min(frame.shape[1], x + hat_width)

    # Adjust the size of the resized hat to fit within the detected bounding box and frame boundaries
    hat_resized_height = y2 - y1
    hat_resized_width = x2 - x1

    if hat_resized_height <= 0 or hat_resized_width <= 0:
        return  # Skip if the region is out of frame

    # Resize the hat again to fit within the new bounds
    resized_hat = cv2.resize(resized_hat, (hat_resized_width, hat_resized_height))
    hat_bgr = resized_hat[:, :, :3]
    hat_alpha = resized_hat[:, :, 3]      #set the number to 2 if the background is dark 

    # Extract the region of interest (ROI) from the frame where the hat will be placed
    roi = frame[y1:y2, x1:x2]

    # Normalize the alpha mask to keep it between 0 and 1
    hat_alpha = hat_alpha / 255.0
    inv_alpha = 1.0 - hat_alpha

    # Blend the hat with the ROI using the alpha mask
    for c in range(0, 3):  # For each color channel (BGR)
        roi[:, :, c] = (hat_alpha * hat_bgr[:, :, c] + inv_alpha * roi[:, :, c])

    # Replace the region in the original frame with the blended result
    frame[y1:y2, x1:x2] = roi

# Remove white background from hat image
hat_img_transparent = remove_white_background(hat_img)

# Open video capture (0 for laptop camera)
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use the YOLO model to detect objects (yourself in this case) in the current frame
    results = model_jeff.predict(source=frame, conf=0.5, verbose=False)

    # Loop through each detection
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            if int(cls) == 0:  # Assuming you are class 0 (yourself)
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box)
                width = x2 - x1
                height = y2 - y1

                # Overlay the hat on the detected bounding box
                overlay_hat(frame, hat_img_transparent, x1, y1, width, height)

    # Display the frame with the hat overlay
    cv2.imshow('Hat On Filter Naive :) ', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
