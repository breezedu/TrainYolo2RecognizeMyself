import cv2
from ultralytics import YOLO

# Load the trained YOLO model
#model_jeff = YOLO("C:\\Users\\duj\\runs\\detect\\jeff_yolo_self_recognition4\\weights\\best.pt")
model_jeff = YOLO("C:\\Users\\duj\\runs\\detect\\jeff_yolo_self_recognition5\\weights\\best.pt") 
#model_jeff = YOLO("C:\\Users\\duj\\runs\\detect\\jeff_yolo_self_recognition5\\weights\\last.pt") 

# Open the webcam
cap = cv2.VideoCapture(1)

# Define a function to display the label
def draw_label(image, text, pos, color=(0, 255, 0), font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.6, thickness=2):
    cv2.putText(image, text, pos, font, scale, color, thickness, cv2.LINE_AA) 


# Load the hat image and resize it to an appropriate size
hat_img = cv2.imread("C:\\Users\\\duj\\TriangleAI_Presentation\\Beamer_slides\\images\\Hat.png", cv2.IMREAD_UNCHANGED)  
# Ensure hat image has an alpha channel (transparency)

# Function to overlay hat on head
def overlay_hat(frame, hat_img, x, y, w, h):
    # Resize the hat image based on the detected bounding box size
    hat_width = w
    hat_height = int(hat_width * hat_img.shape[0] / hat_img.shape[1])  # Maintain aspect ratio
    resized_hat = cv2.resize(hat_img, (hat_width, hat_height))

    # Calculate position for overlaying the hat (a bit above the bounding box)
    y_offset = y - hat_height

    # Ensure the hat doesn't go out of the frame
    if y_offset < 0:
        y_offset = 0

    # Extract the region of interest (ROI) from the frame where the hat will be placed
    roi = frame[y_offset:y_offset + hat_height, x:x + hat_width]

    # Split the hat image into 3 color channels and alpha channel (transparency)
    hat_rgb = resized_hat[:, :, :3]
    hat_alpha = resized_hat[:, :, 3] / 255.0

    # Blend the hat with the ROI using the alpha channel
    for c in range(0, 3):  # Iterate over the color channels
        roi[:, :, c] = (1 - hat_alpha) * roi[:, :, c] + hat_alpha * hat_rgb[:, :, c]

    # Replace the region in the original frame with the blended image
    frame[y_offset:y_offset + hat_height, x:x + hat_width] = roi


# Loop to capture frames from the webcam
while True:
    ret, frame = cap.read() 
    if not ret:
        break

    # Run inference on the frame
    results = model_jeff.predict(source=frame)

    # Loop through the detected results
    for result in results:
        # Loop through each detection (each bounding box)
        for box in result.boxes.xyxy:  # Extract the bounding box coordinates
            x1, y1, x2, y2 = map(int, box)  # Convert to integer coordinates
            width = x2 - x1
            height = y2 - y1
            # Draw rectangle around the person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle

            # Draw label "somebody" next to the bounding box
            draw_label(frame, "Jeff.G", (x1, y1 - 10))
            # Put the label of the person next to the rectangle
            #cv2.putText(frame, class_names[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            overlay_hat(frame, hat_img, x1, y1, width, height)
    
    # Display the frame
    cv2.imshow("YOLO11 - Hat On Filter", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
