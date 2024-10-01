import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model_jeff = YOLO("C:\\Users\\duj\\TriangleAI_Presentation\\runs\\detect\\jeff_yolo_self_recognition4\\weights\\best.pt")
# Open the webcam
cap = cv2.VideoCapture(1)

# Define a function to display the label
def draw_label(image, text, pos, color=(0, 255, 0), font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.6, thickness=2):
    cv2.putText(image, text, pos, font, scale, color, thickness, cv2.LINE_AA)

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
            # Draw rectangle around the person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
            # Draw label "somebody" next to the bounding box
            draw_label(frame, "Jeff.G", (x1, y1 - 10))

    # Display the frame
    cv2.imshow("YOLOv8 - Self Recognition", frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
