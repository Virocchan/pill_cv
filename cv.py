from ultralytics import YOLO
import cv2

model = YOLO('/Users/virocchan/Downloads/AIshowcase/pill.pt')

cap = cv2.VideoCapture(0)

print("Starting live detection. Press 'q' to quit.")

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Error: Could not detect frame from webcam")
        break

    results = model(frame, stream=True, conf=0.60)

    custom_colors = {
        0: (0, 0, 255)
    }

    for result in results:
        bounding_box = result.plot(
            conf=True,
            labels=True,
            boxes=True,
            line_width=2,
            font_size=1.5
        )

    cv2.imshow("YOLO Live Pill Detection", bounding_box)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()