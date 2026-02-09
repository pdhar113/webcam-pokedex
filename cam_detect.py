from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/pokemon_detection_runs/yolov8m_pokemon4/weights/best.pt")
cap = cv2.VideoCapture(1)

print("Pokemon Detection Started - Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.75, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow("Pokemon Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
