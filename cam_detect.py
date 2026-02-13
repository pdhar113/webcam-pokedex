from ultralytics import YOLO
import cv2

model = YOLO(
    "runs/detect/pokemon_detection_runs/yolo11s_pokemon_finetuned/weights/best.pt"
)
cap = cv2.VideoCapture(1)

print("Pokemon Detection Started - Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Horizontal flip (mirror)
    results = model(frame, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow("Pokemon Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
