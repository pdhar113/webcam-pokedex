from ultralytics import YOLO

if __name__ == "__main__":
    # Load pretrained YOLO model
    model = YOLO("yolov8m.pt")

    # Train the model
    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,
        save=True,
        project="pokemon_detection_runs",
        name="yolov8m_pokemon",
    )

    print("Training completed!")
    print(f"Model saved to: {results.save_dir}/weights/best.pt")
