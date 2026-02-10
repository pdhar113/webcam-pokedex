from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11s.pt")

    results = model.train(
        data="data.yaml",
        epochs=40,
        imgsz=480,
        batch=16,
        patience=10,
        save=True,
        project="pokemon_detection_runs",
        name="yolo11s_pokemon",
    )

    print("Training completed!")
    print(f"Model saved to: {results.save_dir}/weights/best.pt")
