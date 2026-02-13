from ultralytics import YOLO

if __name__ == "__main__":
    # prev model loaded
    model = YOLO("runs/detect/pokemon_detection_runs/yolo11s_pokemon/weights/best.pt")

    results = model.train(
        data="data.yaml",
        epochs=40,
        imgsz=480,
        batch=16,
        patience=10,
        save=True,
        project="pokemon_detection_runs",
        name="yolo11s_pokemon_finetuned",
    )

    print("Training completed!")
    print(f"Model saved to: {results.save_dir}/weights/best.pt")
