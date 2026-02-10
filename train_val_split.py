from pathlib import Path
import random
import shutil

data_path = "small-pokemon-data"
train_pct = 0.8

input_img_path = Path(data_path) / "images"
input_lbl_path = Path(data_path) / "labels"

train_img = Path("data/train/images")
train_lbl = Path("data/train/labels")
val_img = Path("data/validation/images")
val_lbl = Path("data/validation/labels")

for d in [train_img, train_lbl, val_img, val_lbl]:
    d.mkdir(parents=True, exist_ok=True)

images = list(input_img_path.glob("*"))
random.shuffle(images)

split = int(len(images) * train_pct)
train_images, val_images = images[:split], images[split:]

for img_list, img_dst, lbl_dst in [
    (train_images, train_img, train_lbl),
    (val_images, val_img, val_lbl),
]:
    for img in img_list:
        shutil.copy(img, img_dst / img.name)
        lbl = input_lbl_path / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy(lbl, lbl_dst / lbl.name)

print(f"Train: {len(train_images)}, Validation: {len(val_images)}")
