# Pokemon Detection with YOLO

Real-time Pokemon detection using YOLO11s trained on a custom dataset.

## Classes Detected (17)
Abra, Aerodactyl, Alakazam, Arbok, Arcanine, Articuno, Beedrill, Bellsprout, Blastoise, Bulbasaur, Butterfree, Evee, Electabuzz, Flareon, Dragonite, Gastly, Charizard

## Setup

```bash
pip install ultralytics opencv-python pyyaml
```

## Usage

Run scripts in order:

```bash

# 2. Split data into train/validation (80/20)
python train_val_split.py

# 3. Generate YOLO config file
python yaml_config.py

# 4. Train the model
python train_model.py

# 5. Run live detection
python cam_detect.py
```

## Adding New Pokemon

1. Add labeled images to `small-pokemon-data/images/` and `small-pokemon-data/labels/`
2. Update `small-pokemon-data/classes.txt` with new class names
3. Update `fix_labels.py` with new Pokemon name patterns
4. Run the full pipeline starting from step 1

## Project Structure
```
├── small-pokemon-data/     # Source dataset
│   ├── images/
│   ├── labels/
│   └── classes.txt
├── data/                   # Split dataset (generated)
│   ├── train/
│   └── validation/
├── runs/                   # Training outputs (generated)
├── train_val_split.py      # Split data
├── yaml_config.py          # Generate config
├── train_model.py          # Train YOLO model
├── cam_detect.py           # Live webcam detection
└── data.yaml               # YOLO config (generated)
```

## Training Config
- Model: YOLO11s (or fine-tuned from previous checkpoint)
- Epochs: 40
- Image size: 480x480
- Train/Val split: 80/20

## Controls
- Press `q` to quit webcam detection
