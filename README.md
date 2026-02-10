# Pokemon Detection with YOLO

Real-time Pokemon detection using YOLO11s trained on a custom dataset.

## Classes Detected
Abra, Aerodactyl, Alakazam, Arbok, Arcanine, Articuno, Beedrill, Bellsprout, Blastoise, Bulbasaur, Butterfree

## Setup

```bash
pip install ultralytics opencv-python pyyaml
```

## Usage

Run scripts in order:

```bash
# 1. Split data into train/validation (80/20)
python train_val_split.py

# 2. Generate YOLO config file
python yaml_config.py

# 3. Train the model
python train_model.py

# 4. Run live detection
python cam_detect.py
```

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
- Model: YOLO11s
- Epochs: 40
- Image size: 480x480
- Train/Val split: 80/20

## Controls
- Press `q` to quit webcam detection
