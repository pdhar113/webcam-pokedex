import yaml
import os

classes_file = "small-pokemon-data/classes.txt"
output_yaml = "data.yaml"

with open(classes_file, "r") as f:
    classes = [line.strip() for line in f if line.strip()]

data = {
    "path": os.path.abspath("data"),
    "train": "train/images",
    "val": "validation/images",
    "nc": len(classes),
    "names": classes,
}

with open(output_yaml, "w") as f:
    yaml.dump(data, f, sort_keys=False)

print(f"Created {output_yaml} with {len(classes)} classes")
