"""
Split a YOLO dataset into train/val/test with a fixed seed.

Usage:
    python split_dataset.py --images images/ --labels labels/ --out dataset/ \
        --ratios 0.8 0.1 0.1 --seed 42

Output layout:
    dataset/images/{train,val,test}/  dataset/labels/{train,val,test}/
"""
import argparse
import random
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Split YOLO dataset into train/val/test")
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        raise SystemExit("ratios must sum to 1.0")

    imgs = sorted([f for f in Path(args.images).glob("*.*")
                   if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
    random.Random(args.seed).shuffle(imgs)
    n = len(imgs)
    n_train = int(n * args.ratios[0])
    n_val = int(n * args.ratios[1])
    splits = {"train": imgs[:n_train],
              "val": imgs[n_train:n_train + n_val],
              "test": imgs[n_train + n_val:]}

    for split, files in splits.items():
        (Path(args.out) / "images" / split).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.copy(img, Path(args.out) / "images" / split / img.name)
            lbl = Path(args.labels) / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy(lbl, Path(args.out) / "labels" / split / lbl.name)
        print(f"{split}: {len(files)} images")


if __name__ == "__main__":
    main()
