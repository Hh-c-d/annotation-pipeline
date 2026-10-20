"""
Quality check for a YOLO-format dataset. Writes a Markdown report.

Checks: image/label counts, missing labels, per-class box counts, box size
stats, empty label files, out-of-bounds / invalid boxes.

Usage:
    python qc_report.py --dataset dataset/ --classes classes.txt --out qc_report.md

dataset/ is expected to contain images/ and labels/ subdirectories
(matching filenames: image foo.jpg <-> label foo.txt).
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description="YOLO dataset quality check")
    p.add_argument("--dataset", required=True, help="dataset root (has images/ and labels/)")
    p.add_argument("--classes", required=True, help="classes.txt, one per line")
    p.add_argument("--out", default="qc_report.md")
    args = p.parse_args()

    root = Path(args.dataset)
    classes = [ln.strip() for ln in Path(args.classes).read_text().splitlines() if ln.strip()]
    img_dir, lbl_dir = root / "images", root / "labels"
    images = sorted([f for f in img_dir.rglob("*.*")
                     if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])

    class_count = Counter()
    widths, heights, areas = [], [], []
    empty_files, missing_labels, bad_boxes = [], [], []

    for img in images:
        lbl = lbl_dir / img.relative_to(img_dir).with_suffix(".txt")
        if not lbl.exists():
            missing_labels.append(img.name)
            continue
        text = lbl.read_text().strip()
        if not text:
            empty_files.append(img.name)
            continue
        with Image.open(img) as im:
            iw, ih = im.size
        for line in text.splitlines():
            parts = line.split()
            if len(parts) != 5:
                bad_boxes.append((img.name, line))
                continue
            cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                bad_boxes.append((img.name, line))
                continue
            class_count[cls] += 1
            widths.append(bw * iw), heights.append(bh * ih), areas.append(bw * bh)

    def stats(vals):
        a = np.array(vals, dtype=float)
        return (f"min {a.min():.1f} / mean {a.mean():.1f} / max {a.max():.1f}") if a.size else "n/a"

    lines = [
        "# Dataset QC Report", "",
        f"- Images: {len(images)}",
        f"- Total boxes: {sum(class_count.values())}",
        f"- Missing label files: {len(missing_labels)}",
        f"- Empty label files: {len(empty_files)}",
        f"- Invalid boxes: {len(bad_boxes)}", "",
        "## Boxes per class", "",
    ]
    for i, name in enumerate(classes):
        lines.append(f"- {name}: {class_count.get(i, 0)}")
    lines += ["", "## Box size stats", "",
              f"- Width (px): {stats(widths)}",
              f"- Height (px): {stats(heights)}",
              f"- Area ratio (box/image): {stats(areas)}", ""]
    if missing_labels:
        lines += ["## Missing labels"] + [f"- {n}" for n in missing_labels[:50]] + [""]
    if empty_files:
        lines += ["## Empty label files"] + [f"- {n}" for n in empty_files[:50]] + [""]
    if bad_boxes:
        lines += ["## Invalid boxes"] + [f"- {n}: {t}" for n, t in bad_boxes[:50]] + [""]

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {args.out}")
    print(f"images={len(images)} boxes={sum(class_count.values())} "
          f"missing={len(missing_labels)} empty={len(empty_files)} invalid={len(bad_boxes)}")


if __name__ == "__main__":
    main()
