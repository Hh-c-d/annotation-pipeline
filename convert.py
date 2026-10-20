"""
Annotation format conversion: COCO <-> YOLO, VOC -> YOLO.

COCO -> YOLO:
    python convert.py --format coco2yolo --input annotations.json \
        --images images/ --out labels/

YOLO -> COCO:
    python convert.py --format yolo2coco --labels labels/ --images images/ \
        --classes classes.txt --out annotations.json

VOC -> YOLO:
    python convert.py --format voc2yolo --input voc_annotations/ \
        --classes classes.txt --out labels/
"""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


def coco2yolo(ann_file, images_dir, out_dir):
    data = json.loads(Path(ann_file).read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(sorted(data["categories"], key=lambda c: c["id"]))}
    by_image = {}
    for a in data["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    for im in data["images"]:
        w, h = im["width"], im["height"]
        lines = []
        for a in by_image.get(im["id"], []):
            x, y, bw, bh = a["bbox"]  # COCO bbox = top-left corner + w/h (absolute)
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            lines.append(f"{cat_id_to_idx[a['category_id']]} {cx:.6f} {cy:.6f} {bw/w:.6f} {bh/h:.6f}")
        (out_dir / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines))
    print(f"coco2yolo done: {len(data['images'])} files -> {out_dir}")


def yolo2coco(labels_dir, images_dir, classes_file, out_file):
    classes = [ln.strip() for ln in Path(classes_file).read_text().splitlines() if ln.strip()]
    images, annotations = [], []
    ann_id = 1
    for img_id, img_path in enumerate(sorted(Path(images_dir).glob("*.*")), start=1):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
            annotations.append({
                "id": ann_id, "image_id": img_id,
                "category_id": cls + 1,
                "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
                "area": bw * w * bh * h, "iscrowd": 0,
            })
            ann_id += 1
    out = {"images": images, "annotations": annotations,
           "categories": [{"id": i + 1, "name": n} for i, n in enumerate(classes)]}
    Path(out_file).write_text(json.dumps(out), encoding="utf-8")
    print(f"yolo2coco done: {len(images)} images, {len(annotations)} boxes -> {out_file}")


def voc2yolo(voc_dir, classes_file, out_dir):
    classes = [ln.strip() for ln in Path(classes_file).read_text().splitlines() if ln.strip()]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for xml_file in sorted(Path(voc_dir).glob("*.xml")):
        root = ET.parse(xml_file).getroot()
        w = int(root.findtext("size/width"))
        h = int(root.findtext("size/height"))
        lines = []
        for obj in root.findall("object"):
            name = obj.findtext("name")
            if name not in classes:
                continue
            b = obj.find("bndbox")
            xmin, ymin = float(b.findtext("xmin")), float(b.findtext("ymin"))
            xmax, ymax = float(b.findtext("xmax")), float(b.findtext("ymax"))
            cx, cy = (xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h
            lines.append(f"{classes.index(name)} {cx:.6f} {cy:.6f} {(xmax-xmin)/w:.6f} {(ymax-ymin)/h:.6f}")
        (out_dir / (xml_file.stem + ".txt")).write_text("\n".join(lines))
    print(f"voc2yolo done -> {out_dir}")


def main():
    p = argparse.ArgumentParser(description="Annotation format conversion")
    p.add_argument("--format", required=True, choices=["coco2yolo", "yolo2coco", "voc2yolo"])
    p.add_argument("--input", help="coco2yolo: annotations.json; voc2yolo: dir of XML files")
    p.add_argument("--images", help="image directory (coco2yolo / yolo2coco)")
    p.add_argument("--labels", help="YOLO label directory (yolo2coco)")
    p.add_argument("--classes", help="classes.txt, one class per line (yolo2coco / voc2yolo)")
    p.add_argument("--out", required=True, help="output file or directory")
    args = p.parse_args()
    if args.format == "coco2yolo":
        coco2yolo(args.input, args.images, args.out)
    elif args.format == "yolo2coco":
        yolo2coco(args.labels, args.images, args.classes, args.out)
    else:
        voc2yolo(args.input, args.classes, args.out)


if __name__ == "__main__":
    main()
