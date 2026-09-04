#!/usr/bin/env python3
"""Export a confidence-preserving YOLO11s cache from UAVSwarmV2 MOT-test images.

The exporter reads only ``<dataset-root>/test/<sequence>/img1`` and the frozen
checkpoint. It never loads GT. Each output ``det.txt`` uses original-image MOT
coordinates: ``frame,-1,x,y,w,h,score,-1,-1,-1``.
"""

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--sequences", nargs="+", metavar="SEQUENCE")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_seqinfo(path):
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = raw.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    required = ("imDir", "seqLength", "imExt", "imWidth", "imHeight")
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"{path}: missing {', '.join(missing)}")
    return {
        "image_dir": values["imDir"],
        "length": int(values["seqLength"]),
        "extension": values["imExt"],
        "width": int(values["imWidth"]),
        "height": int(values["imHeight"]),
    }


def available_sequences(dataset_root):
    test_root = dataset_root / "test"
    sequences = [
        candidate.name
        for candidate in sorted(test_root.glob("UAVSwarm-*"))
        if (candidate / "seqinfo.ini").is_file() and (candidate / "img1").is_dir()
    ]
    if not sequences:
        raise FileNotFoundError(f"no UAVSwarm MOT-test sequences under {test_root}")
    return sequences


def sequence_images(dataset_root, sequence, info):
    image_dir = dataset_root / "test" / sequence / info["image_dir"]
    images = sorted(image_dir.glob(f"*{info['extension']}"))
    if len(images) != info["length"]:
        raise ValueError(f"{sequence}: expected {info['length']} images, found {len(images)}")
    expected = [f"{frame:06d}{info['extension']}" for frame in range(1, info["length"] + 1)]
    if [path.name for path in images] != expected:
        raise ValueError(f"{sequence}: image filenames must be contiguous six-digit frame IDs")
    return images


def normalize_box(x1, y1, x2, y2, width, height, source):
    values = (x1, y1, x2, y2)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{source}: non-finite prediction geometry")
    raw = (x1, y1, x2, y2)
    x1, x2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
    y1, y2 = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
    clipped = raw != (x1, y1, x2, y2)
    if x2 <= x1 or y2 <= y1:
        return None, clipped
    return (x1, y1, x2 - x1, y2 - y1), clipped


def write_sequence(output_dir, sequence, rows):
    sequence_dir = output_dir / sequence
    sequence_dir.mkdir(parents=True, exist_ok=False)
    target = sequence_dir / "det.txt"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=sequence_dir, delete=False) as handle:
        temporary = Path(handle.name)
        for frame, x, y, width, height, score in rows:
            handle.write(f"{frame},-1,{x:.6f},{y:.6f},{width:.6f},{height:.6f},{score:.8f},-1,-1,-1\n")
    os.replace(temporary, target)
    return target


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {args.checkpoint}")
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable; refusing an implicit CPU export")
        if str(torch.ones(1, device="mps").device) != "mps:0":
            raise RuntimeError("MPS probe did not select mps:0")
    if args.imgsz <= 0 or args.batch <= 0 or args.max_det <= 0:
        raise ValueError("imgsz, batch and max-det must be positive")
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        raise ValueError("conf and iou must be in [0, 1]")

    dataset_root = args.dataset_root.resolve()
    available = available_sequences(dataset_root)
    selected = available if args.sequences is None else list(dict.fromkeys(args.sequences))
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"unknown MOT-test sequences: {', '.join(unknown)}")

    args.output_dir.mkdir(parents=True)
    model = YOLO(str(args.checkpoint.resolve()))
    exported = {}
    sources = {"checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint)}}
    total_images = 0
    total_rows = 0
    for sequence in selected:
        seqinfo_path = dataset_root / "test" / sequence / "seqinfo.ini"
        info = read_seqinfo(seqinfo_path)
        images = sequence_images(dataset_root, sequence, info)
        predictions = model.predict(
            source=[str(path) for path in images],
            stream=True,
            device=args.device,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            rect=True,
            save=False,
            save_txt=False,
            verbose=False,
        )
        rows = []
        clipped_rows = 0
        discarded_rows = 0
        seen_frames = set()
        for result in predictions:
            frame = int(Path(result.path).stem)
            if frame < 1 or frame > info["length"] or frame in seen_frames:
                raise ValueError(f"{sequence}: invalid or repeated result frame {frame}")
            seen_frames.add(frame)
            if result.boxes is None or len(result.boxes) == 0:
                continue
            xyxy = result.boxes.xyxy.detach().cpu().tolist()
            scores = result.boxes.conf.detach().cpu().tolist()
            for (x1, y1, x2, y2), score in zip(xyxy, scores):
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"{sequence} frame {frame}: invalid confidence")
                geometry, clipped = normalize_box(x1, y1, x2, y2, info["width"], info["height"], f"{sequence} frame {frame}")
                if geometry is None:
                    discarded_rows += 1
                    continue
                if clipped:
                    clipped_rows += 1
                x, y, width, height = geometry
                rows.append((frame, x, y, width, height, score))
        if seen_frames != set(range(1, info["length"] + 1)):
            raise ValueError(f"{sequence}: prediction stream omitted frames")
        rows.sort(key=lambda row: (row[0], row[1], row[2], row[4], row[5]))
        det_path = write_sequence(args.output_dir, sequence, rows)
        exported[sequence] = {"images": len(images), "rows": len(rows), "clipped_rows": clipped_rows, "discarded_nonpositive_rows": discarded_rows, "det_sha256": sha256(det_path)}
        sources[f"seqinfo/{sequence}"] = {"path": str(seqinfo_path.resolve()), "sha256": sha256(seqinfo_path)}
        total_images += len(images)
        total_rows += len(rows)

    manifest = {
        "schema_version": 1,
        "purpose": "Confidence-preserving YOLO11s detector cache for post-hoc UAVSwarmV2 MOT-test visualization.",
        "protocol_boundary": "Reads only MOT-test images and the frozen checkpoint. GT is not read. The cache is not a tracker result or metric-tuning input.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint)},
        "inference": {"device": args.device, "imgsz": args.imgsz, "batch": args.batch, "conf": args.conf, "iou": args.iou, "max_det": args.max_det, "coordinate_space": "original_image_xywh"},
        "sources": sources,
        "totals": {"sequences": len(exported), "images": total_images, "rows": total_rows},
        "sequences": exported,
    }
    (args.output_dir / "cache_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), **manifest["totals"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
