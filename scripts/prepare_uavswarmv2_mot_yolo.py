#!/usr/bin/env python3
"""Create a YOLO detector split from UAVSwarmV2 MOT-train only.

The official 48 MOT-test sequences are not read. Each MOT-train sequence is
split at floor(seqLength / 2) + 1. Supplied temporal-half GT files are used
when present; the 12 sequences without them derive the identical split from
their own gt.txt. The generated output is a local detector protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_info(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    if {"name", "seqLength", "imWidth", "imHeight", "imDir", "imExt"} - values.keys():
        raise ValueError(f"incomplete seqinfo: {path}")
    return values


def read_boxes(path: Path) -> dict[int, list[tuple[float, float, float, float]]]:
    result: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        fields = line.split(",")
        if len(fields) < 8:
            raise ValueError(f"short MOT row at {path}:{line_number}")
        if int(float(fields[6])) == 1 and int(float(fields[7])) == 1:
            result[int(fields[0])].append(tuple(float(value) for value in fields[2:6]))
    return result


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-name", default="yolo11s_mot_temporal")
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    train_root, test_root, output = root / "train", root / "test", root / args.output_name
    if not train_root.is_dir() or not test_root.is_dir():
        raise FileNotFoundError(f"expected train/ and test/ under {root}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite generated split: {output}")
    sequences = sorted(path for path in train_root.iterdir() if path.is_dir())
    if len(sequences) != 48:
        raise ValueError(f"expected 48 MOT-train sequences, found {len(sequences)}")

    output.mkdir()
    paths = {"train": [], "val": []}
    counts = {"images": {"train": 0, "val": 0}, "labels": {"train": 0, "val": 0}, "source_active_boxes": 0, "clipped_boxes": 0, "discarded_zero_area_boxes": 0}
    hashes: dict[str, str] = {}
    sequence_records = []
    for seq_root in sequences:
        info_path, gt_root = seq_root / "seqinfo.ini", seq_root / "gt"
        info = read_info(info_path)
        name, length = info["name"], int(info["seqLength"])
        if name != seq_root.name:
            raise ValueError(f"sequence name mismatch: {info_path}")
        width, height, split_point = int(info["imWidth"]), int(info["imHeight"]), length // 2 + 1
        full_gt, train_half, val_half = gt_root / "gt.txt", gt_root / "gt_train_half.txt", gt_root / "gt_val_half.txt"
        image_root = seq_root / info["imDir"]
        if not full_gt.is_file() or not image_root.is_dir():
            raise FileNotFoundError(f"missing MOT-train source for {name}")
        if train_half.is_file() and val_half.is_file():
            train_boxes, val_boxes = read_boxes(train_half), read_boxes(val_half)
            if any(frame < 1 or frame > split_point for frame in train_boxes) or any(frame < 1 or frame > length - split_point for frame in val_boxes):
                raise ValueError(f"invalid supplied half-file range in {name}")
            label_source = "supplied_gt_train_half_and_gt_val_half"
            hashes[f"train/{name}/gt/gt_train_half.txt"] = sha256(train_half)
            hashes[f"train/{name}/gt/gt_val_half.txt"] = sha256(val_half)
        else:
            all_boxes = read_boxes(full_gt)
            train_boxes = {frame: boxes for frame, boxes in all_boxes.items() if frame <= split_point}
            val_boxes = {frame - split_point: boxes for frame, boxes in all_boxes.items() if frame > split_point}
            label_source = "derived_from_gt_txt_by_identical_temporal_rule"
        hashes[f"train/{name}/gt/gt.txt"] = sha256(full_gt)
        hashes[f"train/{name}/seqinfo.ini"] = sha256(info_path)

        for frame in range(1, length + 1):
            split = "train" if frame <= split_point else "val"
            image = image_root / f"{frame:06d}{info['imExt']}"
            if not image.is_file():
                raise FileNotFoundError(f"missing image: {image}")
            source_boxes = train_boxes.get(frame, []) if split == "train" else val_boxes.get(frame - split_point, [])
            labels = []
            for x, y, box_width, box_height in source_boxes:
                counts["source_active_boxes"] += 1
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(float(width), x + box_width), min(float(height), y + box_height)
                if (x1, y1, x2, y2) != (x, y, x + box_width, y + box_height):
                    counts["clipped_boxes"] += 1
                if x2 <= x1 or y2 <= y1:
                    counts["discarded_zero_area_boxes"] += 1
                    continue
                cx, cy = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
                nw, nh = (x2 - x1) / width, (y2 - y1) / height
                if not all(0 < value <= 1 for value in (cx, cy, nw, nh)):
                    raise ValueError(f"invalid normalized box in {name}, frame {frame}")
                labels.append(f"0 {cx:.8f} {cy:.8f} {nw:.8f} {nh:.8f}")
            image_link = output / "images" / split / name / image.name
            label_path = output / "labels" / split / name / f"{image.stem}.txt"
            make_link(image, image_link)
            write(label_path, "\n".join(labels) + ("\n" if labels else ""))
            paths[split].append(image_link.relative_to(output).as_posix())
            counts["images"][split] += 1
            counts["labels"][split] += len(labels)
        sequence_records.append({"sequence": name, "seq_length": length, "train_original_frames": [1, split_point], "val_original_frames": [split_point + 1, length], "label_source": label_source})

    for split in ("train", "val"):
        write(output / f"{split}.txt", "\n".join(paths[split]) + "\n")
    data_yaml = output / "UAVSwarmV2-MOT-temporal.yaml"
    write(data_yaml, "# Generated by scripts/prepare_uavswarmv2_mot_yolo.py\ntrain: images/train\nval: images/val\nnames:\n  0: UAV\n")
    manifest = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "protocol": "MOT-train-only per-sequence temporal split; MOT-test excluded", "source": {"dataset_root": str(root), "mot_train_sequences": 48, "mot_test_used": False, "mot_test_modified": False, "files_sha256": hashes}, "split": {"rule": "train frames 1..floor(seqLength/2)+1; validation remaining frames", "sequences": sequence_records}, "counts": counts, "outputs": {"data_yaml_sha256": sha256(data_yaml), "train_txt_sha256": sha256(output / "train.txt"), "val_txt_sha256": sha256(output / "val.txt")}}
    write(output / "split_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write(output / "SPLIT_RULE.md", "# UAVSwarmV2 MOT-train detector split\n\nOnly the 48 MOT-train sequences contribute labels. Each sequence uses frames `1..floor(seqLength/2)+1` for training and later frames for validation. MOT-test is excluded. Thirty-six sequences use supplied half files; twelve derive the same split from their own `gt.txt`.\n")
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
