#!/usr/bin/env python3
"""Create a read-only YOLO evaluation view from UAVSwarmV2 MOT-test.

This script is only for one-time detector evaluation after training.  It reads
official MOT-test GT to create YOLO labels, never changes MOT source files, and
does not contribute any input to detector training or checkpoint selection.
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


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_info(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    required = {"name", "seqLength", "imWidth", "imHeight", "imDir", "imExt"}
    if required - values.keys():
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


def make_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-name", default="yolo11s_mot_test_eval")
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    test_root, output = root / "test", root / args.output_name
    if not test_root.is_dir():
        raise FileNotFoundError(f"expected MOT-test directory: {test_root}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite generated evaluation view: {output}")
    sequences = sorted(path for path in test_root.iterdir() if path.is_dir())
    if len(sequences) != 48:
        raise ValueError(f"expected 48 MOT-test sequences, found {len(sequences)}")

    output.mkdir()
    paths: list[str] = []
    counts = {"images": 0, "labels": 0, "source_active_boxes": 0, "clipped_boxes": 0, "discarded_zero_area_boxes": 0}
    hashes: dict[str, str] = {}
    sequence_records = []
    for seq_root in sequences:
        info_path, gt_path = seq_root / "seqinfo.ini", seq_root / "gt" / "gt.txt"
        info = read_info(info_path)
        name, length = info["name"], int(info["seqLength"])
        if name != seq_root.name:
            raise ValueError(f"sequence name mismatch: {info_path}")
        width, height = int(info["imWidth"]), int(info["imHeight"])
        image_root = seq_root / info["imDir"]
        if not gt_path.is_file() or not image_root.is_dir():
            raise FileNotFoundError(f"missing MOT-test source for {name}")
        boxes_by_frame = read_boxes(gt_path)
        hashes[f"test/{name}/seqinfo.ini"] = sha256(info_path)
        hashes[f"test/{name}/gt/gt.txt"] = sha256(gt_path)

        for frame in range(1, length + 1):
            image = image_root / f"{frame:06d}{info['imExt']}"
            if not image.is_file():
                raise FileNotFoundError(f"missing image: {image}")
            labels = []
            for x, y, box_width, box_height in boxes_by_frame.get(frame, []):
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
            image_link = output / "images" / "test" / name / image.name
            label_path = output / "labels" / "test" / name / f"{image.stem}.txt"
            make_link(image, image_link)
            write(label_path, "\n".join(labels) + ("\n" if labels else ""))
            paths.append(image_link.relative_to(output).as_posix())
            counts["images"] += 1
            counts["labels"] += len(labels)
        sequence_records.append({"sequence": name, "seq_length": length})

    write(output / "test.txt", "\n".join(paths) + "\n")
    data_yaml = output / "UAVSwarmV2-MOT-test.yaml"
    write(data_yaml, "# Generated by scripts/prepare_uavswarmv2_mot_test_yolo.py\ntrain: images/test\nval: images/test\ntest: images/test\nnames:\n  0: UAV\n")
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "official MOT-test detector evaluation only; test GT never contributes to training or checkpoint selection",
        "source": {"dataset_root": str(root), "mot_test_sequences": 48, "mot_test_used": True, "mot_test_modified": False, "files_sha256": hashes},
        "sequences": sequence_records,
        "counts": counts,
        "outputs": {"data_yaml_sha256": sha256(data_yaml), "test_txt_sha256": sha256(output / "test.txt")},
    }
    write(output / "test_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write(output / "PROTOCOL.md", "# UAVSwarmV2 MOT-test detector evaluation\n\nThis generated view is evaluation-only. It symlinks official MOT-test images and converts MOT-test `gt.txt` into one-class YOLO labels with source geometry clipping. It never modifies test source files and is forbidden as detector training or checkpoint-selection input.\n")
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
