#!/usr/bin/env python3
"""Create a reproducible YOLO detection split from the original UAVSwarm train set.

The original UAVSwarm paper releases 36 train and 36 test sequences, but no
detector-validation split.  The dataset package includes MOTChallenge-style
``gt_train_half.txt`` and ``gt_val_half.txt`` files for every training
sequence.  This program reproduces their temporal convention without touching
the official test directory:

* train: original frames ``1 .. floor(seqLength / 2) + 1``;
* val: the remaining original frames (the supplied ``gt_val_half.txt``
  renumbers this suffix from one, so it cannot be used directly as filenames);
* test: excluded entirely.

It builds ``<dataset-root>/yolo26`` with symlinked images, YOLO labels,
train/validation lists, a data YAML, and a split manifest.  Input boxes are
clipped to image bounds before YOLO normalisation; boxes with no remaining area
are excluded and counted in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_lines(lines: list[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_seq_length(seqinfo: Path) -> int:
    values: dict[str, str] = {}
    for raw_line in seqinfo.read_text(encoding="utf-8").splitlines():
        if "=" in raw_line:
            key, value = raw_line.split("=", 1)
            values[key.strip()] = value.strip()
    try:
        return int(values["seqLength"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"missing valid seqLength in {seqinfo}") from exc


def count_mot_frames(path: Path) -> tuple[int, int, int]:
    """Return minimum frame, maximum frame and number of unique frame IDs."""
    frames = {int(line.split(",", 1)[0]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not frames:
        raise ValueError(f"no MOT rows in {path}")
    return min(frames), max(frames), len(frames)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def create_relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    relative_source = os.path.relpath(source, destination.parent)
    destination.symlink_to(relative_source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="UAVSwarm-dataset-master directory containing annotations/, train/ and test/.",
    )
    parser.add_argument(
        "--output-name",
        default="yolo26",
        help="Name of the generated directory under --dataset-root (default: yolo26).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    annotations_path = dataset_root / "annotations" / "train.json"
    train_root = dataset_root / "train"
    output_root = dataset_root / args.output_name

    if not annotations_path.is_file() or not train_root.is_dir():
        raise FileNotFoundError(
            f"expected annotations/train.json and train/ below dataset root: {dataset_root}"
        )
    if (dataset_root / "test").exists() and not (dataset_root / "test").is_dir():
        raise ValueError(f"test exists but is not a directory: {dataset_root / 'test'}")
    if output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing generated split: {output_root}. "
            "Choose another --output-name or remove it deliberately after preserving its manifest."
        )

    source = json.loads(annotations_path.read_text(encoding="utf-8"))
    images: list[dict[str, Any]] = source["images"]
    annotations: list[dict[str, Any]] = source["annotations"]
    videos: list[dict[str, Any]] = source["videos"]
    category_ids = {int(item["id"]) for item in source["categories"]}
    if category_ids != {1}:
        raise ValueError(f"expected one source category with id 1, found {sorted(category_ids)}")

    video_names = {int(video["id"]): str(video["file_name"]) for video in videos}
    image_by_id = {int(image["id"]): image for image in images}
    if len(image_by_id) != len(images):
        raise ValueError("duplicate image id in annotations/train.json")

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        if image_id not in image_by_id:
            raise ValueError(f"annotation references unknown image_id={image_id}")
        if int(annotation["category_id"]) != 1:
            raise ValueError(f"unexpected category_id in annotation id={annotation.get('id')}")
        annotations_by_image[image_id].append(annotation)

    image_counts_by_video: dict[str, int] = defaultdict(int)
    for image in images:
        image_counts_by_video[video_names[int(image["video_id"])]] += 1

    sequence_manifest: list[dict[str, Any]] = []
    split_by_image_id: dict[int, str] = {}
    seen_videos: set[str] = set()
    for video_id, sequence in sorted(video_names.items(), key=lambda item: item[1]):
        sequence_root = train_root / sequence
        seq_length = parse_seq_length(sequence_root / "seqinfo.ini")
        split_point = seq_length // 2 + 1
        train_half = sequence_root / "gt" / "gt_train_half.txt"
        val_half = sequence_root / "gt" / "gt_val_half.txt"
        train_min, train_max, train_unique = count_mot_frames(train_half)
        val_min, val_max, val_unique = count_mot_frames(val_half)
        expected_val_length = seq_length - split_point
        if (train_min, train_max, train_unique) != (1, split_point, split_point):
            raise ValueError(f"provided train-half convention differs in {train_half}")
        if (val_min, val_max, val_unique) != (1, expected_val_length, expected_val_length):
            raise ValueError(f"provided val-half convention differs in {val_half}")
        if image_counts_by_video[sequence] != seq_length:
            raise ValueError(
                f"annotation image count ({image_counts_by_video[sequence]}) differs from seqLength ({seq_length}) "
                f"for {sequence}"
            )

        sequence_images = sorted(
            (image for image in images if int(image["video_id"]) == video_id),
            key=lambda image: int(image["frame_id"]),
        )
        frame_ids = [int(image["frame_id"]) for image in sequence_images]
        if frame_ids != list(range(1, seq_length + 1)):
            raise ValueError(f"annotation frame IDs are not contiguous 1..seqLength for {sequence}")
        for image in sequence_images:
            split_by_image_id[int(image["id"])] = "train" if int(image["frame_id"]) <= split_point else "val"
        sequence_manifest.append(
            {
                "sequence": sequence,
                "seq_length": seq_length,
                "train_original_frames": [1, split_point],
                "val_original_frames": [split_point + 1, seq_length],
                "provided_gt_train_half_frames": train_unique,
                "provided_gt_val_half_frames": val_unique,
                "provided_gt_val_half_frame_id_mapping": "original_frame - split_point",
            }
        )
        seen_videos.add(sequence)

    if seen_videos != set(image_counts_by_video):
        raise ValueError("annotation videos and train sequence directories disagree")

    output_root.mkdir()
    train_paths: list[str] = []
    val_paths: list[str] = []
    annotations_clipped = 0
    annotations_discarded = 0
    labels_written = 0
    image_counts = {"train": 0, "val": 0}
    label_counts = {"train": 0, "val": 0}

    for image in sorted(images, key=lambda item: (str(item["file_name"]), int(item["id"]))):
        image_id = int(image["id"])
        split = split_by_image_id[image_id]
        source_image = train_root / str(image["file_name"])
        if not source_image.is_file():
            raise FileNotFoundError(f"annotated image is missing: {source_image}")
        sequence, _, filename = str(image["file_name"]).partition("/img1/")
        if not filename:
            raise ValueError(f"unexpected source image layout: {image['file_name']}")

        image_link = output_root / "images" / split / sequence / filename
        label_path = output_root / "labels" / split / sequence / f"{Path(filename).stem}.txt"
        create_relative_symlink(source_image, image_link)
        width, height = float(image["width"]), float(image["height"])
        rows: list[str] = []
        for annotation in annotations_by_image[image_id]:
            x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(width, x + box_width), min(height, y + box_height)
            if (x1, y1, x2, y2) != (x, y, x + box_width, y + box_height):
                annotations_clipped += 1
            if x2 <= x1 or y2 <= y1:
                annotations_discarded += 1
                continue
            normalized_width = (x2 - x1) / width
            normalized_height = (y2 - y1) / height
            center_x = (x1 + x2) / (2.0 * width)
            center_y = (y1 + y2) / (2.0 * height)
            if not all(0.0 < value <= 1.0 for value in (center_x, center_y, normalized_width, normalized_height)):
                raise ValueError(f"invalid normalized label for annotation id={annotation.get('id')}")
            rows.append(f"0 {center_x:.8f} {center_y:.8f} {normalized_width:.8f} {normalized_height:.8f}")
        write_text(label_path, "\n".join(rows) + ("\n" if rows else ""))
        labels_written += len(rows)
        label_counts[split] += len(rows)
        image_counts[split] += 1
        relative_image = image_link.relative_to(output_root).as_posix()
        if split == "train":
            train_paths.append(relative_image)
        else:
            val_paths.append(relative_image)

    write_text(output_root / "train.txt", "\n".join(train_paths) + "\n")
    write_text(output_root / "val.txt", "\n".join(val_paths) + "\n")
    write_text(
        output_root / "UAVSwarm-yolo26.yaml",
        "# Generated by scripts/prepare_uavswarm_yolo26.py; dataset root is this YAML's directory.\n"
        "train: train.txt\n"
        "val: val.txt\n"
        "names:\n"
        "  0: UAV\n",
    )
    source_annotation_hash = sha256_file(annotations_path)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/prepare_uavswarm_yolo26.py",
        "source": {
            "dataset_root": str(dataset_root),
            "annotations_train_json_sha256": source_annotation_hash,
            "official_test_directory": str(dataset_root / "test"),
            "official_test_used": False,
            "official_test_modified": False,
        },
        "split_rule": {
            "evidence": "dataset-supplied gt_train_half.txt and gt_val_half.txt for each official train sequence",
            "train": "per sequence, original frames 1 through floor(seqLength / 2) + 1 (inclusive)",
            "val": "per sequence, remaining original frames; validation suffix is not reindexed in YOLO filenames",
            "test": "official test split excluded from detector train/validation preparation",
        },
        "counts": {
            "official_train_sequences": len(sequence_manifest),
            "official_train_images": len(images),
            "train_images": image_counts["train"],
            "val_images": image_counts["val"],
            "source_annotations": len(annotations),
            "output_labels": labels_written,
            "train_labels": label_counts["train"],
            "val_labels": label_counts["val"],
            "clipped_source_boxes": annotations_clipped,
            "discarded_zero_area_boxes": annotations_discarded,
        },
        "artifacts": {
            "train_txt_sha256": sha256_text_lines(train_paths),
            "val_txt_sha256": sha256_text_lines(val_paths),
            "data_yaml": "UAVSwarm-yolo26.yaml",
        },
        "sequences": sequence_manifest,
    }
    write_text(output_root / "split_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_text(
        output_root / "SPLIT_RULE.md",
        "# UAVSwarm YOLO26 detector split\n\n"
        "This directory is generated from the original UAVSwarm official training split. The original paper defines "
        "36 training and 36 testing sequences; it does not publish a detector validation partition. The dataset package "
        "provides `gt_train_half.txt` and `gt_val_half.txt` for every training sequence, and this preparation exactly "
        "reproduces their temporal partition.\n\n"
        "For a sequence of length `L`, train receives original frames `1 .. floor(L / 2) + 1`; validation receives "
        "the remaining original frames. The supplied validation MOT file renumbers its suffix from 1, but this YOLO "
        "export deliberately retains original image filenames. The official `test/` directory is not read for labels, "
        "is not linked, and is never modified.\n\n"
        "Images are symlinked rather than copied. Labels have class `0` (`UAV`); source boxes are clipped to image "
        "bounds before normalization, and any zero-area remainder is discarded. Exact counts, source and split hashes, "
        "and every sequence boundary are in `split_manifest.json`.\n",
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    print(f"prepared: {output_root}")


if __name__ == "__main__":
    main()
