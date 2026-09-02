# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Export original-coordinate UAVSwarm detector outputs as an immutable MOTChallenge cache."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_images(sequence: Path) -> tuple[list[Path], int]:
    """Read and validate the ordered frame list declared by a UAVSwarm sequence."""
    config = configparser.ConfigParser()
    config.read(sequence / "seqinfo.ini")
    sequence_info = config["Sequence"]
    image_dir = sequence / sequence_info["imDir"]
    expected_length = int(sequence_info["seqLength"])
    images = sorted(image_dir.glob(f"*{sequence_info['imExt']}"))
    if len(images) != expected_length:
        msg = f"{sequence.name}: seqinfo expects {expected_length} images, found {len(images)}"
        raise RuntimeError(msg)
    frame_ids = [int(path.stem) for path in images]
    if frame_ids != list(range(1, expected_length + 1)):
        raise RuntimeError(f"{sequence.name}: image names are not consecutive 1..{expected_length}")
    return images, expected_length


def export_sequence(
    model: YOLO,
    sequence: Path,
    output_root: Path,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    batch: int,
    device: str,
) -> dict:
    """Run one sequence and atomically write its MOTChallenge detector file."""
    images, expected_length = sequence_images(sequence)
    sequence_output = output_root / sequence.name
    sequence_output.mkdir(parents=True, exist_ok=False)
    final_path = sequence_output / "det.txt"
    temporary_path = sequence_output / "det.txt.partial"
    frame_counts: Counter[int] = Counter()
    seen_frames: set[int] = set()

    predictions = model.predict(
        source=str(images[0].parent),
        stream=True,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        batch=batch,
        device=device,
        save=False,
        verbose=False,
    )
    with temporary_path.open("w", encoding="utf-8") as file:
        for result in predictions:
            frame_id = int(Path(result.path).stem)
            if frame_id in seen_frames or not 1 <= frame_id <= expected_length:
                raise RuntimeError(f"{sequence.name}: unexpected inference output {result.path}")
            seen_frames.add(frame_id)
            boxes = result.boxes
            if boxes is None:
                continue
            for xyxy, score in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist()):
                left, top, right, bottom = xyxy
                width, height = right - left, bottom - top
                if width <= 0 or height <= 0:
                    raise RuntimeError(f"{sequence.name} frame {frame_id}: non-positive detector box")
                file.write(
                    f"{frame_id},-1,{left:.8f},{top:.8f},{width:.8f},{height:.8f},{score:.8f},-1,-1,-1\n"
                )
                frame_counts[frame_id] += 1

    if seen_frames != set(range(1, expected_length + 1)):
        missing = sorted(set(range(1, expected_length + 1)) - seen_frames)
        raise RuntimeError(f"{sequence.name}: inference skipped frames {missing[:10]}")
    temporary_path.replace(final_path)
    return {
        "sequence": sequence.name,
        "frames": expected_length,
        "detections": sum(frame_counts.values()),
        "empty_frames": expected_length - len(frame_counts),
        "max_detections_in_frame": max(frame_counts.values(), default=0),
        "path": str(final_path.relative_to(output_root)),
        "sha256": sha256(final_path),
    }


def parse_args() -> argparse.Namespace:
    """Parse immutable cache-export parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def main() -> None:
    """Export all official test sequences and write a cache manifest."""
    args = parse_args()
    test_root = args.dataset_root / "test"
    sequences = sorted(path for path in test_root.iterdir() if path.is_dir() and (path / "seqinfo.ini").is_file())
    if not sequences:
        raise RuntimeError(f"No UAVSwarm sequences under {test_root}")
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing cache output: {args.output_root}")
    args.output_root.mkdir(parents=True)

    model = YOLO(args.weights)
    summaries = [
        export_sequence(
            model, sequence, args.output_root, args.imgsz, args.conf, args.iou, args.max_det, args.batch, args.device
        )
        for sequence in sequences
    ]
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "format": {
            "name": "MOTChallenge detector input",
            "line": "frame,-1,x,y,w,h,score,-1,-1,-1",
            "coordinates": "post-NMS original-image top-left xywh",
            "confidence": "post-NMS detector confidence",
        },
        "code_commit": args.code_commit,
        "weights": {"path": str(args.weights.resolve()), "sha256": sha256(args.weights)},
        "dataset": {"root": str(args.dataset_root.resolve()), "split": "official test", "sequences": len(summaries)},
        "predict": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "batch": args.batch,
            "device": args.device,
        },
        "runtime": {
            "torch": torch.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "summary": {
            "frames": sum(item["frames"] for item in summaries),
            "detections": sum(item["detections"] for item in summaries),
            "empty_frames": sum(item["empty_frames"] for item in summaries),
        },
        "sequences": summaries,
    }
    with (args.output_root / "cache_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
