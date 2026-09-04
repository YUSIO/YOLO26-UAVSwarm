#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <UAVSwarmV2_MOT-root> <best.pt> <experiment-results-root>" >&2
  exit 2
fi

dataset_root=$(cd "$1" && pwd)
checkpoint=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
results_root=$(mkdir -p "$3" && cd "$3" && pwd)
run_name="${RUN_NAME:-run_005}"
run_dir="$results_root/$run_name"
python_bin="${PYTHON_BIN:-python}"

if [ -e "$run_dir" ]; then
  echo "refusing to overwrite run directory: $run_dir" >&2
  exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "source tree is dirty" >&2
  exit 1
fi
if ! python_bin=$(command -v "$python_bin"); then
  echo "PYTHON_BIN is not executable" >&2
  exit 1
fi
if ! "$python_bin" -c 'import torch, ultralytics' >/dev/null 2>&1; then
  echo "PyTorch or Ultralytics is unavailable in $python_bin" >&2
  exit 1
fi

mkdir -p "$run_dir"
commit=$(git rev-parse HEAD)
checkpoint_hash=$(shasum -a 256 "$checkpoint" | awk '{print $1}')
"$python_bin" - "$run_dir/manifest.yaml" "$dataset_root" "$checkpoint" "$checkpoint_hash" "$commit" <<'PY'
import sys
from pathlib import Path

path, dataset_root, checkpoint, checkpoint_hash, commit = sys.argv[1:]
Path(path).write_text(
    "schema_version: 1\n"
    "status: running\n"
    "execution: local_macos_mps\n"
    "purpose: confidence_preserving_test_detection_cache_for_visualization\n"
    "code:\n"
    "  repository: git@github.com:YUSIO/YOLO26-UAVSwarm.git\n"
    "  branch: exp/011-yolo11s-uavswarmv2\n"
    f"  commit: {commit}\n"
    "  working_tree: clean_before_execution\n"
    "dataset:\n"
    f"  root: {dataset_root}\n"
    "  split: official_MOT_test_images_only\n"
    "  gt_read: false\n"
    "model:\n"
    f"  checkpoint: {checkpoint}\n"
    f"  checkpoint_sha256: {checkpoint_hash}\n"
    "inference:\n"
    "  device: mps\n"
    "  imgsz: 640\n"
    "  batch: 16\n"
    "  confidence: 0.001\n"
    "  iou: 0.7\n"
    "  max_det: 300\n"
)
PY
printf '%s scripts/export_uavswarmv2_mot_test_detections.py --dataset-root %s --checkpoint %s --output-dir %s --device mps --imgsz 640 --batch 16 --conf 0.001 --iou 0.7 --max-det 300\n' "$python_bin" "$dataset_root" "$checkpoint" "$run_dir/detections" > "$run_dir/command.txt"
set +e
"$python_bin" scripts/export_uavswarmv2_mot_test_detections.py --dataset-root "$dataset_root" --checkpoint "$checkpoint" --output-dir "$run_dir/detections" --device mps --imgsz 640 --batch 16 --conf 0.001 --iou 0.7 --max-det 300 2>&1 | tee "$run_dir/combined.log"
exit_code=${PIPESTATUS[0]}
set -e
printf '%s\n' "$exit_code" > "$run_dir/exit_code.txt"
status=failed
if [ "$exit_code" -eq 0 ]; then status=completed; fi
sed -i '' "s/status: running/status: $status/" "$run_dir/manifest.yaml"
exit "$exit_code"
