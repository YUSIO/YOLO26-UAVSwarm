#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <UAVSwarmV2_MOT-root> <best.pt> <result-root>" >&2
  exit 2
fi

dataset_root=$(cd "$1" && pwd)
checkpoint=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
result_root=$(mkdir -p "$3" && cd "$3" && pwd)
data_yaml="$dataset_root/yolo11s_mot_test_eval/UAVSwarmV2-MOT-test.yaml"
test_manifest="$dataset_root/yolo11s_mot_test_eval/test_manifest.json"
run_name="${RUN_NAME:-run_004}"
run_dir="$result_root/$run_name"
python_bin="${PYTHON_BIN:-python}"
imgsz=640
batch=16
workers=0

for required in "$data_yaml" "$test_manifest" "$checkpoint"; do
  if [ ! -f "$required" ]; then
    echo "missing required input: $required" >&2
    exit 1
  fi
done
if [ -e "$run_dir" ]; then
  echo "refusing to overwrite run directory: $run_dir" >&2
  exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "source tree is dirty" >&2
  exit 1
fi
if ! python_bin=$(command -v "$python_bin"); then
  echo "PYTHON_BIN is not executable: $python_bin" >&2
  exit 1
fi
if [ -n "${YOLO_BIN:-}" ]; then
  yolo_bin="$YOLO_BIN"
else
  yolo_bin="$(dirname "$python_bin")/yolo"
fi
if ! "$python_bin" -c 'import ultralytics' >/dev/null 2>&1; then
  echo "Ultralytics is unavailable in PYTHON_BIN=$python_bin" >&2
  exit 1
fi
if [ ! -x "$yolo_bin" ]; then
  echo "Ultralytics CLI is unavailable at YOLO_BIN=$yolo_bin" >&2
  exit 1
fi

"$python_bin" - <<'PY'
import torch
if not torch.backends.mps.is_available():
    raise SystemExit("MPS is unavailable in this process; refusing CPU fallback for this run")
x = torch.ones(1, device="mps")
assert str(x.device) == "mps:0"
PY

mkdir -p "$run_dir"
cp "$data_yaml" "$run_dir/UAVSwarmV2-MOT-test.yaml"
cp "$test_manifest" "$run_dir/test_manifest.json"
cp "$dataset_root/yolo11s_mot_test_eval/PROTOCOL.md" "$run_dir/PROTOCOL.md"
commit=$(git rev-parse HEAD)
checkpoint_hash=$(shasum -a 256 "$checkpoint" | awk '{print $1}')
test_hash=$(shasum -a 256 "$test_manifest" | awk '{print $1}')
"$python_bin" - "$run_dir/manifest.yaml" "$dataset_root" "$data_yaml" "$checkpoint" "$checkpoint_hash" "$test_hash" "$commit" <<'PY'
import platform
import sys
from pathlib import Path
import torch

path, dataset_root, data_yaml, checkpoint, checkpoint_hash, test_hash, commit = sys.argv[1:]
Path(path).write_text(
    "schema_version: 1\n"
    "status: running\n"
    "execution: local_macos_mps\n"
    "code:\n"
    "  repository: git@github.com:YUSIO/YOLO26-UAVSwarm.git\n"
    "  branch: exp/011-yolo11s-uavswarmv2\n"
    f"  commit: {commit}\n"
    "  working_tree: clean_before_execution\n"
    "dataset:\n"
    f"  root: {dataset_root}\n"
    f"  data_yaml: {data_yaml}\n"
    "  protocol: official_MOT_test_detector_evaluation\n"
    "  official_mot_test_used: true\n"
    "  official_mot_test_modified: false\n"
    f"  test_manifest_sha256: {test_hash}\n"
    "model:\n"
    f"  checkpoint: {checkpoint}\n"
    f"  checkpoint_sha256: {checkpoint_hash}\n"
    "inference:\n"
    "  device: mps\n"
    "  imgsz: 640\n"
    "  batch: 16\n"
    "  workers: 0\n"
    "  confidence: 0.001\n"
    "  iou: 0.7\n"
    "  max_det: 300\n"
    "runtime:\n"
    f"  python: {platform.python_version()}\n"
    f"  python_executable: {sys.executable}\n"
    f"  torch: {torch.__version__}\n"
    f"  platform: {platform.platform()}\n"
)
PY
printf '%s detect val model=%s data=%s split=test imgsz=%s batch=%s device=mps workers=%s conf=0.001 iou=0.7 max_det=300 project=%s name=%s exist_ok=True\n' "$yolo_bin" "$checkpoint" "$data_yaml" "$imgsz" "$batch" "$workers" "$result_root" "$run_name" > "$run_dir/command.txt"

set +e
PYTORCH_ENABLE_MPS_FALLBACK=1 "$yolo_bin" detect val model="$checkpoint" data="$data_yaml" split=test imgsz="$imgsz" batch="$batch" device=mps workers="$workers" conf=0.001 iou=0.7 max_det=300 project="$result_root" name="$run_name" exist_ok=True 2>&1 | tee "$run_dir/combined.log"
exit_code=${PIPESTATUS[0]}
set -e
printf '%s\n' "$exit_code" > "$run_dir/exit_code.txt"

"$python_bin" - "$run_dir" "$exit_code" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
exit_code = int(sys.argv[2])
manifest = run_dir / "manifest.yaml"
status = "completed" if exit_code == 0 else "failed"
manifest.write_text(manifest.read_text().replace("status: running", f"status: {status}"))
if exit_code == 0:
    with (run_dir / "results.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("results.csv has no metric rows")
    fields = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    metrics = {key: float(rows[-1][key]) for key in fields}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
PY

exit "$exit_code"
