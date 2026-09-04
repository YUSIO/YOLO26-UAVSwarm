#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <UAVSwarmV2_MOT-root> <best.pt> <runs-root>" >&2
  exit 2
fi

dataset_root=$(cd "$1" && pwd)
checkpoint=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
runs_root=$(mkdir -p "$3" && cd "$3" && pwd)
data_yaml="$dataset_root/yolo11s_mot_test_eval/UAVSwarmV2-MOT-test.yaml"
test_manifest="$dataset_root/yolo11s_mot_test_eval/test_manifest.json"
run_name="${RUN_NAME:-exp011_yolo11s_uavswarmv2_mot_test_run_002}"
run_dir="$runs_root/$run_name"
network_turbo_enabled=false
if [ -f /etc/network_turbo ]; then
  source /etc/network_turbo
  network_turbo_enabled=true
fi
if [ ! -f "$data_yaml" ] || [ ! -f "$test_manifest" ]; then
  echo "missing generated official MOT-test evaluation view" >&2
  exit 1
fi
if [ ! -f "$checkpoint" ]; then
  echo "missing checkpoint: $checkpoint" >&2
  exit 1
fi
if [ -e "$run_dir" ]; then
  echo "refusing to overwrite run directory: $run_dir" >&2
  exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "remote source tree is dirty" >&2
  exit 1
fi
if ! python -c 'import tensorboard' >/dev/null 2>&1; then
  echo "tensorboard unavailable" >&2
  exit 1
fi

mkdir -p "$run_dir"
exec >>"$run_dir/combined.log" 2>&1
cp "$data_yaml" "$run_dir/UAVSwarmV2-MOT-test.yaml"
cp "$test_manifest" "$run_dir/test_manifest.json"
cp "$dataset_root/yolo11s_mot_test_eval/PROTOCOL.md" "$run_dir/PROTOCOL.md"
commit=$(git rev-parse HEAD)
base=dad7bb4534c95021bc14969ab25d77b77c4efdc3
if ! git merge-base --is-ancestor "$base" HEAD; then
  echo "unexpected baseline ancestry" >&2
  exit 1
fi
test_hash=$(sha256sum "$test_manifest" | awk '{print $1}')
checkpoint_hash=$(sha256sum "$checkpoint" | awk '{print $1}')
tb_pid_file="$runs_root/tensorboard.pid"
tb_mode=started
if [ -s "$tb_pid_file" ] && kill -0 "$(cat "$tb_pid_file")" 2>/dev/null; then
  tb_mode=reused
else
  nohup python -m tensorboard.main --logdir "$runs_root" --host 0.0.0.0 --port 6006 >"$runs_root/tensorboard.log" 2>&1 &
  printf '%s\n' "$!" >"$tb_pid_file"
fi
{
  printf '%s\n' 'schema_version: 1' 'status: running' 'code:'
  printf '  repository: %s\n' 'git@github.com:YUSIO/YOLO26-UAVSwarm.git'
  printf '  branch: %s\n' "$(git branch --show-current)"
  printf '  commit: %s\n' "$commit"
  printf '  base_commit: %s\n' "$base"
  printf '%s\n' '  working_tree: clean_remote_before_execution' 'dataset:'
  printf '  root: %s\n' "$dataset_root"
  printf '  data_yaml: %s\n' "$data_yaml"
  printf '  test_manifest_sha256: %s\n' "$test_hash"
  printf '%s\n' '  protocol: official_MOT_test_detector_evaluation' '  official_mot_test_used: true' '  official_mot_test_modified: false' 'model:'
  printf '  checkpoint: %s\n' "$checkpoint"
  printf '  checkpoint_sha256: %s\n' "$checkpoint_hash"
  printf '%s\n' '  imgsz: 640' '  confidence: 0.001' '  iou: 0.7' '  max_det: 300' 'tensorboard:'
  printf '  mode: %s\n' "$tb_mode"
  printf '  logdir: %s\n' "$runs_root"
  printf '%s\n' '  host: 0.0.0.0' '  port: 6006' 'network_turbo:'
  printf '  enabled: %s\n' "$network_turbo_enabled"
  printf '%s\n' '  source: /etc/network_turbo'
} >"$run_dir/manifest.yaml"
printf 'yolo detect val model=%s data=%s split=test imgsz=640 batch=-1 device=0 workers=8 conf=0.001 iou=0.7 max_det=300 project=%s name=%s exist_ok=True\n' "$checkpoint" "$data_yaml" "$runs_root" "$run_name" >"$run_dir/command.txt"
python - <<'PY'
from ultralytics import settings
settings.update({"tensorboard": True})
PY
set +e
yolo detect val model="$checkpoint" data="$data_yaml" split=test imgsz=640 batch=-1 device=0 workers=8 conf=0.001 iou=0.7 max_det=300 project="$runs_root" name="$run_name" exist_ok=True
exit_code=$?
set -e
printf '%s\n' "$exit_code" >"$run_dir/exit_code.txt"
exit "$exit_code"
