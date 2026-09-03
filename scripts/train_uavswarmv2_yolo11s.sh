#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <UAVSwarmV2_MOT-root> <runs-root>" >&2
  exit 2
fi
dataset_root=$(cd "$1" && pwd)
runs_root=$(mkdir -p "$2" && cd "$2" && pwd)
data_yaml="$dataset_root/yolo11s_mot_temporal/UAVSwarmV2-MOT-temporal.yaml"
split_manifest="$dataset_root/yolo11s_mot_temporal/split_manifest.json"
run_name="${RUN_NAME:-exp011_yolo11s_uavswarmv2_mot_run_001}"
config_source="${RUN_CONFIG:-configs/uavswarmv2_yolo11s_mot_run_001.yaml}"
run_dir="$runs_root/$run_name"
network_turbo_enabled=false
if [ -f /etc/network_turbo ]; then
  source /etc/network_turbo
  network_turbo_enabled=true
fi
if [ ! -f "$data_yaml" ] || [ ! -f "$split_manifest" ]; then
  echo "missing generated MOT-train-only detector split" >&2
  exit 1
fi
if [ -e "$run_dir" ]; then
  echo "refusing to overwrite run directory: $run_dir" >&2
  exit 1
fi
if [ ! -f "$config_source" ]; then
  echo "missing config: $config_source" >&2
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
cp "$config_source" "$run_dir/config.yaml"
cp "$data_yaml" "$run_dir/UAVSwarmV2-MOT-temporal.yaml"
cp "$split_manifest" "$run_dir/split_manifest.json"
cp "$dataset_root/yolo11s_mot_temporal/SPLIT_RULE.md" "$run_dir/SPLIT_RULE.md"
commit=$(git rev-parse HEAD)
base=dad7bb4534c95021bc14969ab25d77b77c4efdc3
if ! git merge-base --is-ancestor "$base" HEAD; then
  echo "unexpected baseline ancestry" >&2
  exit 1
fi
tb_pid_file="$runs_root/tensorboard.pid"
tb_mode=started
if [ -s "$tb_pid_file" ] && kill -0 "$(cat "$tb_pid_file")" 2>/dev/null; then
  tb_mode=reused
else
  nohup python -m tensorboard.main --logdir "$runs_root" --host 0.0.0.0 --port 6006 >"$runs_root/tensorboard.log" 2>&1 &
  printf '%s\n' "$!" >"$tb_pid_file"
fi
split_hash=$(sha256sum "$split_manifest" | awk '{print $1}')
config_hash=$(sha256sum "$run_dir/config.yaml" | awk '{print $1}')
{
  printf '%s\n' 'schema_version: 1' 'status: running' 'code:'
  printf '  repository: %s\n' 'git@github.com:YUSIO/YOLO26-UAVSwarm.git'
  printf '  branch: %s\n' "$(git branch --show-current)"
  printf '  commit: %s\n' "$commit"
  printf '  base_commit: %s\n' "$base"
  printf '%s\n' '  working_tree: clean_remote_before_execution' 'dataset:'
  printf '  root: %s\n' "$dataset_root"
  printf '  data_yaml: %s\n' "$data_yaml"
  printf '  split_manifest_sha256: %s\n' "$split_hash"
  printf '%s\n' '  protocol: MOT_train_only_per_sequence_temporal_split' '  official_mot_test_used: false' '  official_mot_test_modified: false' 'model:' '  weights: yolo11s.pt' '  imgsz: 640' '  epochs: 150' '  early_stopping: disabled' 'config: config.yaml'
  printf 'config_sha256: %s\n' "$config_hash"
  printf '%s\n' 'tensorboard:'
  printf '  mode: %s\n' "$tb_mode"
  printf '  logdir: %s\n' "$runs_root"
  printf '%s\n' '  host: 0.0.0.0' '  port: 6006' 'network_turbo:'
  printf '  enabled: %s\n' "$network_turbo_enabled"
  printf '%s\n' '  source: /etc/network_turbo'
} >"$run_dir/manifest.yaml"
{
  printf '%s\n' 'if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi'
  printf 'yolo detect train model=yolo11s.pt data=%s epochs=150 patience=0 imgsz=640 batch=-1 device=0 workers=8 optimizer=auto seed=0 deterministic=True pretrained=True project=%s name=%s exist_ok=True\n' "$data_yaml" "$runs_root" "$run_name"
} >"$run_dir/command.txt"
python - <<'PY'
from ultralytics import settings
settings.update({"tensorboard": True})
PY
set +e
yolo detect train model=yolo11s.pt data="$data_yaml" epochs=150 patience=0 imgsz=640 batch=-1 device=0 workers=8 optimizer=auto seed=0 deterministic=True pretrained=True project="$runs_root" name="$run_name" exist_ok=True
exit_code=$?
set -e
printf '%s\n' "$exit_code" >"$run_dir/exit_code.txt"
exit "$exit_code"
