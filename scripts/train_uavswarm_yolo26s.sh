#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_uavswarm_yolo26s.sh /root/autodl-tmp/UAVSwarm-dataset-master /root/autodl-tmp/runs
#
# The dataset split is generated under <dataset-root>/yolo26.  The run directory
# is intentionally external to the clone so a fresh clone can reproduce code
# while checkpoints and TensorBoard event files remain persistent.

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <UAVSwarm-dataset-master> <runs-root>" >&2
  exit 2
fi

dataset_root=$(cd "$1" && pwd)
runs_root=$(mkdir -p "$2" && cd "$2" && pwd)
data_yaml="$dataset_root/yolo26/UAVSwarm-yolo26.yaml"
run_name="${RUN_NAME:-exp008_yolo26s_uavswarm_run_001}"
config_source="${RUN_CONFIG:-configs/uavswarm_yolo26s_run_001.yaml}"
run_dir="$runs_root/$run_name"
network_turbo_enabled=false
if [ -f /etc/network_turbo ]; then
  # AutoDL's documented accelerator covers GitHub/GitHub assets and Hugging Face.
  # It is enabled here so the pretrained YOLO26 weight download inherits the proxy.
  source /etc/network_turbo
  network_turbo_enabled=true
fi

if [ ! -f "$data_yaml" ]; then
  echo "missing generated dataset config: $data_yaml" >&2
  echo "run: python scripts/prepare_uavswarm_yolo26.py --dataset-root $dataset_root" >&2
  exit 1
fi
if [ -e "$run_dir" ]; then
  echo "refusing to overwrite an existing run directory: $run_dir" >&2
  exit 1
fi
if [ ! -f "$config_source" ]; then
  echo "missing immutable run config: $config_source" >&2
  exit 1
fi

mkdir -p "$run_dir"
exec >>"$run_dir/combined.log" 2>&1
cp "$config_source" "$run_dir/config.yaml"
cp "$data_yaml" "$run_dir/UAVSwarm-yolo26.yaml"
cp "$dataset_root/yolo26/split_manifest.json" "$run_dir/split_manifest.json"
cp "$dataset_root/yolo26/SPLIT_RULE.md" "$run_dir/SPLIT_RULE.md"
code_commit=$(git rev-parse HEAD)
base_commit=dad7bb4534c95021bc14969ab25d77b77c4efdc3
if ! git merge-base --is-ancestor "$base_commit" HEAD; then
  echo "expected UAVSwarm experiment base $base_commit is not an ancestor of HEAD" >&2
  exit 1
fi
data_manifest_sha256=$(sha256sum "$dataset_root/yolo26/split_manifest.json" | awk '{print $1}')
config_sha256=$(sha256sum "$run_dir/config.yaml" | awk '{print $1}')
if ! python -c 'import tensorboard' >/dev/null 2>&1; then
  echo "tensorboard module is unavailable; install it in the training environment before submitting this run" >&2
  exit 1
fi
tensorboard_pid_file="$runs_root/tensorboard.pid"
tensorboard_mode=started
if [ -s "$tensorboard_pid_file" ] && kill -0 "$(cat "$tensorboard_pid_file")" 2>/dev/null; then
  tensorboard_pid=$(cat "$tensorboard_pid_file")
  tensorboard_mode=reused
else
  nohup python -m tensorboard.main --logdir "$runs_root" --host 0.0.0.0 --port 6006 >"$runs_root/tensorboard.log" 2>&1 &
  tensorboard_pid=$!
  printf '%s\n' "$tensorboard_pid" >"$tensorboard_pid_file"
fi
cat >"$run_dir/manifest.yaml" <<EOF
schema_version: 1
status: running
code:
  repository: https://github.com/YUSIO/YOLO26-UAVSwarm.git
  branch: $(git branch --show-current)
  commit: $code_commit
  base_commit: $base_commit
  working_tree: clean_required
dataset:
  root: $dataset_root
  data_yaml: $data_yaml
  split_manifest_sha256: $data_manifest_sha256
  official_test_used: false
model: yolo26s.pt
config: config.yaml
config_sha256: $config_sha256
tensorboard:
  mode: $tensorboard_mode
  logdir: $runs_root
  host: 0.0.0.0
  port: 6006
  pid_file: $runs_root/tensorboard.pid
network_turbo:
  enabled: $network_turbo_enabled
  source: /etc/network_turbo
EOF
cat >"$run_dir/command.txt" <<EOF
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi
yolo detect train model=yolo26s.pt data=$data_yaml epochs=300 patience=100 imgsz=1280 batch=-1 device=0 workers=8 optimizer=auto seed=0 deterministic=True pretrained=True project=$runs_root name=$run_name exist_ok=True
EOF
python - <<'PY'
from ultralytics import settings

settings.update({"tensorboard": True})
PY

set +e
yolo detect train \
  model=yolo26s.pt \
  data="$data_yaml" \
  epochs=300 \
  patience=100 \
  imgsz=1280 \
  batch=-1 \
  device=0 \
  workers=8 \
  optimizer=auto \
  seed=0 \
  deterministic=True \
  pretrained=True \
  project="$runs_root" \
  name="$run_name" \
  exist_ok=True
training_exit_code=$?
set -e
printf '%s\n' "$training_exit_code" >"$run_dir/exit_code.txt"
exit "$training_exit_code"
