#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <UAVSwarmV2_YOLO-root> <runs-root>" >&2
  exit 2
fi

dataset_root=$(cd "$1" && pwd)
runs_root=$(mkdir -p "$2" && cd "$2" && pwd)
data_yaml="$dataset_root/UAVSwarmV2-YOLO.yaml"
runtime_manifest="$dataset_root/runtime_manifest.json"
run_name="${RUN_NAME:-exp011_yolo11s_uavswarmv2_yolo_run_001}"
config_source="${RUN_CONFIG:-configs/uavswarmv2_yolo11s_run_001.yaml}"
run_dir="$runs_root/$run_name"
network_turbo_enabled=false
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; network_turbo_enabled=true; fi

if [ ! -f "$data_yaml" ] || [ ! -f "$runtime_manifest" ]; then echo "missing validated YOLO dataset metadata" >&2; exit 1; fi
if [ -e "$run_dir" ]; then echo "refusing to overwrite run directory: $run_dir" >&2; exit 1; fi
if [ ! -f "$config_source" ]; then echo "missing config: $config_source" >&2; exit 1; fi
if ! git diff --quiet || ! git diff --cached --quiet; then echo "remote source tree is dirty" >&2; exit 1; fi
if ! python -c 'import tensorboard' >/dev/null 2>&1; then echo "tensorboard unavailable" >&2; exit 1; fi

mkdir -p "$run_dir"
exec >>"$run_dir/combined.log" 2>&1
cp "$config_source" "$run_dir/config.yaml"
cp "$data_yaml" "$run_dir/UAVSwarmV2-YOLO.yaml"
cp "$runtime_manifest" "$run_dir/runtime_manifest.json"
commit=$(git rev-parse HEAD)
base=dad7bb4534c95021bc14969ab25d77b77c4efdc3
if ! git merge-base --is-ancestor "$base" HEAD; then echo "unexpected baseline ancestry" >&2; exit 1; fi
tb_pid_file="$runs_root/tensorboard.pid"; tb_mode=started
if [ -s "$tb_pid_file" ] && kill -0 "$(cat "$tb_pid_file")" 2>/dev/null; then tb_mode=reused; else nohup python -m tensorboard.main --logdir "$runs_root" --host 0.0.0.0 --port 6006 >"$runs_root/tensorboard.log" 2>&1 & printf '%s\n' "$!" >"$tb_pid_file"; fi
manifest_hash=$(sha256sum "$runtime_manifest" | awk '{print $1}')
config_hash=$(sha256sum "$run_dir/config.yaml" | awk '{print $1}')
cat >"$run_dir/manifest.yaml" <<EOF
schema_version: 1
status: running
code:
  repository: git@github.com:YUSIO/YOLO26-UAVSwarm.git
  branch: $(git branch --show-current)
  commit: $commit
  base_commit: $base
  working_tree: clean_remote_before_execution
dataset:
  root: $dataset_root
  data_yaml: $data_yaml
  runtime_manifest_sha256: $manifest_hash
  protocol: released_UAVSwarmV2_YOLO_package
  mot_overlap_warning: true
model:
  weights: yolo11s.pt
  imgsz: 640
  epochs: 150
  early_stopping: disabled
config: config.yaml
config_sha256: $config_hash
tensorboard:
  mode: $tb_mode
  logdir: $runs_root
  host: 0.0.0.0
  port: 6006
network_turbo:
  enabled: $network_turbo_enabled
  source: /etc/network_turbo
EOF
cat >"$run_dir/command.txt" <<EOF
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi
yolo detect train model=yolo11s.pt data=$data_yaml epochs=150 patience=0 imgsz=640 batch=-1 device=0 workers=8 optimizer=auto seed=0 deterministic=True pretrained=True project=$runs_root name=$run_name exist_ok=True
EOF
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
