#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
WORKSPACE=$(cd "$ROOT/.." && pwd)
OUTPUT=${1:-"$ROOT/data/openpi_x1pro_rollout"}
STEPS=${2:-30}
PORT=${OPENPI_X1PRO_PORT:-8010}
source "$WORKSPACE/deployment/robodojo/env.sh"
ROBODOJO_GPU=${ROBODOJO_GPU:-1} "$ROBODOJO_ENV/bin/python" "$ROOT/x1pro/noodle_scene/noodle_expert/collect.py"   --output "$OUTPUT" --openpi-steps "$STEPS" --openpi-port "$PORT"   --headless --enable_cameras --device cuda:0   --kit_args "--/renderer/activeGpu=$ROBODOJO_GPU --/renderer/multiGpu/enabled=false"
