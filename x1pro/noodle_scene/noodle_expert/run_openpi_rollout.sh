#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
WORKSPACE=$(cd "$ROOT/.." && pwd)
OUTPUT=${1:-"$ROOT/data/openpi_x1pro_rollout"}
STEPS=${2:-30}
TASK=${3:-place_noodles_in_pot}
PORT=${OPENPI_X1PRO_PORT:-8010}

# Keep the original workspace setup convenient, while allowing a standalone
# clone to use an explicit or currently activated RoboDojo Python environment.
if [[ -f "$WORKSPACE/deployment/robodojo/env.sh" ]]; then
  source "$WORKSPACE/deployment/robodojo/env.sh"
fi
if [[ -z ${ROBODOJO_PYTHON:-} ]]; then
  if [[ -n ${ROBODOJO_ENV:-} && -x "$ROBODOJO_ENV/bin/python" ]]; then
    ROBODOJO_PYTHON="$ROBODOJO_ENV/bin/python"
  else
    ROBODOJO_PYTHON=$(command -v python3 || command -v python)
  fi
fi
ROBODOJO_GPU=${ROBODOJO_GPU:-1}

"$ROBODOJO_PYTHON" "$ROOT/x1pro/noodle_scene/noodle_expert/collect.py" \
  --output "$OUTPUT" \
  --openpi-steps "$STEPS" \
  --openpi-task "$TASK" \
  --openpi-port "$PORT" \
  --headless --enable_cameras --device cuda:0 \
  --kit_args "--/renderer/activeGpu=$ROBODOJO_GPU --/renderer/multiGpu/enabled=false"
