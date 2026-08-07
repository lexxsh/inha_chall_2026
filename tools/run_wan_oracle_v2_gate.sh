#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CHECKPOINT="${CHECKPOINT:-$REPO/open/baseline/outputs/wan_oracle_track_v2_250/step-250.safetensors}"
export LABEL="${LABEL:-wan_oracle_track_v2_250}"
export STRUCTURAL_ZERO=1
export ADAPTER_ONLY=1

exec bash "$REPO/tools/run_wan_oracle_gate.sh"
