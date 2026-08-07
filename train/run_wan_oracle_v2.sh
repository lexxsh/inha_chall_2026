#!/usr/bin/env bash
# Final cheap oracle gate: exact-zero spatial adapter, frozen Wan, cross-clip ranking.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OUT="${OUT:-$REPO/open/baseline/outputs/wan_oracle_track_v2_250}"
export STRUCTURAL_ZERO=1
export ADAPTER_ONLY=1
export WRONG_CONTROL_MODE=cross_clip
export RANK_MARGIN="${RANK_MARGIN:-0.005}"
export RANK_WEIGHT="${RANK_WEIGHT:-1.0}"

exec bash "$REPO/train/run_wan_oracle_control.sh"
