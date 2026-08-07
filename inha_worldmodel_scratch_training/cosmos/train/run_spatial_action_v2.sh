#!/usr/bin/env bash
# Conservative Cosmos champion extension: existing AdaLN + frame-level action tokens.
set -euo pipefail

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$TRAIN_DIR/../../.." && pwd)"
COSMOS_REPO="${COSMOS_REPO:-$WORKSPACE/third_party/cosmos-predict2.5}"
PYTHON="${COSMOS_PYTHON:-$COSMOS_REPO/.venv/bin/python}"
PHASE="${PHASE:-audit}"

export COSMOS_REPO
export COSMOS_CHECKPOINT_DIR="${COSMOS_CHECKPOINT_DIR:-$TRAIN_DIR/../checkpoints}"
export SO100_TRAIN_ROOT="${SO100_TRAIN_ROOT:-$WORKSPACE/open/data/train}"
export SO100_EVAL_ROOT="${SO100_EVAL_ROOT:-$WORKSPACE/open/data/eval}"
export SO100_INDEX_PATH="${SO100_INDEX_PATH:-$TRAIN_DIR/episode_index.parquet}"
export COSMOS_LATENT_DIR="${COSMOS_LATENT_DIR:-$TRAIN_DIR/latents}"
export COSMOS_VAL_LATENT_DIR="${COSMOS_VAL_LATENT_DIR:-$TRAIN_DIR/val_latents}"
export COSMOS_LATENT_MANIFEST="${COSMOS_LATENT_MANIFEST:-$TRAIN_DIR/latent_manifest.parquet}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CHAMPION="${CHAMPION:-$TRAIN_DIR/runs/v6_draft/step_095500.pt}"
RUN_NAME="${RUN_NAME:-cosmos_unified_action_v2}"
CANDIDATE="${CANDIDATE:-$TRAIN_DIR/runs/$RUN_NAME/latest.pt}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[missing] $1" >&2
    exit 2
  fi
}

require_assets() {
  require_file "$COSMOS_CHECKPOINT_DIR/robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt"
  require_file "$COSMOS_CHECKPOINT_DIR/robot/action-cond/cr1_empty_string_text_embeddings.pt"
  require_file "$COSMOS_CHECKPOINT_DIR/tokenizer.pth"
}

case "$PHASE" in
  prepare)
    PHASE=assets bash "$0"
    PHASE=index bash "$0"
    PHASE=latents NPROC="${NPROC:-8}" bash "$0"
    PHASE=audit bash "$0"
    PHASE=preflight bash "$0"
    ;;
  all)
    PHASE=prepare NPROC="${NPROC:-8}" bash "$0"
    PHASE=fit bash "$0"
    ;;
  fit)
    # One optimizer update validates the complete GPU path.  The following
    # invocation restores the same weights, optimizer, and sampler position.
    train_nproc="${TRAIN_NPROC:-8}"
    batch="${BATCH:-1}"
    effective_batch="${EFFECTIVE_BATCH:-8}"
    global_micro="$((train_nproc * batch))"
    if (( effective_batch < global_micro || effective_batch % global_micro != 0 )); then
      echo "EFFECTIVE_BATCH=$effective_batch must be divisible by TRAIN_NPROC*BATCH=$global_micro" >&2
      exit 2
    fi
    PHASE=unified_train TRAIN_NPROC="$train_nproc" BATCH="$batch" \
      EFFECTIVE_BATCH="$effective_batch" STOP_AT=1 bash "$0"
    PHASE=unified_train TRAIN_NPROC="$train_nproc" BATCH="$batch" \
      EFFECTIVE_BATCH="$effective_batch" bash "$0"
    ;;
  assets)
    hf_cli="${HF_CLI:-$COSMOS_REPO/.venv/bin/hf}"
    mkdir -p "$COSMOS_CHECKPOINT_DIR"
    "$hf_cli" download nvidia/Cosmos-Predict2.5-2B \
      robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt \
      robot/action-cond/cr1_empty_string_text_embeddings.pt \
      tokenizer.pth \
      --local-dir "$COSMOS_CHECKPOINT_DIR"
    ;;
  audit)
    "$PYTHON" "$TRAIN_DIR/audit_action_token_dit.py"
    "$PYTHON" "$TRAIN_DIR/audit_unified_losses.py"
    ;;
  preflight)
    require_assets
    "$PYTHON" "$TRAIN_DIR/audit_training_inputs.py"
    ;;
  index)
    "$PYTHON" "$TRAIN_DIR/build_index.py"
    ;;
  latents)
    require_assets
    nproc="${NPROC:-8}"
    export LATENT_BATCH="${LATENT_BATCH:-8}"
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$nproc" \
      "$TRAIN_DIR/precompute_latents.py"
    "$PYTHON" "$TRAIN_DIR/rebuild_latent_manifest.py"
    if [[ "${PRECOMPUTE_VAL:-1}" == 1 ]]; then
      COSMOS_LATENT_DIR="$COSMOS_VAL_LATENT_DIR" LATENT_SPLIT=val \
        "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$nproc" \
        "$TRAIN_DIR/precompute_latents.py"
    fi
    ;;
  train)
    require_assets
    require_file "$CHAMPION"
    export ATOK=1
    export ACTX_ONLY=1
    export RUN_NAME="${CHAMPION_RUN_NAME:-cosmos_spatial_action_v2}"
    export RESUME_FROM="$CHAMPION"
    export STOP_AT="${STOP_AT:-98500}"  # champion +3k screen; use 105500 only after it survives
    export SAVE_EVERY="${SAVE_EVERY:-1500}"
    # H100 80GB default.  Counterfactual steps retain two forward graphs, so
    # batch 2 is the safe starting point; batch 4 is a measured opt-in.
    export BATCH="${BATCH:-2}"
    export LR=0
    export LR_NEW="${LR_NEW:-5e-4}"
    export SAMPLER_SEED="${SAMPLER_SEED:-1234}"
    export CLEAN="${CLEAN:-1}"
    export MOTION_POW="${MOTION_POW:-1}"
    unset DRAFT AUX W_AUX P_AUX DELTA ABS_AUG ACT_DROP FULLFT
    cd "$TRAIN_DIR"
    "$PYTHON" train_lora.py
    ;;
  unified_train)
    require_assets
    "$PYTHON" "$TRAIN_DIR/audit_training_inputs.py"
    train_nproc="${TRAIN_NPROC:-8}"
    export ATOK=1
    export ACTX_ONLY=0
    export UNIFIED_V2=1
    export SCRATCH=1
    export RUN_NAME="${UNIFIED_RUN_NAME:-$RUN_NAME}"
    export BATCH="${BATCH:-1}"
    export EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-8}"
    global_micro="$((train_nproc * BATCH))"
    if (( EFFECTIVE_BATCH < global_micro || EFFECTIVE_BATCH % global_micro != 0 )); then
      echo "EFFECTIVE_BATCH=$EFFECTIVE_BATCH must be divisible by TRAIN_NPROC*BATCH=$global_micro" >&2
      exit 2
    fi
    # 104,538 positive windows.  Use the largest prefix divisible by the
    # effective batch so the final checkpoint always follows opt.step().
    usable_exposures="$((104538 / EFFECTIVE_BATCH * EFFECTIVE_BATCH))"
    default_stop="$((usable_exposures / global_micro))"
    export STOP_AT="${STOP_AT:-$default_stop}"
    save_exposures="${SAVE_EXPOSURES:-5000}"
    export SAVE_EVERY="${SAVE_EVERY:-$((save_exposures / global_micro))}"
    export LR="${LR:-5e-5}"
    export LR_NEW="${LR_NEW:-2e-4}"
    export UNIFIED_WARMUP="${UNIFIED_WARMUP:-10000}"
    export W_X0="${W_X0:-0.25}"
    export W_TEMPORAL="${W_TEMPORAL:-0.10}"
    export W_PRESERVE="${W_PRESERVE:-0.10}"
    export W_COUNTERFACTUAL="${W_COUNTERFACTUAL:-0.05}"
    export COUNTERFACTUAL_P="${COUNTERFACTUAL_P:-0.25}"
    export COUNTERFACTUAL_MARGIN="${COUNTERFACTUAL_MARGIN:-0.02}"
    export SAMPLER_SEED="${SAMPLER_SEED:-1234}"
    export NOREPL=1
    export CLEAN="${CLEAN:-1}"
    export MOTION_POW="${MOTION_POW:-0.5}"
    unset DRAFT AUX W_AUX P_AUX DELTA ABS_AUG ACT_DROP FULLFT RESUME_FROM
    cd "$TRAIN_DIR"
    if (( train_nproc > 1 )); then
      "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$train_nproc" \
        train_lora.py
    else
      "$PYTHON" train_lora.py
    fi
    ;;
  identity|generate)
    require_assets
    if [[ "$PHASE" == identity ]]; then
      require_file "$CHAMPION"
      export RUN_CKPT="$CHAMPION"
      export ATOK_GATE_SCALE=0
      default_out="$WORKSPACE/diagnostics/cosmos_spatial_action_v2_identity"
    else
      require_file "$CANDIDATE"
      export RUN_CKPT="$CANDIDATE"
      export ATOK_GATE_SCALE="${ATOK_GATE_SCALE:-1.0}"
      default_out="$WORKSPACE/diagnostics/${RUN_NAME}_gate${ATOK_GATE_SCALE}"
    fi
    export ATOK=1
    export STEPS="${STEPS:-30}"
    export SHIFT="${SHIFT:-5.0}"
    export AUTO_G="${AUTO_G:-0.6}"
    export SEED="${SEED:-7}"
    export LORA_SCALE=1.0
    export SUBSET="${SUBSET:-27}"  # 8-sample visual screen; set 0 for all 216
    export OUTDIR="${OUTDIR:-$default_out}"
    export OVERWRITE="${OVERWRITE:-0}"
    unset DELTA PERBLOCK FULLFT
    cd "$TRAIN_DIR"
    "$PYTHON" generate_eval.py
    ;;
  *)
    echo "Unknown PHASE=$PHASE (prepare, fit, all, assets, audit, preflight, index, latents, train, unified_train, identity, generate)" >&2
    exit 2
    ;;
esac
