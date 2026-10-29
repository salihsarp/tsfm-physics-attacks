#!/bin/bash
# TSFM adversarial-robustness study — one-command run on an Apple-Silicon Mac (uses the Apple GPU via PyTorch MPS).
# Usage (in Terminal, from this folder):   bash run_all.sh
# Safe to re-run: finished stages are skipped (marker files results/.done_*). Keep the Mac plugged in, lid open.
set -o pipefail
cd "$(dirname "$0")"
mkdir -p results logs weights data
LOG="logs/run_all.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date '+%F %T') start on $(hostname) ($(sysctl -n machdep.cpu.brand_string 2>/dev/null))"

# ---------------------------------------------------------------- 1. Python environment (conda env "tsfm")
CONDA_BASE=""
for c in "$HOME/miniforge3" "$HOME/anaconda3" "$HOME/opt/anaconda3" "/opt/homebrew/Caskroom/miniforge/base"; do
  if [ -f "$c/etc/profile.d/conda.sh" ]; then CONDA_BASE="$c"; break; fi
done
if [ -z "$CONDA_BASE" ]; then echo "ERROR: conda (miniforge/anaconda) not found"; exit 1; fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
if ! conda env list | grep -qE '^tsfm[[:space:]]'; then conda create -y -n tsfm python=3.11 || exit 1; fi
conda activate tsfm || exit 1
if [ ! -f .deps_ok ]; then
  python -m pip install -q --upgrade pip
  python -m pip install -q "torch>=2.4" einops "accelerate<2" "transformers<6" pvlib scikit-learn h5py pandas \
      matplotlib requests huggingface_hub safetensors || exit 1
  python -m pip install -q --no-deps chronos-forecasting==2.3.2 timesfm==3.0.2 || exit 1
  touch .deps_ok
fi
python -c "import torch; print('torch', torch.__version__, '| Apple GPU (MPS) available:', torch.backends.mps.is_available())"
export PYTORCH_ENABLE_MPS_FALLBACK=1      # unsupported ops fall back to CPU instead of failing
export TSFM_THREADS=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 4)

# ---------------------------------------------------------------- 2. model weights (Hugging Face)
python - <<'PY' || exit 1
from huggingface_hub import snapshot_download
for repo, d in [("amazon/chronos-bolt-small", "chronos-bolt-small"), ("amazon/chronos-2", "chronos-2"),
                ("google/timesfm-2.5-200m-pytorch", "timesfm-2.5-200m-pytorch")]:
    snapshot_download(repo, local_dir=f"weights/{d}")
    print("weights ok:", repo)
PY

# ---------------------------------------------------------------- 3. data (ARPA-E PERFORM, ERCOT)
for D in solar load; do
  if [ ! -f "data/.download_$D" ]; then python data.py --download --domain $D && touch "data/.download_$D" || exit 1; fi
done

# ---------------------------------------------------------------- 4. experiments
# Code version of the experiments. A results/ folder from an older version is archived as results_<old>/
# (figures likewise); the LSTM weights are reused because the training windows do not change.
CURRENT=v3
if [ -f results/.done_prep_solar ] && [ "$(cat results/.version 2>/dev/null)" != "$CURRENT" ]; then
  OLD=$(cat results/.version 2>/dev/null || echo v2)
  [ -d results_v1 ] || OLD=v1
  mv results "results_$OLD" && mv figures "figures_$OLD" 2>/dev/null; mkdir -p results figures
  cp results_$OLD/*_lstm_seed*.pt results_$OLD/*_lstm_training.csv results/ 2>/dev/null
  for D in solar load; do [ -f results/${D}_lstm_seed2.pt ] && touch results/.done_lstm_$D; done
  echo "--- archived the previous run in results_$OLD/ (LSTM weights reused)"
fi
mkdir -p results; echo "$CURRENT" > results/.version
run_stage () {   # run_stage <tag> <args...>
  local tag="$1"; shift
  if [ -f "results/.done_$tag" ]; then echo "--- skip $tag (done)"; return 0; fi
  rm -f "results/.failed_$tag"
  echo "=== $(date '+%F %T') $tag"
  local t0=$(date +%s)
  if caffeinate -i python run_experiments.py "$@"; then
    touch "results/.done_$tag"; echo "--- $tag finished in $(( ($(date +%s) - t0) / 60 )) min"
  else
    echo "!!! $tag FAILED (see above); continuing with the next stage"; touch "results/.failed_$tag"
  fi
}
for D in solar load; do
  run_stage "prep_$D"     prep    --domain $D
  run_stage "verify_$D"   verify  --domain $D
  if [ -f "results/.failed_verify_$D" ]; then echo "STOP: wrapper check failed for $D — see logs/run_all.log"; exit 1; fi
  run_stage "lstm_$D"     lstm    --domain $D
  run_stage "clean_$D"    clean   --domain $D
  for M in chronos-bolt-small lstm chronos-2 timesfm-2.5; do
    run_stage "attack_${D}_$M" attack --domain $D --model $M
  done
  run_stage "transfer_$D" transfer --domain $D
  run_stage "cov_$D"      cov      --domain $D
  run_stage "detect_$D"   detect   --domain $D
  run_stage "defense_$D"  defense  --domain $D
  run_stage "steps_$D"    steps    --domain $D
  run_stage "window_$D"   window   --domain $D
  python make_figures.py --domain $D || echo "!!! figures for $D failed"
done
# ---------------------------------------------------------------- 5. paper tables and figures (95 % day-cluster bootstrap CIs)
python analyze.py . paper_results || echo "!!! analyze.py failed"
echo "=== $(date '+%F %T') ALL DONE. Results: $(pwd)/results  Figures: $(pwd)/figures"
ls results/.failed_* 2>/dev/null && echo "Some stages failed — see logs/run_all.log." || true
