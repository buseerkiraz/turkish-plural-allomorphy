#!/usr/bin/env bash
# Run the whole study end to end. Requires python3 and network access for stage 00.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p output logs

STAGES=(
  "00_fetch_data.py"
  "01_extract.py"
  "02_phase0_audit.py"
  "03_at_control.py"
  "04_circularity_audit.py"
  "05_plural_check.py"
  "06_features.py"
  "07_nonce_items.py"
  "08_alcove_benchmark.py"
  "09_alcove_turkish.py"
  "10_rulex_benchmark.py"
  "11_rulex_turkish.py"
  "12_alcove_sensitivity.py"
  "13_human_survey.py"
  "14_frequency.py"
  "15_token_training.py"
)

for s in "${STAGES[@]}"; do
  name="${s%.py}"
  echo ""
  echo "=================================================================="
  echo ">>> $s"
  echo "=================================================================="
  python3 "src/$s" 2>&1 | tee "logs/${name}.log"
done

echo ""
echo "=================================================================="
echo "DONE. Outputs in output/, full logs in logs/"
ls -la output/
