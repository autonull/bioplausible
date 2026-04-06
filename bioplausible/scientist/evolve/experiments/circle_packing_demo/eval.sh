#!/bin/bash
# =============================================================================
# Circle Packing Evaluation Script for Evolve Framework
# =============================================================================
# This script evaluates a circle packing solution for n=26 circles.
# Uses evaluator.py (aligned with openevolve) for actual evaluation.
#
# Expected file structure:
#   experiments/experiment_name/
#   |-- eval.sh           (this script)
#   |-- evaluator.py     (evaluation logic, aligned with openevolve)
#   `-- steps/
#       `-- step_N        (step results)
#
# Output:
#   step_N/results.json   (evaluation metrics)
# =============================================================================

set -e
set -o pipefail

# --- Path Configuration ---
STEP_DIR="$(pwd)"
if [[ "$STEP_DIR" == */steps/step_* ]]; then
    EXPERIMENT_DIR="$(dirname "$(dirname "$STEP_DIR")")"
else
    EXPERIMENT_DIR="$(dirname "$STEP_DIR")"
fi

SRC_CODE_FILE="${STEP_DIR}/code"
RESULT_JSON="${STEP_DIR}/results.json"
LOG_FILE="${STEP_DIR}/eval.log"
EVALUATOR_PY="${EXPERIMENT_DIR}/evaluator.py"

# --- Error Handling ---
handle_error() {
    local exit_code=$?
    echo "ERROR: Evaluation failed (Exit Code: $exit_code)" >&2
    cat > "$RESULT_JSON" << EOF
{
    "success": false,
    "eval_score": 0.0,
    "sum_radii": 0.0,
    "target_ratio": 0.0,
    "validity": 0.0,
    "eval_time": 0.0,
    "combined_score": 0.0,
    "complexity": 0.0,
    "temp": {
        "error": "Evaluation failed. See eval.log for details."
    }
}
EOF
    exit 0
}
trap 'if [ $? -ne 0 ]; then handle_error; fi' EXIT

# --- Evaluation ---
echo "=== Circle Packing Evaluation ===" > "$LOG_FILE"
echo "Step Directory: ${STEP_DIR}" >> "$LOG_FILE"
echo "Evaluator: ${EVALUATOR_PY}" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

if [ ! -f "$SRC_CODE_FILE" ]; then
    echo "ERROR: Source code file not found: ${SRC_CODE_FILE}" >> "$LOG_FILE"
    exit 1
fi

if [ ! -f "$EVALUATOR_PY" ]; then
    echo "ERROR: Evaluator script not found: ${EVALUATOR_PY}" >> "$LOG_FILE"
    exit 1
fi

echo "Running evaluation..." >> "$LOG_FILE"
python3 "$EVALUATOR_PY" "$SRC_CODE_FILE" "$RESULT_JSON" >> "$LOG_FILE" 2>&1

echo "Evaluation complete." >> "$LOG_FILE"

if [ -f "$RESULT_JSON" ]; then
    sum_radii=$(python3 -c "import json; print(json.load(open('$RESULT_JSON')).get('sum_radii', 0.0))")
    target_ratio=$(python3 -c "import json; print(json.load(open('$RESULT_JSON')).get('target_ratio', 0.0))")
    combined_score=$(python3 -c "import json; print(json.load(open('$RESULT_JSON')).get('combined_score', 0.0))")
    validity=$(python3 -c "import json; print(json.load(open('$RESULT_JSON')).get('validity', 0.0))")
    eval_time=$(python3 -c "import json; print(json.load(open('$RESULT_JSON')).get('eval_time', 0.0))")

    echo "Results:"
    echo "  Sum of radii: ${sum_radii}"
    echo "  Target ratio: ${target_ratio}"
    echo "  Combined score: ${combined_score}"
    echo "  Validity: ${validity}"
    echo "  Eval time: ${eval_time}s"
    echo "  Target (AlphaEvolve): 2.635"
fi

exit 0
