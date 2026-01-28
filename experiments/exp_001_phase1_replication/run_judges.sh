#!/bin/bash
# Run judges on all batches
# Usage: ./run_judges.sh [batch_pattern]
# Example: ./run_judges.sh "batch_00*"  (first 10 batches)
#          ./run_judges.sh              (all batches)

cd "$(dirname "$0")/judging"

PATTERN="${1:-batch_*}"
BATCHES=$(ls -d $PATTERN 2>/dev/null | sort)

echo "Running judges on: $BATCHES"
echo "---"

for batch in $BATCHES; do
    n_samples=$(ls "$batch/samples/" 2>/dev/null | wc -l)
    n_done=$(ls "$batch/judgments/" 2>/dev/null | wc -l)

    if [ "$n_done" -ge "$n_samples" ] && [ "$n_samples" -gt 0 ]; then
        echo "[$batch] Already done ($n_done/$n_samples)"
        continue
    fi

    echo "[$batch] Judging $n_samples samples..."
    cd "$batch"
    claude --agent clab:judge --model haiku --print "Judge all samples in samples/, write judgments to judgments/" 2>&1 | tee judge_output.txt
    cd ..
    echo "---"
done

echo "Done!"
