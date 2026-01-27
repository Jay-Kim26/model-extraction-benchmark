#!/bin/bash

# Find all configs in the matrix folder
configs=$(ls configs/matrix/*.yaml)

echo "Starting Experimental Matrix Execution..."
echo "Total experiments: $(echo "$configs" | wc -l)"

for config in $configs; do
    name=$(basename "$config" .yaml)
    
    # Check if a summary already exists to avoid redundant runs
    # We look for any directory starting with the run name in runs/
    if ls "runs/${name}"/*/seed_*/summary.json >/dev/null 2>&1; then
        echo "[SKIP] $name already completed."
        continue
    fi

    echo "=========================================================="
    echo "Running: $name"
    echo "=========================================================="
    
    python -m mebench run --config "$config" --device cuda:0
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] $name failed."
    fi
done

echo "Matrix execution complete."
python aggregate_matrix.py
