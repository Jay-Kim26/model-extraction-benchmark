#!/bin/bash
configs=(
    "configs/mnist_lenet_soft_maze_10k_seed0.yaml"
    "configs/mnist_lenet_soft_dfms_10k_seed0.yaml"
    "configs/mnist_lenet_soft_game_10k_seed0.yaml"
    "configs/mnist_lenet_soft_es_10k_seed0.yaml"
    "configs/mnist_lenet_soft_blackbox_ripper_10k_seed0.yaml"
    "configs/mnist_lenet_soft_swiftthief_10k_seed0.yaml"
)

for config in "${configs[@]}"; do
    echo "=========================================================="
    echo "Starting experiment: $config"
    echo "=========================================================="
    python -m mebench run --config "$config" --device cuda:0
    if [ $? -ne 0 ]; then
        echo "Error running $config. Continuing to next..."
    fi
done
