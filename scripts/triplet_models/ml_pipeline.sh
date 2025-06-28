#!/bin/bash

VECTOR_SEARCH_URL_LIST_FOLDER_PATH=dataset_metadata

DATASET_LIST=(
    'dataset'
)

MATRIX_FIG_TITLES=(
    'URL and Script'
)

for i in "${!DATASET_LIST[@]}"; do
    dataset=${DATASET_LIST[$i]}
    matrix_fig_title=${MATRIX_FIG_TITLES[$i]}
    echo "Processing $dataset..."

    echo "Running triplet model config generation"
    python -m "tools.generator.triplet_model_configs" -s vec_search '' export/${dataset} export/models

    echo "Training k-fold triplet models"
    ./scripts/triplet_models/train_k_fold_triplet_models.sh

    echo "Running k-fold experiment pipeline"
    ./scripts/triplet_models/k_fold_exp_pipeline.sh vec_search '' ${VECTOR_SEARCH_URL_LIST_FOLDER_PATH} $dataset "${matrix_fig_title}"

    echo "Copying the reproduction files to export"
    ./scripts/exp_result/copy_reproduction_files.sh export
done
