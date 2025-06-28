#!/bin/bash

EXP_PREFIX=$1
EXP_POSTFIX=$2

VECTOR_SEARCH_URL_LIST_FOLDER_PATH=$3
DATASET_NAME=$4

MATRIX_FIG_TITLE=$5

echo "Starting inference on k-fold website data..."
./scripts/triplet_models/inference_k_fold_website_data.sh $EXP_PREFIX "$EXP_POSTFIX" export/$DATASET_NAME export${EXP_POSTFIX}

# echo "Visualizing k-fold inferenced test results..."
# ./scripts/triplet_models/visualize_k_fold_inferenced_test_results.sh export/$DATASET_NAME export${EXP_POSTFIX}

echo "Visualizing k-fold inferenced overall results..."
./scripts/triplet_models/visualize_k_fold_inferenced_overall_results.sh export/$DATASET_NAME export${EXP_POSTFIX}

echo "Querying k-fold website data..."
./scripts/triplet_models/query_k_fold_website_data.sh ${VECTOR_SEARCH_URL_LIST_FOLDER_PATH} export${EXP_POSTFIX}

echo "Generating loss figure..."
./scripts/triplet_models/gen_loss_fig.sh export${EXP_POSTFIX} $EXP_PREFIX "$EXP_POSTFIX"

echo "Generating validation average recall figure..."
./scripts/triplet_models/gen_val_avg_recall_fig.sh export${EXP_POSTFIX} $EXP_PREFIX "$EXP_POSTFIX"

echo "Generating class similarity matrix..."
./scripts/triplet_models/gen_class_similarity_matrix.sh "$MATRIX_FIG_TITLE" export${EXP_POSTFIX}

echo "Generating x-at-k report..."
./scripts/triplet_models/gen_x_at_k_report.sh ${VECTOR_SEARCH_URL_LIST_FOLDER_PATH} export${EXP_POSTFIX}

echo "Generating X@K figure..."
./scripts/triplet_models/gen_metric_fig.sh export${EXP_POSTFIX}
