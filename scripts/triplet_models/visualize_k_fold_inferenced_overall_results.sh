#!/bin/bash

DATASET_PATH=$1
EXPORT_FOLDER_PATH=$2

for i in {1..5}; do
    python -W ignore -m "tools.inference.visualizer" -m 2d -f t-sne ${DATASET_PATH}_f${i} ${EXPORT_FOLDER_PATH}/embeddings/wsimnet_f${i}_inference_result_overall.pkl "Deep Sets (Attention) (Fold ${i})" ${EXPORT_FOLDER_PATH}/figures/overall_visualization wsimnet_f${i}_overall_tsne
done
