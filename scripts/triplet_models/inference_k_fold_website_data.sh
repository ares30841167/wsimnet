#!/bin/bash

EXP_PREFIX=$1
EXP_POSTFIX=$2

DATASET_PATH=$3
EXPORT_FOLDER_PATH=$4

for i in {1..5}; do
    python -W ignore -m "embedding.wsimnet.inference_best" -d all ${DATASET_PATH}_f${i} ${EXPORT_FOLDER_PATH}/models/embedding/wsimnet/${EXP_PREFIX}_f${i}${EXP_POSTFIX} ${EXPORT_FOLDER_PATH}/embeddings wsimnet_f${i}_inference_result -m 35
done
