#!/bin/bash

EXPORT_FOLDER_PATH=$1
TOP_K=$2
MODE=$3

for i in {1..5}; do
    python -W ignore -m "tools.classifier.mes_vote" ${EXPORT_FOLDER_PATH}/query_result/wsimnet_f${i}.npy ${EXPORT_FOLDER_PATH}/voting_classification_report/${MODE}/top_${TOP_K} f${i}_voting_classification_report -k ${TOP_K} -m ${MODE}
done
