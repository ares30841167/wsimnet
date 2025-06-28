#!/bin/bash

EXPORT_FOLDER_PATH=$1

for i in {1..5}; do
    python -W ignore -m "tools.classifier.mes" ${EXPORT_FOLDER_PATH}/query_result/wsimnet_f${i}.npy ${EXPORT_FOLDER_PATH}/classification_report f${i}_classification_report
done
