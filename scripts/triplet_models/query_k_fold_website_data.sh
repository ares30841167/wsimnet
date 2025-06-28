#!/bin/bash

VECTOR_SEARCH_URL_LIST_FOLDER_PATH=$1
EXPORT_FOLDER_PATH=$2

for i in {1..5}; do
    python -W ignore -m "tools.inquirer.embedding" ${VECTOR_SEARCH_URL_LIST_FOLDER_PATH}/train/供應鏈網站蒐集_過往案例向量搜尋_f${i}.xlsx ${EXPORT_FOLDER_PATH}/embeddings/wsimnet_f${i}_inference_result_overall.pkl ${EXPORT_FOLDER_PATH}/query_result wsimnet_f${i}
done
