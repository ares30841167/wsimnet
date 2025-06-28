#!/bin/bash

WEBSITE_TRFFIC_PATH=filtered_traffic
EXPORT_FOLDER_PATH=export
VECTOR_SEARCH_URL_LIST_FOLDER_PATH=dataset_metadata

DATASET_PREFIX_LIST=(
    'dataset'
)

for EXPORT_DATASET_PREFIX in "${DATASET_PREFIX_LIST[@]}"; do
    echo "Processing dataset..."

    python -m "tools.url_list.stratified" ${VECTOR_SEARCH_URL_LIST_FOLDER_PATH}/供應鏈網站蒐集_過往案例向量搜尋_DB.xlsx ${VECTOR_SEARCH_URL_LIST_FOLDER_PATH}/train 供應鏈網站蒐集_過往案例向量搜尋
    python create_dataset.py -ul 供應鏈網站蒐集_過往案例向量搜尋_DB.xlsx ${WEBSITE_TRFFIC_PATH} ${VECTOR_SEARCH_URL_LIST_FOLDER_PATH} ${EXPORT_FOLDER_PATH}

    ./scripts/dataset/create_stratified_datasets.sh ${EXPORT_FOLDER_PATH} ${VECTOR_SEARCH_URL_LIST_FOLDER_PATH} "${EXPORT_DATASET_PREFIX}"
    ./scripts/dataset/extract_centroid_features.sh ${EXPORT_FOLDER_PATH} ${VECTOR_SEARCH_URL_LIST_FOLDER_PATH} "${EXPORT_DATASET_PREFIX}"
done
