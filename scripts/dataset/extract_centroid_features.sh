#!/bin/bash

EXPORT_FOLDER_PATH=$1
VECTOR_SEARCH_URL_LIST_FOLDER_PATH=$2
DATASET_NAME=$3

python -m "tools.extractor.centroid_feature" ${EXPORT_FOLDER_PATH}/${DATASET_NAME}_f1 ${VECTOR_SEARCH_URL_LIST_FOLDER_PATH}/train/供應鏈網站蒐集_過往案例向量搜尋_f1.xlsx
python -m "tools.extractor.centroid_feature" ${EXPORT_FOLDER_PATH}/${DATASET_NAME}_f2 ${VECTOR_SEARCH_URL_LIST_FOLDER_PATH}/train/供應鏈網站蒐集_過往案例向量搜尋_f2.xlsx
python -m "tools.extractor.centroid_feature" ${EXPORT_FOLDER_PATH}/${DATASET_NAME}_f3 ${VECTOR_SEARCH_URL_LIST_FOLDER_PATH}/train/供應鏈網站蒐集_過往案例向量搜尋_f3.xlsx
python -m "tools.extractor.centroid_feature" ${EXPORT_FOLDER_PATH}/${DATASET_NAME}_f4 ${VECTOR_SEARCH_URL_LIST_FOLDER_PATH}/train/供應鏈網站蒐集_過往案例向量搜尋_f4.xlsx
python -m "tools.extractor.centroid_feature" ${EXPORT_FOLDER_PATH}/${DATASET_NAME}_f5 ${VECTOR_SEARCH_URL_LIST_FOLDER_PATH}/train/供應鏈網站蒐集_過往案例向量搜尋_f5.xlsx