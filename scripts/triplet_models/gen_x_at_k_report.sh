#!/bin/bash

VECTOR_SEARCH_URL_LIST_FOLDER_PATH=$1
EXPORT_FOLDER_PATH=$2

K_RANGE=10
URL_LIST_PATH=${VECTOR_SEARCH_URL_LIST_FOLDER_PATH}/train/供應鏈網站蒐集_過往案例向量搜尋
QUERY_RESULT_FOLDER=${EXPORT_FOLDER_PATH}/query_result
EXPORT_PATH=${EXPORT_FOLDER_PATH}/report
JSON_EXPORT_PATH=${EXPORT_PATH}/json
EXPORT_NAME=x_at_k_report.log

if [ ! -d $EXPORT_PATH ]; then
  mkdir -p $EXPORT_PATH;
fi

echo 'WSimNet' > ${EXPORT_PATH}/${EXPORT_NAME}
for i in {1..5}; do
  for k in $(seq 1 $K_RANGE); do
    echo "F$i@$k" >> ${EXPORT_PATH}/${EXPORT_NAME}
    python -m "tools.experiment.x_at_k" ${URL_LIST_PATH}_f${i}.xlsx ${QUERY_RESULT_FOLDER}/wsimnet_f${i}.npy ${JSON_EXPORT_PATH} -k ${k} >> ${EXPORT_PATH}/${EXPORT_NAME}
  done
done
