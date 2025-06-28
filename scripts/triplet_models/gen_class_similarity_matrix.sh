#!/bin/bash

FIG_TITLE=$1
EXPORT_FOLDER_PATH=$2

python -m "tools.aggregator.label_grouped_avg_scores" ${EXPORT_FOLDER_PATH}/query_result ${EXPORT_FOLDER_PATH}/avg_scores -t wsimnet
python -m "tools.drawer.matrix.class_similarity_paper" ${EXPORT_FOLDER_PATH}/avg_scores "WSimNet - ${FIG_TITLE}" ${EXPORT_FOLDER_PATH}/figures/matrix -t wsimnet
