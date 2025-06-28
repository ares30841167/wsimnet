#!/bin/bash

EXPORT_FOLDER_PATH=$1

EXP_PREFIX=$2
EXP_POSTFIX=$3

python -m "tools.drawer.val_avg_recall.lines_chart_avg_recall_vs_epoch" ${EXPORT_FOLDER_PATH}/models/embedding ${EXP_PREFIX} "${EXP_POSTFIX}" 'WSimNet - Val Average Recall' ${EXPORT_FOLDER_PATH}/figures/val_avg_recall -m wsimnet