#!/bin/bash

EXPORT_FOLDER_PATH=$1

EXP_PREFIX=$2
EXP_POSTFIX=$3

python -m "tools.drawer.loss_fig.lines_chart_loss_vs_epoch" ${EXPORT_FOLDER_PATH}/models/embedding ${EXP_PREFIX} "${EXP_POSTFIX}" 'WSimNet - Loss' ${EXPORT_FOLDER_PATH}/figures/loss_fig -m wsimnet