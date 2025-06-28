#!/bin/bash

EXPORT_FOLDER_PATH=$1

python -m "tools.drawer.metric.x_vs_k" ${EXPORT_FOLDER_PATH}/report/json ${EXPORT_FOLDER_PATH}/figures/metric_fig -t wsimnet -n "WSimNet"