#!/bin/bash

EXPORT_FOLDER_PATH=$1

mkdir -p ${EXPORT_FOLDER_PATH}/logs
cp $(ls -t logs/wsimnet_training* | head -n 5) ${EXPORT_FOLDER_PATH}/logs/

mkdir -p ${EXPORT_FOLDER_PATH}/network
cp embedding/wsimnet/models/network.py ${EXPORT_FOLDER_PATH}/network/wsimnet_network.py

mkdir -p ${EXPORT_FOLDER_PATH}/configs
for i in {1..5}; do
    cp embedding/wsimnet/f${i}_config.json ${EXPORT_FOLDER_PATH}/configs/wsimnet_f${i}_config.json
done