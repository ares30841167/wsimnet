#!/bin/bash

EXPORT_FOLDER_PATH=export


echo "Evaluating WSimNet with MES..."

./scripts/eval/mes_classifier.sh ${EXPORT_FOLDER_PATH}

for i in {1..3}
do
    TOP_K=$i
    MODE="top_${TOP_K}"
    ./scripts/eval/mes_voting_classifier.sh ${EXPORT_FOLDER_PATH} ${TOP_K} count
    ./scripts/eval/mes_voting_classifier.sh ${EXPORT_FOLDER_PATH} ${TOP_K} weighted
done