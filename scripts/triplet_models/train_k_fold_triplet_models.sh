#!/bin/bash

for i in {1..5}
do
    python -W ignore -m "embedding.wsimnet.train" -c embedding/wsimnet/f${i}_config.json
done
