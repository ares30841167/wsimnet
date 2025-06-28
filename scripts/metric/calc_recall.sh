#!/bin/bash

POSTFIX_LIST=(
    ''
)

TITLES=(
    'URL and Script'
)

for i in "${!POSTFIX_LIST[@]}"; do
    postfix=${POSTFIX_LIST[$i]}
    title=${TITLES[$i]}
    echo "title: $title"
    
    python -m "tools.calculator.overall_recall" export${postfix}/report/json -t wsimnet
    echo ""
done
