#!/bin/bash

export PYTHONPATH=`pwd`
echo $PYTHONPATH

source $1
#exp=$2
#gpu=$3
#ARGS=${@:4}

group_examples_by_query_flag=''
if [[ $group_examples_by_query = *"True"* ]]; then
    group_examples_by_query_flag="--group_examples_by_query"
fi
relation_only_flag=''
if [[ $relation_only = *"True"* ]]; then
    relation_only_flag="--relation_only"
fi
use_action_space_bucketing_flag=''
if [[ $use_action_space_bucketing = *"True"* ]]; then
    use_action_space_bucketing_flag='--use_action_space_bucketing'
fi


cmd="python experiments.py  \
    --dataset icews18 \
    --random_embed \
    --embed_dim $em \
    --max_batches $max_batches \
    --eval_every $eval_every \
    --dropout_input $dropout_input  \
    --few $few \
    $ARGS"

echo "Executing $cmd"

$cmd
