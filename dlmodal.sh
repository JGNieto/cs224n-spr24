#!/bin/bash

# Usage: dlmodal.sh <experiment_name1> <experiment_name2> ... <experiment_nameN>

for i in "$@"
do
    echo "Downloading experiment $i"
    /Users/javiergarcia/miniconda3/envs/cs224n_dfp/bin/python3 -m modal volume get jbelle-data "/$i/predictions" modal_results
    /Users/javiergarcia/miniconda3/envs/cs224n_dfp/bin/python3 -m modal volume get jbelle-data "/$i/logs" modal_results
done

