#!/bin/bash

# Usage: dlmodal.sh <experiment_name>

/Users/javiergarcia/miniconda3/envs/cs224n_dfp/bin/python3 -m modal volume get jbelle-data "/$1/predictions" modal_results
/Users/javiergarcia/miniconda3/envs/cs224n_dfp/bin/python3 -m modal volume get jbelle-data "/$1/logs" modal_results
