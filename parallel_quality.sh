#!/bin/bash

# Base directories
BASE_DIR=~/watermark-analysis/attacked/4
REF_FOLDER=~/watermark-analysis/cache/test_dataset_dwtdct

# Python script
SCRIPT=calculate_quailty_metrics.py

# Export necessary variables for parallel execution
export SCRIPT
export REF_FOLDER

# Find all subdirectories in BASE_DIR and run the Python script in parallel
find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d | parallel --jobs 12 \
"CUDA_VISIBLE_DEVICES=6 python \$SCRIPT --ref_folder \$REF_FOLDER --target_folder {}"
