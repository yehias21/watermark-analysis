#!/bin/bash

# Base directory containing the subfolders
BASE_DIR=~/watermark-analysis/attacked/4
CACHE_DIR=~/watermark-analysis/cache/test_dataset_dwtdct

# Python script and arguments
SCRIPT=calculate_performance_metrics.py
ALGORITHM=dwtdctsvd

# Export necessary variables for parallel execution
export SCRIPT
export CACHE_DIR
export ALGORITHM

# Find all subdirectories in BASE_DIR and run the Python script in parallel
find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d | parallel --jobs 23 \
"python \$SCRIPT --images_path {} --csv_path \$CACHE_DIR --algorithm \$ALGORITHM"
