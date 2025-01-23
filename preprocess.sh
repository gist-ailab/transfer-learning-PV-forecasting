#!/bin/bash

# Array of base directories
# base_dirs=("data_preprocessing_day" "data_preprocessing_night")
base_dirs=("data_preprocessing_day")

# Loop through each base directory
for base_dir in "${base_dirs[@]}"; do
  # Loop through each subdirectory in the base directory
  for dir in "$base_dir"/*/; do
    # Check if 1_unify_format.py exists in the subdirectory
    # if [ -f "${dir}1_unify_format.py" ]; then
    #   echo "Executing 1_unify_format.py in $dir"
    #   python "${dir}1_unify_format.py"
    # else
    #   echo "1_unify_format.py not found in $dir"
    # fi

    # Check if 2_drop_anomaly.py exists in the subdirectory
    if [ -f "${dir}2_drop_anomaly.py" ]; then
      echo "Executing 2_drop_anomaly.py in $dir"
      python "${dir}2_drop_anomaly.py"
    else
      echo "2_drop_anomaly.py not found in $dir"
    fi
  done
done
