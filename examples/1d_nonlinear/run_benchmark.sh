#!/bin/bash

# Define the number of scripts to run
N=9

# Loop through each script from python1.py to pythonN.py
for i in $(seq 0 $N)
do
  # Construct the script name
  script="main_train_pnet_benchmark.py"
  # Run the script
  python3 $script --seed=${i}
  # Check if the script was successful
  if [ $? -eq 0 ]; then
    echo "$script executed successfully."
  else
    echo "$script failed to execute."
    exit 1
  fi

  # Construct the script name
  script="main_train_enet_benchmark.py"
  # Run the script
  python3 $script --seed=${i}
  # Check if the script was successful
  if [ $? -eq 0 ]; then
    echo "$script executed successfully."
  else
    echo "$script failed to execute."
    exit 1
  fi
done
