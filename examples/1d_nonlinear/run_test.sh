#!/bin/bash

# Define the random seed
i=4

# # Construct the script name
# script="main_train_pnet.py"
# # Run the script
# python3 $script --seed=${i}
# # Check if the script was successful
# if [ $? -eq 0 ]; then
#   echo "$script executed successfully."
# else
#   echo "$script failed to execute."
#   exit 1
# fi

# Construct the script name
script="main_train_enet.py"
# Run the script
python3 $script --seed=${i}
# Check if the script was successful
if [ $? -eq 0 ]; then
  echo "$script executed successfully."
else
  echo "$script failed to execute."
  exit 1
fi
