#!/bin/bash

# Define the number of scripts to run
N=5  # Change this value to the number of scripts you have

# Loop through each script from python1.py to pythonN.py
for i in $(seq 0 $N)
do
  # Construct the script name
  script="main_alphas-0.py"
  
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
