#!/bin/bash

# Array of directories to process
directories=("qqp" "rte" "sst2" "sts-b")

# Iterate through each directory
for dir in "${directories[@]}"; do
  echo "Processing directory: $dir"
  
  # Find all scripts in the directory and sort them
  scripts=$(find "$dir" -type f -name "*.sh" | sort)

  # Iterate through each script in the current directory
  for script in $scripts; do
    echo "Running script: $script"
    
    # Make sure the script is executable
    chmod +x "$script"

    # Execute the script
    ./"$script"

    # Check if the script executed successfully
    if [ $? -ne 0 ]; then
      echo "Error occurred while running $script. Exiting."
      exit 1
    fi
  done
done

echo "All experiments completed successfully."
