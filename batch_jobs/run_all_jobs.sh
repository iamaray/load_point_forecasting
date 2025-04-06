#!/bin/bash
# Script to run all batch jobs

# Navigate to the batch_jobs directory
cd "$(dirname "$0")"

# Loop through all batch job files and submit them
for job_file in *.sbatch
do
    echo "Submitting job: $job_file"
    sbatch "$job_file"
    
    # Add a small delay between job submissions
    sleep 2
done

echo "All jobs submitted successfully!" 