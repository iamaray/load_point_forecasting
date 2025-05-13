#!/bin/bash
#SBATCH -Jtransformer_all                        # Job name
#SBATCH --account=gts-sd111                      # charge account
#SBATCH -N1                                      # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=12G                        # Memory per core
#SBATCH -t120:00:00                              # Duration of the job (allow for all datasets)
#SBATCH -qinferno                                # QOS Name
#SBATCH -oReport-transformer-all-%j.out          # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL               # Mail preferences
#SBATCH --mail-user=anahri3@gatech.edu           # E-mail address for notifications
#SBATCH --gres=gpu:V100:1

cd $HOME/load_point_forecasting               # Change to working directory created in $HOME

# Conda env
module load anaconda3
conda activate epf

# Define the datasets
datasets=("ercot" "spain" "elia" "homestead")

# Loop through datasets and run grid search for each
for dataset in "${datasets[@]}"; do
    echo "Starting transformer grid search for $dataset dataset"
    
    # Run grid search for current dataset
    python main.py --config="cfgs/grid_search/transformer/transformer_gs_${dataset}.json"
    
    # Check if the job was successful
    if [ $? -eq 0 ]; then
        echo "Completed transformer grid search for $dataset dataset"
    else
        echo "Failed transformer grid search for $dataset dataset"
    fi
    
    # Wait a bit before starting the next job to ensure resources are released
    sleep 30
done

echo "All transformer grid search jobs completed" 