#!/bin/bash
#SBATCH -Jrun_all                    # Job name
#SBATCH --account=gts-sd111                      # charge account
#SBATCH -N1                                      # Number of nodes and cores per node required
#SBATCH -t1:00:00                               # Duration of the job (Ex: 1 hour)
#SBATCH -qinferno                                # QOS Name
#SBATCH -oReport-%j.out                          # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL               # Mail preferences
#SBATCH --mail-user=anahri3@gatech.edu           # E-mail address for notifications

cd $HOME/load_point_forecasting

module load anaconda3
conda activate epf

config_files=($(find ./batch_jobs -name "*.sbatch"))

echo "Found ${#config_files[@]} config files to process"

for batch_file in "${config_files[@]}"; do
    echo "Processing config: $cfg_file"
    sbatch "$batch_file"
    
    sleep 2
done

echo "All jobs completed"