#!/bin/bash
#SBATCH -Jlstm_elia                              # Job name
#SBATCH --account=gts-sd111                      # charge account
#SBATCH -N1                                      # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=12G                        # Memory per core
#SBATCH -t48:00:00                               # Duration of the job (Ex: 1 hour)
#SBATCH -qinferno                                # QOS Name
#SBATCH -oReport-%j.out                          # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL               # Mail preferences
#SBATCH --mail-user=anahri3@gatech.edu           # E-mail address for notifications
#SBATCH --gres=gpu:V100:1

cd $HOME/load_point_forecasting               # Change to working directory created in $HOME

# Conda env
module load anaconda3
conda activate epf

# Run Code
srun python main.py --config="cfgs/grid_search/lstm/lstm_gs_elia_cfg.json"
