#!/bin/bash
#SBATCH --job-name="scgpt"
#SBATCH -p RM-shared
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 2:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=zah47@pitt.edu



# Load required modules
module load anaconda3

# Activate your conda environment
source activate scgpt

# Go to the working directory
cd /ocean/projects/cis240075p/heidarir/scgpt_model

# Run the script
python extract_embeddings_by_timepoint.py 