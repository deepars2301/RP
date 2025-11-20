#!/bin/bash
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=24       # Number of cores per node
#SBATCH --time=24:00:00            # Max run time
#SBATCH --mail-user=Niteesh-bharadwaj.GANESH@student.isae-supaero.fr
#SBATCH --mail-type=FAIL,END       # Email on fail or end
#SBATCH --job-name=hover        # Job name
#SBATCH -e takeoff.%j.err          # Error log
#SBATCH --partition=p24            # Target partition

# Activate virtual environment (adjust if needed)
source /scratch/students/n.ganesh/vtol/venv/bin/activate

# Run the training script
python3 ./mavion_hover_train_1.py

