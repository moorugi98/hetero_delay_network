#!/bin/bash

# set time limit to 2 days and memory to 60 GB, kernel to 48
#SBATCH --time=00:20:00
#SBATCH --mem=61000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24


# send email
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikang@uos.de


# seperate error and output file
#SBATCH -o /users/kang/output/%A.out
#SBATCH -e /users/kang/output/%A.err

# activate the conda environment
source activate test

srun python main_delay.py noise 200 0 unimodal null 0.2 0
