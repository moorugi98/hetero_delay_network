#!/bin/bash

# set time limit to 2 days and memory to 60 GB, kernel to 48
#SBATCH --time=02:00:00
#SBATCH --mem=60000
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=3


# send email
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikang@uos.de


# seperate error and output file
#SBATCH -o /users/kang/output/%A.out
#SBATCH -e /users/kang/output/%A.err

# activate the conda environment
source activate test

# for each configuration repeat five times
declare -a configlist=('noise' 'random' 'topo')
for network in "${configlist[@]}"
do
    for runindex in {0..4}
    do
        srun python /users/kang/main.py $network 10000 $runindex &
    done
    wait
done
