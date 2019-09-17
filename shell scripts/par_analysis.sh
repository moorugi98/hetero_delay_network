#!/bin/bash
FLAGS="--ntasks=1 --cpus-per-task=24 --mem=60000 --time=08:00:00 -o /users/kang/output/%A.out -e /users/kang/output/%A.err"
source activate test
declare -a configlist=("noise" 'random' 'topo')
for network in "${configlist[@]}"; do
  for intra_params in {0.01 0.05 0.1 0.2 0.5 1.0}; do
  ####cp /users/kang/data/*$network* ${TMPDIR} maybe include them inside the wrapper
  ####echo ${TMPDIR}
  sbatch $FLAGS --wrap="python measures.py $network 10000 unimodal null $intra_params 0"
  ####cp ${TMPDIR}/* /users/kang/data/summary/
  done
done