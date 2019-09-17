#!/bin/bash
FLAGS="--ntasks=1 --cpus-per-task=24 --mem=60000 --time=01:00:00 -o /users/kang/output/%A.out -e /users/kang/output/%A.err"
declare -a configlist=('noise' 'random' 'topo')
intra_dist="unimodal"
inter_dist="null"
inter_params=0
for network in "${configlist[@]}"; do
	for trial in {0..4}; do
		for intra_params in {0.01 0.05 0.1 0.2 0.5 1.0}; do
			####cp /users/kang/data/*$network* ${TMPDIR} maybe include them inside the wrapper
        		sbatch $FLAGS --wrap="python main_delay_fast.py $network 10000 $trial $intra_dist $inter_dist $intra_params  $inter_params"
			###cp ${TMPDIR}/* /users/kang/data/
		done
	done
done

