#!/bin/bash
FLAGS="--ntasks=1 --cpus-per-task=24 --mem=60000 --time=00:10:00 -o /users/kang/output/%A.out -e /users/kang/output/%A.err"
declare -a configlist=('noise' 'random' 'topo')
for network in "${configlist[@]}"; do
    	for trial in {0..4}; do
                sbatch $FLAGS --wrap="python main.py $network 10000 $trial"
		###cp ${TMPDIR}/* /users/kang/data/
        done
    done
done
