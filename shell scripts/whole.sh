#!/bin/bash
source activate test
simtime=10000.0
declare -a networklist=('noise' 'random' 'topo')
declare -a paramslist=(0.5 1.0 2.0 5.0)
intra_dist="unimodal"
inter_dist="null"
inter_params="d"

##### simulate
FLAGS="--ntasks=1 --cpus-per-task=24 --mem=60000 --time=04:00:00 -o /users/kang/output/sim_%A.out
-e /users/kang/output/sim_%A.err"

for network in "${networklist[@]}"; do
	for trial in {0..4}; do
		for intra_params in "${paramslist[@]}"; do
			####cp /users/kang/data/*$network* ${TMPDIR} maybe include them inside the wrapper
        		sbatch $FLAGS --wrap="python /users/kang/main_delay_fast.py $network $simtime $trial $intra_dist $inter_dist $intra_params $inter_params"
			###cp ${TMPDIR}/* /users/kang/data/
		done
	done
done
wait
#
###### measures
#FLAGS="--ntasks=1 --cpus-per-task=24 --mem=60000 --time=06:00:00 -o /users/kang/output/ana_%A.out
#-e /users/kang/output/ana_%A.err"
#for network in "${networklist[@]}"; do
#  for intra_params in "${paramslist[@]}"; do
#  sbatch $FLAGS --wrap="python /users/kang/measures.py $network $simtime $intra_dist $inter_dist $intra_params $inter_params &"
#  done
#done
#wait

###### train
#FLAGS="--ntasks=1 --cpus-per-task=24 --mem=60000 --time=02:00:00 -o /users/kang/output/train_%A.out
#-e /users/kang/output/train_%A.err"
#for network in "${networklist[@]}"; do
#  for intra_params in "${paramslist[@]}"; do
#  sbatch $FLAGS --wrap="python /users/kang/training.py $network $intra_dist $inter_dist $intra_params $inter_params"
#  done
#done

#### concatanate the data
FLAGS="--ntasks=1 --cpus-per-task=1 --mem=60000 --time=00:30:00 -o /users/kang/output/concat_%A.out
-e /users/kang/output/concat_%A.err"
sbatch $FLAGS --wrap="python /users/kang/concat.py $intra_dist $inter_dist $intra_params $inter_params"