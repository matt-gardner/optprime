#!/bin/bash
#PBS -l nodes=1:ppn=1,pmem=1GB,walltime=100:00:00
#PBS -N manyiters2
NAME="manyiters2"
OUTDIR="$HOME/specex/$NAME"
mkdir -p "$OUTDIR"
OUTFILE="$(mktemp -p $OUTDIR out.XXXXX)"
ERRFILE="$(mktemp -p $OUTDIR err.XXXXX)"

# rbf dims: num_bases * (1 + 2 * num_input_dimensions)
python2.6 $HOME/clone/amlpso/specex.py \
    -f rbf.RBF -d 30 \
    --func-npoints=10000 --func-data-noise=2 --func-evenly-spaced \
    -t Rand --top-num=18 --top-neighbors=2 \
    -s PickBestChild -p ManyItersSevenEvals \
    -i 1000 -b 1 -o Everything \
    >"$OUTFILE" 2>"$ERRFILE"
