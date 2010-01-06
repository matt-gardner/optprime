#!/bin/bash

DIMS=50
FUNCTION=rastrigin.Rastrigin
TOPOLOGY=Ring
PARTICLES=50
NEIGHBORS=5
ITERATIONS=5000
OUTFREQ=100

shortfunc="${FUNCTION#*.}"
datadir="$HOME/clone/psodata/data_${shortfunc}_${DIMS}/${TOPOLOGY}_${PARTICLES}_${NEIGHBORS}"
template="iters_${ITERATIONS}_freq_${OUTFREQ}"

mkdir -p "$datadir"
outfile="$(mktemp --tmpdir=$datadir $template.XXX)"

python standardpso.py -i "$ITERATIONS" --out-freq=$OUTFREQ -b 20 -f $FUNCTION -d $DIMS -t $TOPOLOGY -n $PARTICLES |tee "$outfile"
