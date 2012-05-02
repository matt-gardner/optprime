PROCS=128
SUBITERS=100
# Set to 100*$SUBITERS:
ITERS=10000
JOBNAME="a$PROCS-$SUBITERS"
NODESPEC="westmere"
$HOME/c/mrs/examples/fulton.py -N "$JOBNAME" \
    -n $PROCS -s $PROCS -o "$HOME/out/$JOBNAME" -t 0.5 -m 1 \
    --nodespec "$NODESPEC" \
    --interpreter /usr/bin/python2.6 \
    $HOME/c/amlpso/subswarmpso.py \
    -f rosenbrock.Rosenbrock -d 250 \
    -l Ring --link-num $PROCS --link-neighbors 1 \
    -t Ring --top-num 5 --top-neighbors 1 \
    -s $SUBITERS -i $ITERS \
    -o TimedBasic --out-freq=0 \
    --mrs-timing-interval=5 \
    --async

