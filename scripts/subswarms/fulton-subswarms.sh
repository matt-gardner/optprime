JOBNAME="ring-subring1"
$HOME/c/mrs/examples/fulton.py -N "$JOBNAME" \
    -n 100 -o "$HOME/out/$JOBNAME" -t 6 -m 2 \
    $HOME/c/amlpso/subswarmpso.py \
    -f rastrigin.Rastrigin -d 50 \
    -l Ring --link-num 100 --link-neighbors 1 \
    -t Ring --top-num 5000 --top-neighbors 10 \
    -s 100 -i 10000 \
    -o TimedBasic --out-freq=5 \
    --hey-im-testing
