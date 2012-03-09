JOBNAME="ring-subring6"
../mrs/examples/clusterrun.py -n "$JOBNAME" \
    --interpreter='PYTHONPATH="$PYTHONPATH:/usr/lib/python2.7/site-packages" pypy' \
    -h /admin/potatoes/all6 \
    ./subswarmpso.py -f rastrigin.Rastrigin -d 50 \
    -t Ring -l Ring --link-num 100 \
    --top-num 1000 \
    --link-neighbors 5 --top-neighbors 25 -s 100 -i 3000
