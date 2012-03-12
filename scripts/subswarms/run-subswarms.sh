JOBNAME="ring-subring4"
../mrs/examples/clusterrun.py -n "$JOBNAME" \
    --interpreter='PYTHONPATH="$PYTHONPATH:/usr/lib/python2.7/site-packages" pypy' \
    --mrs-tmpdir=/local/tmp \
    -h /admin/potatoes/russet6 \
    -h /admin/potatoes/sweet6 \
    -h /admin/potatoes/yukon6 \
    ./subswarmpso.py \
    -f rastrigin.Rastrigin -d 50 \
    -l Ring --link-num 80 --link-neighbors 2 \
    -t Ring --top-num 50000 --top-neighbors 25 \
    -s 100 -i 10000 \
    -o TimedBasic \
    --hey-im-testing
