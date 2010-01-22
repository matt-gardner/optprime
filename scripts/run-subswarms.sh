JOBNAME="ring-subring5"
SCRATCH_DIR="/prodigy/amcnabb"
../mrs/examples/potato.py -n "$JOBNAME" -s "$SCRATCH_DIR" -h /admin/potatoes/russet4 -h /admin/potatoes/sweet4 -h /admin/potatoes/yukon4 \
    ./subswarmpso.py -f rastrigin.Rastrigin -d 50 -t Ring -l Ring --link-num 100 --top-num 1000 --link-neighbors 5 --top-neighbors 25 -s 100 -i 3000 -b 20
