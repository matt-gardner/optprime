JOBNAME="rbf-manyiters"
# rbf dims: num_bases * (1 + 2 * num_input_dimensions)
$HOME/c/mrs/examples/potato.py -n "$JOBNAME" -h /admin/potatoes/all6 \
    $HOME/c/optprime/specex.py \
    -f rbf.RBF -d 30 \
    --func-npoints=10000 --func-data-noise=2 --func-evenly-spaced \
    -t Rand --top-num=18 --top-neighbors=2 \
    -s PickBestChild -p ManyItersSevenEvals \
    -i 1000 -b 15 -o Everything \
