#!/usr/bin/env python

def default_specex_opts():
    import optparse
    opts = optparse.Values()
    opts.verbose = False
    opts.func_center = '0.5'
    opts.mrs__verbose = False
    opts.top__num = 10
    opts.out = 'optprime.output.Basic'
    opts.mrs__seed = '2'
    opts.batches = 1
    opts.top__noselflink = False
    opts.top = 'optprime.topology.Ring'
    opts.out__freq = 1
    opts.motion__restrictvel = False
    opts.spec = 'optprime.specmethod.ReproducePSO'
    opts.func__dims = 2
    opts.func_maximize = False
    opts.iters = 100
    opts.top__initoffset = 0.0
    opts.top__initscale = 1.0
    opts.motion__phi2 = 2.05
    opts.motion__phi1 = 2.05
    opts.pruner = 'optprime.specmethod.OneCompleteIteration'
    opts.hey_im_testing = True
    opts.numtasks = 0
    opts.motion__Kappa = 1
    opts.mrs__debug = False
    opts.quiet = False
    opts.mrs = 'mrs.impl.Serial'
    opts.motion = 'optprime.motion.basic.Constricted'
    opts.transitive_best = False
    opts.func = 'optprime.functions.sphere.Sphere'
    opts.min_tokens = 0
    opts.tokens = 0
    return opts

# vim: et sw=4 sts=4
