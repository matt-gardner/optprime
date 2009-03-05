#!/usr/bin/env python
"""Run a PSO batch experiment.

To find out how to use this program, run it with the '-h' or '--help' option
""" 

from __future__ import division
import os
import sys
from mrs import param
from population import Population


def main():
    parser = param.OptionParser()
    parser.add_option('-q', '--quiet',
            dest='quiet',
            action='store_true',
            help='Refrain from printing version and option information',
            )
    parser.add_option('-v','--verbose',
            dest='verbose',
            action='store_true',
            default=False,
            help="Print out verbose error messages",
            )
    parser.add_option('-b','--batches',
            dest='batches',
            type='int',
            default=1,
            help='Number of complete experiments to run',
            )
    parser.add_option('-i','--iters',
            dest='iters',
            type='int',
            default=100,
            help='Number of iterations per batch',
            )
    parser.add_option('-f','--func',
            action='extend',
            dest='func',
            metavar='FUNCTION',
            default='sphere.Sphere',
            search=['functions'],
            help='Function to optimize.',
            )
    parser.add_option('-m','--motion',
            action='extend',
            dest='motion',
            search=['motion.basic', 'motion'],
            default='Basic',
            help='Particle motion type',
            )
    parser.add_option('-t','--top',
            action='extend',
            dest='top',
            metavar='TOPOLOGY',
            search=['neighborhood.fixed', 'neighborhood'],
            default='Complete',
            help='Particle topology/sociometry',
            )
    parser.add_option('-o', '--out',
            action='extend',
            dest='out',
            metavar='OUTPUTTER',
            search=['output'],
            default='Basic',
            help='Style of output',
            )

    # We add suboptions for Population, but no --pop option (since there's
    # only one Population class).
    parser.set_defaults(pop=Population)
    parser.add_param_object(Population, 'pop')

    options, args = parser.parse_args()


    # Create the simulation arguments, output header information.
    if not options.quiet:
        from datetime import datetime
        date = datetime.now()
        print "# Date run: %d-%d-%d" %(date.year, date.month, date.day)
        print "# ** OPTIONS **"
        for o in parser.option_list:
            if o.dest is not None:
                print "#     %s = %r" % (o.dest, getattr(options,o.dest))

    # Perform the simulation in batches
    for batch in xrange(options.batches):
        function = param.instantiate(options, 'func')
        topology = param.instantiate(options, 'top')
        motion = param.instantiate(options, 'motion')
        population = param.instantiate(options, 'pop')
        output = param.instantiate(options, 'out')

        try:
            tmpfiles = function.tmpfiles()
        except AttributeError:
            tmpfiles = []

        simiter = sim.iterbatches()

        # Separate by two blank lines and a header.
        print
        print
        if (options.batches > 1):
            print "# Batch %d" % batch

        # Perform the simulation.
        output.start()
        try:
            for i in xrange(options.iters):
                soc, iters = simiter.next()
                if 0 == (i+1) % output.freq:
                    outputter(soc, iters)
            print "# DONE" 
        except KeyboardInterrupt, e:
            print "# INTERRUPTED"
        except Exception, e:
            if options.verbose:
                raise
            else:
                print "# ERROR"

        for f in tmpfiles:
            os.unlink(f)


if __name__ == '__main__':
    main()
