#!/usr/bin/env python
"""Run a PSO batch experiment.

To find out how to use this program, run it with the '-h' or '--help' option
""" 

#------------------------------------------------------------------------------

from __future__ import division
from mrs import param

from simulation import Simulation
from cli import outputtypes, gen_simple_options, gen_varargs_options, \
        prefix_args

import sys, os


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def main():
    simprefix = 'sim'

    versioninfo = '$Id: pso-batch 1.8 2008-9-11 1:13PM mjg82 $'

    parser = param.OptionParser()
    parser.add_option('-q', '--quiet',
            dest='quiet',
            action='store_true',
            help='Refrain from printing version and option information'
            )
    parser.add_option('-V', '--version',
            dest='version',
            action='store_true',
            help='Print version information and exit'
            )
    parser.add_option('-o', '--outputfreq',
            dest='outputfreq',
            default=1,
            type='int',
            help='Number of iterations per value output',
            )
    parser.add_option('-t', '--outputtype',
            dest='outputtype',
            default='BasicOutput',
            help='Style of output {%s}' % ", ".join(outputtypes),
            )
    parser.add_option('-d', '--dimensions',
            dest='dimensions',
            default=2,
            type='int',
            help='Number of dimensions'
            )
    parser.add_option('-n','--num-particles',
            dest='numparts',
            default=2,
            type='int',
            help='Number of particles'
            )
    parser.add_option('-f','--function',
            action='extend',
            dest='function',
            default='sphere.Sphere',
            search=['functions'],
            help='Function to optimize.'
            )
    parser.add_option('-b','--batches',
            dest='batches',
            type='int',
            default=1,
            help='Number of complete experiments to run'
            )
    parser.add_option('-v','--verbose',
            dest='verbose',
            action='store_true',
            default=False,
            help="Print out verbose error messages",
            )

    gen_simple_options( parser, simprefix, 'Simulation', Simulation.args )

    parser.add_option('-m','--motion',
            action='extend',
            dest='motion',
            search=['motion.basic', 'motion'],
            default='Basic',
            help='Particle motion type',
            )

    parser.add_option('-s','--sociometry',
            action='extend',
            dest='sociometry',
            search=['neighborhood.fixed', 'neighborhood'],
            default='Star',
            help='Particle sociometry'
            )

    options, args = parser.parse_args()

    if options.version:
        print "%s" % versioninfo
        sys.exit(0)

    #--------------------------------------------------------------------------
    # Create the simulation arguments, output header information
    #--------------------------------------------------------------------------

    if not options.quiet:
        from datetime import datetime
        date = datetime.now()
        print "# %s" % (versioninfo,)
        print "# Date run: %d-%d-%d" %(date.year, date.month, date.day)
        print "# ** OPTIONS **"
        for o in parser.option_list:
            if o.dest is not None:
                print "#     %s = %r" % (o.dest, getattr(options,o.dest))

    numparticles = options.numparts
    numiters = options.iterations
    numdims = options.dimensions

    # Format the 'extra' arguments for the simulation object -- removing
    # prefixes and such.
    simargs = {}
    simargs.update(prefix_args(simprefix, options))
    simargs['dims'] = numdims

    #--------------------------------------------------------------------------
    # Perform the simulation in batches
    #--------------------------------------------------------------------------
    freq = options.outputfreq

    for batch in xrange(options.batches):
        function = param.instantiate(options, 'function')
        sociometry = param.instantiate(options, 'sociometry')
        motion = param.instantiate(options, 'motion')
        sim = Simulation(
                options.numparts,
                sociometry,
                function,
                motion,

                **simargs
                )

        try:
            tmpfiles = sim.func.tmpfiles()
        except AttributeError:
            tmpfiles = []

        if options.useevals:
            simiter = sim.iterevals()
        else:
            simiter = sim.iterbatches()


        # Separate by two blank lines and a header
        print
        print
        if (options.batches > 1):
            print "# Batch %d" % batch

        # Note: some output types really need to get initialized just in time.
        outputter = outputtypes[options.outputtype]()

        # Perform the simulation
        try:
            for i in xrange(numiters):
                soc, iters = simiter.next()
                if 0 == (i+1) % freq:
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
    #--------------------------------------------------------------------------

#------------------------------------------------------------------------------
if __name__ == '__main__': main()
#------------------------------------------------------------------------------
