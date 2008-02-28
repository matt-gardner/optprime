#!/usr/bin/env python

from __future__ import division
import optparse, operator

import mrs
from aml.opt.particle import Particle
from aml.opt.motion.basic import Basic
from aml.opt.simulation import functions
from aml.opt.cli import outputtypes


# TODO: The fact that we're minimizing instead of maximizing is currently
# hardcoded in.  Also note that we currently assume that your gbest is the
# best pbest you've ever seen among all of your neighbors; an alternative
# interpretation might be that it's the best pbest among all of your current
# neighbors.

# TODO: allow the initial set of particles to be given (use --sim-maximize)

function = None
motion = Basic(operator.lt, ((-100, 100),))


def run(job, args, opts):
    """Run Mrs PSO function
    
    This is run on the master.
    """

    # Note: some output types really need to get initialized just in time.
    outputter = outputtypes[opts.outputtype]()

    # Create the initial population:
    pop = Population(function)
    pop.add_random(opts.numparts)
    new_data = pop.mrsdataset()

    iters = 0
    while (opts.iterations < 0) or (iters <= opts.iterations):
        interm_data = job.map_data(new_data, mapper)
        new_data = job.reduce_data(inter_data, reducer)

        # FIXME
        if 0 == (iters+1) % opts.outputfreq:
            pass
            # TODO: make it so pop can read from a dataset
            #outputter(pop, iters)

        iters += 1

    print "# DONE"


def main():
    """Mrs PSO Main

    This is run on both master and slave.
    """
    global function

    parser = option_parser()
    opts, args = parser.parse_args()

    funcls = functions[opts.function]
    from aml.opt.cli import prefix_args
    funcargs = prefix_args(FUNCPREFIX, opts)
    function = funcls(**funcargs)

    # Report parameters
    if mrs.primary_impl(opts.mrs_impl) and not opts.quiet:
        #print "# %s" % (versioninfo,)
        print "# ** OPTIONS **"
        for o in parser.option_list:
            if o.dest is not None:
                print "#     %s = %r" % (o.dest, getattr(opts,o.dest))

    # TODO: at some point, we probably want to call func.tmpfiles()

    # Do MapReduce
    mrs.main(globals(), run, parser)


##############################################################################
# MAP FUNCTION

def mapper(key, value):
    particle = Particle(pid=int(key), state=value)

    # Update the particle:
    newpos, newvel = motion(particle, particle.gbest)
    value = func(newpos)
    particle.update(newpos, newvel, value, motion.comparator)

    # Emit a message for each dependent particle:
    message = particle.make_message()
    for dep_id in particle.deps:
        if dep_id == particle.id:
            particle.gbest_cand(particle.bestpos, particle.bestval)
        else:
            yield (str(dep_id), repr(message))

    # Emit the particle without changing its id:
    yield (str(key), repr(particle))


##############################################################################
# REDUCE FUNCTION

def reducer(key, value_iter):
    particle = None
    best = None
    bestval = float('inf')

    for value in value_iter:
        record = Particle(pid=int(key), state=value)
        if record.gbest.bestval <= bestval:
            best = record
            bestval = record.gbest.bestval

        if not record.is_message():
            particle = record

    if particle:
        particle.gbest_cand(best.gbest.bestpos, bestval)
        yield repr(particle)
    else:
        yield repr(best)


##############################################################################
# POPULATION

class Population(object):
    """Population of particles.

    A Population in Mrs PSO is much like a neighborhood in Chris' PSO, but
    the interface is a little different.
    """
    def __init__(self, func, **kargs):
        """Initialize Population instance using a function instance."""
        self.particles = []
        self.func = func
        self.is_better = kargs.get('comparator', operator.lt)
        try:
            self.rand = kargs['rand']
        except KeyError:
            import random
            self.rand = random.Random()

    def mrsdataset(self):
        """Create a Mrs DataSet for the particles in the population."""
        particles = [(str(p.id), str(p)) for p in self.particles]
        nparticles = len(particles)
        dataset = mrs.datasets.Output(mrs.mod_partition, nparticles)
        dataset.collect(particles)
        # TODO: this should eventually happen automatically in Mrs:
        dataset.dump()
        return dataset

    def get_particles(self):
        """Return a list of all particles in the population."""
        return self.particles

    def __len__(self):
        return len(self.particles)

    def numparticles(self):
        return len(self.particles)

    def bestparticle(self):
        best = None
        for p in self.particles:
            if (best is None) or (self.is_better(p.bestval, best.bestval)):
                best = p
        return best

    def add_random(self, n=1):
        """Add n new random particles to the population."""
        from aml.opt.cubes.cube import Cube
        # Fully connected sociometry:
        #deps = range(Particle.next_id, Particle.next_id + n)
        deps = range(n)
        dep_str = 'all-%s' % n
        sizes = [abs(cr-cl) for cl,cr in self.func.constraints]
        vconstraints = [(-s,s) for s in sizes]
        c = Cube(self.func.constraints)
        vc = Cube(vconstraints)
        for i in xrange(n):
            newpos = c.random_vec(self.rand)
            newvel = vc.random_vec(self.rand)
            p = Particle(newpos, newvel)
            p.deps = deps
            p.dep_str = dep_str
            self.particles.append(p)

    def __str__(self):
        return '\n'.join(str(p) for p in self.particles)

    def __repr__(self):
        return str(self)


##############################################################################
# BUSYWORK

FUNCPREFIX = 'func'

def option_parser():
    parser = mrs.option_parser()
    parser.add_option('-q', '--quiet', dest='quiet', action='store_true',
            help='Refrain from printing version and option information')
    parser.add_option('-i', '--iterations', dest='iterations', type='int',
            help='Number of iterations to run')
    parser.add_option('-n', '--num-particles', dest='numparts', type='int',
            help='Number of particles')
    parser.add_option('-o', '--outputfreq', dest='outputfreq', type='int',
            help='Number of iterations per value output')
    parser.add_option('-t', '--outputtype', dest='outputtype',
            help='Style of output {%s}' % ', '.join(outputtypes))
    parser.add_option('-d', '--dimensions', dest='dims', type='int',
            help='Number of dimensions')
    parser.add_option('-f', '--function', dest='function',
            help='Function to optimize {%s}' % ', '.join(functions))
    parser.add_option('-v', '--verbose', dest='verbose', action='store_true',
            default=False, help='Print out verbose error messages')
    parser.set_defaults(iterations=100, outputfreq=1, outputtype='BasicOutput',
            dims=2, numparts=2, function='Sphere')

    from aml.opt.cli import gen_varargs_options
    gen_varargs_options(parser, FUNCPREFIX, 'Function', functions)

    return parser

if __name__ == '__main__':
    main()

# vim: et sw=4 sts=4
