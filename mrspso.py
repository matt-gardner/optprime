#!/usr/bin/env python

from __future__ import division
import sys, optparse, operator

import mrs
from particle import Particle


# TODO: The fact that we're minimizing instead of maximizing is currently
# hardcoded in. (use --sim-maximize)

# TODO: We currently assume that your gbest is the best pbest you've ever seen
# among all of your neighbors; an alternative interpretation might be that
# it's the best pbest among all of your current neighbors.

# TODO: allow the initial set of particles to be given

function = None
motion = None
cli_parser = None
comparator = None


def run(job, args, opts):
    """Run Mrs PSO function
    
    This is run on the master.
    """
    # Report parameters
    if not opts.quiet:
        #print "# %s" % (versioninfo,)
        print "# ** OPTIONS **"
        for o in cli_parser.option_list:
            if o.dest is not None:
                print "#     %s = %r" % (o.dest, getattr(opts,o.dest))
        sys.stdout.flush()

    try:
        tty = open('/dev/tty', 'w')
    except IOError:
        tty = None

    # Note: some output types really need to get initialized just in time.
    from cli import outputtypes
    outputter = outputtypes[opts.outputtype]()

    numparts = opts.numparts
    numtasks = opts.numtasks
    if numtasks == 0:
        numtasks = numparts

    # Create the initial population:
    import tempfile
    directory = tempfile.mkdtemp(dir=opts.mrs_shared, prefix=('population_'))
    pop = Population(function, directory)
    pop.add_random(numparts)

    new_data = pop.mrsdataset(numtasks)

    iters = 1
    wait = False
    running = True
    while True:
        if (opts.iterations >= 0) and (iters > opts.iterations):
            running = False

        if (opts.iterations < 0) or (iters <= opts.iterations):
            # Submit one PSO iteration to the job:
            interm_data = job.map_data(new_data, pso_map, nparts=numtasks,
                    parter=mrs.mod_partition)
            new_data = job.reduce_data(interm_data, pso_reduce,
                    nparts=numtasks, parter=mrs.mod_partition)

        if wait:
            # Wait until output_data are computed.
            ready = []
            while not ready:
                if tty:
                    ready = job.wait(output_data, timeout=1.0)
                    if ready:
                        print >>tty, "Finished iteration", iters-1
                    else:
                        print >>tty, job.status()
                else:
                    ready = job.wait(new_data)

            # Download output_data and update population accordingly.
            output_data.fetchall()
            reduce_id, first_item = output_data[0, 0][0]
            pop.set_bestparticle(Particle(state=first_item))

            # Print out the results.
            outputter(pop, iters)
            sys.stdout.flush()
            wait = False

        if not running:
            break

        if 0 == (iters+1) % opts.outputfreq:
            # Create a new output_data MapReduce phase to find the best
            # particle in the population.
            collapsed_data = job.map_data(new_data, collapse_map, nparts=1)
            output_data = job.reduce_data(collapsed_data, findbest_reduce,
                    nparts=1)

            # The next PSO iteration will be computed concurrently with the
            # output phase (they both depend on the same data).  We will wait
            # for our results after the PSO iteration is submitted.
            wait = True

        iters += 1

    print "# DONE"


def setup(opts):
    """Mrs Setup (run on both master and slave)"""
    from motion.basic import Basic
    from simulation import functions
    from cli import prefix_args

    global function, motion, comparator

    funcls = functions[opts.function]
    funcargs = prefix_args(FUNCPREFIX, opts)
    funcargs['dims'] = opts.dims
    function = funcls(**funcargs)

    #TODO: comparator = opts.soc_maximize and operator.gt or operator.lt
    comparator = operator.lt
    motion = Basic(comparator, function.constraints)



##############################################################################
# PRIMARY MAPREDUCE

def pso_map(key, value):
    particle = Particle(pid=int(key), state=value)

    # Update the particle:
    newpos, newvel = motion(particle, particle.gbest)
    value = function(newpos)
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


def pso_reduce(key, value_iter):
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
# MAPREDUCE TO FIND BEST PARTICLE

def collapse_map(key, value):
    yield '0', value

def findbest_reduce(key, value_iter):
    best = None
    for value in value_iter:
        p = Particle(state=value)
        if (best is None) or (comparator(p.bestval, best.bestval)):
            best = p
    yield repr(best)


##############################################################################
# POPULATION


class Population(object):
    """Population of particles.

    A Population in Mrs PSO is much like a neighborhood in Chris' PSO, but
    the interface is a little different.
    """
    def __init__(self, func, directory, **kargs):
        """Initialize Population instance using a function instance."""
        self.directory = directory
        self.particles = []
        self._bestparticle = None
        self.func = func
        #self.is_better = kargs.get('comparator', operator.lt)
        try:
            self.rand = kargs['rand']
        except KeyError:
            import random
            self.rand = random.Random()

    def mrsdataset(self, partitions=None):
        """Create a Mrs DataSet for the particles in the population.
        
        The number of partitions may be specified.
        """
        particles = [(str(p.id), repr(p)) for p in self.particles]

        if partitions is None:
            partitions = len(particles)

        dataset = mrs.datasets.Output(mrs.mod_partition, partitions,
                directory=self.directory)
        dataset.collect(particles)
        # TODO: this should eventually happen automatically in Mrs:
        dataset.dump()
        return dataset

    def get_particles(self):
        """Return a list of all particles in the population."""
        return self.particles

    def bestparticle(self):
        return self._bestparticle

    def set_bestparticle(self, particle):
        self._bestparticle = particle

    def __len__(self):
        return len(self.particles)

    def numparticles(self):
        return len(self.particles)

    def add_random(self, n=1):
        """Add n new random particles to the population."""
        from cubes.cube import Cube
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
            p = Particle(newpos, newvel, pid=i)
            #p.deps = deps
            #p.dep_str = dep_str
            # Loosely connected ring sociometry:
            p.deps = [(i-1)%n,(i+1)%n]
            p.dep_str = str((i-1)%n) + ',' + str((i+1)%n)
            self.particles.append(p)

    def __str__(self):
        return '\n'.join(str(p) for p in self.particles)

    def __repr__(self):
        return str(self)


##############################################################################
# BUSYWORK

FUNCPREFIX = 'func'

def update_parser(parser):
    from simulation import functions
    from cli import outputtypes
    global cli_parser

    cli_parser = parser
    parser.add_option('-q', '--quiet', dest='quiet', action='store_true',
            help='Refrain from printing version and option information')
    parser.add_option('-i', '--iterations', dest='iterations', type='int',
            help='Number of iterations to run')
    parser.add_option('-n', '--num-particles', dest='numparts', type='int',
            help='Number of particles')
    parser.add_option('-N', '--num-tasks', dest='numtasks', type='int',
            help='Number of tasks (if 0, create 1 task per particle)')
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
    parser.set_defaults(quiet=False, iterations=100, outputfreq=1,
            outputtype='BasicOutput', dims=2, numtasks=0, numparts=2,
            function='Sphere')

    from cli import gen_varargs_options
    gen_varargs_options(parser, FUNCPREFIX, 'Function', functions)

    return parser


if __name__ == '__main__':
    registry = mrs.Registry(globals())
    registry.add(mrs.mod_partition)
    mrs.main(registry, run, setup, update_parser)

# vim: et sw=4 sts=4
