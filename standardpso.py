#!/usr/bin/env python

from __future__ import division
import sys, optparse, operator

import mrs
from mrs import param
from particle import Particle


# TODO: We currently assume that your nbest is the best pbest you've ever seen
# among all of your neighbors; an alternative interpretation might be that
# it's the best pbest among all of your current neighbors.

# TODO: allow the initial set of particles to be given


class MrsPSO(mrs.MapReduce):
    def __init__(self, opts, args):
        """Mrs Setup (run on both master and slave)"""

        super(MrsPSO, self).__init__(opts, args)

        self.func = param.instantiate(opts, 'func')
        self.motion = param.instantiate(opts, 'motion')

        self.func.setup()
        self.motion.setup(self.func)

    def bypass(self):
        """Run a "native" version of PSO without MapReduce."""

        self.cli_startup()

        self.topology = param.instantiate(self.opts, 'top')
        self.topology.setup(self.func)

        self.output = param.instantiate(self.opts, 'out')
        self.output.start()

    def run(self, job):
        """Run Mrs PSO function
        
        This is run on the master.
        """
        self.cli_startup()
        try:
            tty = open('/dev/tty', 'w')
        except IOError:
            tty = None


        particles = [(str(p.id), repr(p)) for p in self.particles]
        numtasks = self.opts.numtasks
        if not numtasks:
            numtasks = len(particles)
        new_data = job.local_data(particles, parter=mrs.mod_partition,
                splits=numtasks)

        iters = 1
        running = True
        while True:
            # Check whether we need to collect output for the previous
            # iteration.
            output_data = None
            if (iters != 1) and (((iters-1) % opts.outputfreq) == 0):
                if outputter.require_all:
                    output_data = new_data
                else:
                    # Create a new output_data MapReduce phase to find the
                    # best particle in the population.
                    collapsed_data = job.map_data(new_data, collapse_map,
                            splits=1)
                    output_data = job.reduce_data(collapsed_data,
                            findbest_reduce, splits=1)

            if (opts.iterations >= 0) and (iters > opts.iterations):
                # The previous iteration was the last iteration.
                running = False
            else:
                # Submit one PSO iteration to the job:
                interm_data = job.map_data(new_data, pso_map, splits=numtasks,
                        parter=mrs.mod_partition)
                new_data = job.reduce_data(interm_data, pso_reduce,
                        splits=numtasks, parter=mrs.mod_partition)

            if output_data:
                # Note: The next PSO iteration is being computed concurrently
                # with the output phase (they both depend on the same data).

                ready = []
                while not ready:
                    if tty:
                        ready = job.wait(output_data, timeout=1.0)
                        if ready:
                            print >>tty, "Finished iteration", iters-1
                        else:
                            print >>tty, job.status()
                    else:
                        ready = job.wait(output_data)

                # Download output_data and update population accordingly.
                output_data.fetchall()
                pop.particles = []
                for bucket in output_data:
                    for reduce_id, particle in bucket:
                        pop.particles.append(Particle(state=particle))

                if not outputter.require_all:
                    pop.set_bestparticle(pop.particles[0])

                # Print out the results.
                outputter(pop, iters)
                sys.stdout.flush()

            if not running:
                job.wait(new_data)
                break

            iters += 1

        self.output.finish()
        print "# DONE"

    ##########################################################################
    # PRIMARY MAPREDUCE

    def pso_map(key, value):
        particle = Particle(pid=int(key), state=value)

        # Update the particle:
        newpos, newvel = motion(particle)
        value = function(newpos)
        particle.update(newpos, newvel, value, comparator)

        # Emit a message for each dependent particle:
        message = particle.make_message()
        for dep_id in particle.deps:
            if dep_id == particle.id:
                particle.nbest_cand(particle.pbestpos, particle.pbestval,
                        comparator)
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
            if comparator(record.nbestval, bestval):
                best = record
                bestval = record.nbestval

            if not record.is_message():
                particle = record

        if particle:
            particle.nbest_cand(best.nbestpos, bestval, comparator)
            yield repr(particle)
        else:
            yield repr(best)


    ##########################################################################
    # MAPREDUCE TO FIND BEST PARTICLE

    def collapse_map(key, value):
        yield '0', value

    def findbest_reduce(key, value_iter):
        best = None
        for value in value_iter:
            p = Particle(state=value)
            if (best is None) or (comparator(p.pbestval, best.pbestval)):
                best = p
        yield repr(best)


    ##########################################################################
    # Helper Functions (shared by bypass and mrs implementations)

    def cli_startup(self):
        """Checks whether the repository is dirty and reports options."""
        import cli
        import sys

        # Check whether the repository is dirty.
        mrs_status = cli.GitStatus(mrs)
        amlpso_status = cli.GitStatus(sys.modules[__name__])
        if not self.opts.shamefully_dirty:
            if amlpso_status.dirty:
                print >>sys.stderr, (('Repository amlpso (%s) is dirty!'
                        '  Use --shamefully-dirty if necessary.') %
                        amlpso_status.directory)
                sys.exit(-1)
            if mrs_status.dirty:
                print >>sys.stderr, (('Repository mrs (%s) is dirty!'
                        '  Use --shamefully-dirty if necessary.') %
                        mrs_status.directory)
                sys.exit(-1)

        # Report command-line options.
        if not self.opts.quiet:
            import email
            date = email.Utils.formatdate(localtime=True)
            print '#', sys.argv[0]
            print '# Date:', date
            print '# Git Status:'
            print '#   amlpso:', amlpso_status
            print '#   mrs:', mrs_status
            print "# Options:"
            for key, value in sorted(self.opts.__dict__.iteritems()):
                print '#   %s = %s' % (key, value)
            print ""
            sys.stdout.flush()


##############################################################################
# POPULATION


class Population(object):
    """Population of particles."""
    def __init__(self, constraints, **kargs):
        """Initialize Population instance using a function instance."""
        self.particles = []
        self._bestparticle = None
        self.constraints = constraints

    def mrsdataset(self, job, partitions=None):
        """Create a Mrs DataSet for the particles in the population.
        
        The number of partitions may be specified.
        """
        particles = [(str(p.id), repr(p)) for p in self.particles]

        if partitions is None:
            partitions = len(particles)

        dataset = job.local_data(particles, parter=mrs.mod_partition,
                splits=partitions)
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
        sizes = [abs(cr-cl) for cl,cr in self.constraints]
        vconstraints = [(-s,s) for s in sizes]
        c = Cube(self.constraints)
        vc = Cube(vconstraints)
        for i in xrange(n):
            newpos = c.random_vec(self.rand)
            newvel = vc.random_vec(self.rand)
            p = Particle(newpos, newvel, pid=i)
            p.deps = deps
            p.dep_str = dep_str
            # Loosely connected ring sociometry:
            p.deps = [i%n for i in xrange(i-1,i+2)]
            p.dep_str = ''
            for i in xrange(len(p.deps)):
                p.dep_str  += str(p.deps[i]) + ','
            p.dep_str = p.dep_str[:-1]
            # End loosely connected ring sociometry - uncomment these blocks
            # for ring (changing the constant in the first line as desired)
            # Comment them out for star
            self.particles.append(p)

    def __str__(self):
        return '\n'.join(str(p) for p in self.particles)

    def __repr__(self):
        return str(self)


##############################################################################

#class StandardPSO(ParamObj):
class StandardPSO:
    def __init__(self):
        self.rand = Random()
        self.nparts = nparts
        self.func = function
        self.particles = {}
        self.motion = motion

        self.func.setup(self.dims)
        self.motion.setup(self.comparator, self.func.constraints)

    def iterevals(self):
        for i, (particle, soc) in enumerate(self.particles):
            yield soc, i

    def setup(self):
        self.iters = 0
        self._bestparticle = None
        self.last_updated_particle = None

    def __iter__(self):
        return self

    def next(self):
        p, soc = self.evaliter.next()
        self.last_updated_particle = p
        return p, soc

    def _iterevals(self):
        # First create the particles:
        n = self.startparticles
        sim = self.simulation
        for i, p in enumerate(sim.iternewparticles( numparts=n )):
            # Set up the index and yield self, from which everything can be
            # obtained.  We yield here because each particle creation is a
            # function evaluation.
            p.idx = i
            if self._bestparticle is None:
                self._bestparticle = p
            elif self.is_better(p.bestval, self._bestparticle.bestval):
                self._bestparticle = p
            self.addparticle( p )
            # We have to set initialized BEFORE we yield so that it's ready
            # AFTER we yield
            self.iters += 1
            yield p, self

        while True:
            # Now update all of the particles in batch
            np = self.numparticles()
            newstates = [None] * np
            # We move them in batches, which requires us to keep track of a set
            # of new values and to set them all at once.
            particles = tuple(self.iterparticles())
            bestpart = None
            bestval = None
            bestidx = None

            sim.motion.pre_batch( self )
            for i, p in enumerate(particles):
                if bestpart is None or self.is_better(p.bestval,bestval):
                    bestpart = p
                    bestval = p.bestval
                    bestidx = i
                self.update_nbest(p)
                newstates[i] = sim.motion(p)
            sim.motion.post_batch( self )
            bestpos = bestpart.bestpos

            for i, (p, state) in enumerate(izip(particles, newstates)):
                pos, vel = state
                val = sim.func(pos)
                p.update(pos, vel, val, sim.comparator)
                p.new = False
                self.iters += 1
                if self.is_better( p.bestval, self._bestparticle.bestval ):
                    self._bestparticle = p
                yield p, self

    def addparticle(self, particle):
        self.particles.append(particle)

    def iterparticles(self):
        return iter(self.particles)

    def bestparticle(self):
        return self._bestparticle

    def update_nbest(self, particle):
        """Updates the global best for this particle"""
        for i in self.iterneighbors(particle):
            p = self.particles[i]
            particle.nbest_cand(p.bestpos, p.bestval, self.is_better)
            if self.transitive_best:
                particle.nbest_cand(p.nbestpos, p.nbestval, self.is_better)

    def numparticles(self):
        return len(self.particles)


def main():
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

##############################################################################
# BUSYWORK

def update_parser(parser):
    """Adds PSO options to an OptionParser instance."""
    # Set the default Mrs implementation to Bypass.
    parser.usage = parser.usage.replace('Serial', 'Bypass')
    parser.set_default('mrs', 'Bypass')

    parser.add_option('-q', '--quiet',
            dest='quiet', action='store_true',
            help='Refrain from printing version and option information',
            default=False,
            )
    parser.add_option('-v','--verbose',
            dest='verbose', action='store_true',
            help="Print out verbose error messages",
            default=False,
            )
    parser.add_option('-b','--batches',
            dest='batches', type='int',
            help='Number of complete experiments to run',
            default=1,
            )
    parser.add_option('-i','--iters',
            dest='iters', type='int',
            help='Number of iterations per batch',
            default=100,
            )
    parser.add_option('-f','--func', metavar='FUNCTION',
            dest='func', action='extend', search=['functions'],
            help='Function to optimize',
            default='sphere.Sphere',
            )
    parser.add_option('-m','--motion',
            dest='motion', action='extend', search=['motion.basic', 'motion'],
            help='Particle motion type',
            default='Constricted',
            )
    parser.add_option('-t','--top', metavar='TOPOLOGY',
            dest='top', action='extend', search=['topology'],
            help='Particle topology/sociometry',
            default='Complete',
            )
    parser.add_option('-o', '--out', metavar='OUTPUTTER',
            dest='out', action='extend', search=['output'],
            help='Style of output',
            default='Basic',
            )
    parser.add_option('-N', '--num-tasks',
            dest='numtasks', type='int',
            help='Number of tasks (if 0, create 1 task per particle)',
            default=0,
            )
    parser.add_option('--transitive-best',
            dest='transitive_best', action='store_true',
            help='Whether to send nbest to others instead of pbest',
            default=False
            )
    parser.add_option('--shamefully-dirty',
            dest='shamefully_dirty', action='store_true',
            help='Ignore errors from uncommitted changes (for testing only!)',
            default=False,
            )

    return parser


if __name__ == '__main__':
    mrs.main(MrsPSO, update_parser)

# vim: et sw=4 sts=4
