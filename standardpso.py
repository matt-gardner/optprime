#!/usr/bin/env python

from __future__ import division
import sys, operator

import mrs
from mrs import param
from particle import Particle, Message, unpack


# TODO: allow the initial set of particles to be given


class StandardPSO(mrs.MapReduce):
    def __init__(self, opts, args):
        """Mrs Setup (run on both master and slave)"""

        super(StandardPSO, self).__init__(opts, args)

        self.tmpfiles = None
        self.function = param.instantiate(opts, 'func')
        self.motion = param.instantiate(opts, 'motion')
        self.topology = param.instantiate(opts, 'top')

        self.function.setup()
        self.motion.setup(self.function)
        self.topology.setup(self.function)

    ##########################################################################
    # Bypass Implementation

    def bypass(self):
        """Run a "native" version of PSO without MapReduce."""

        if not self.cli_startup():
            return

        # Perform the simulation in batches
        try:
            for batch in xrange(self.opts.batches):
                # Separate by two blank lines and a header.
                print
                print
                if (self.opts.batches > 1):
                    print "# Batch %d" % batch
                self.bypass_batch(batch)
                print "# DONE" 
        except KeyboardInterrupt, e:
            print "# INTERRUPTED"

    def bypass_batch(self, batch):
        """Performs a single batch of PSO without MapReduce.

        Compare to the run_batch method, which uses MapReduce to do the same
        thing.
        """
        # Create the Population.
        rand = self.initialization_rand(batch)
        particles = list(self.topology.newparticles(batch, rand))

        # Perform PSO Iterations.  The iteration number represents the total
        # number of function evaluations that have been performed for each
        # particle by the end of the iteration.
        output = param.instantiate(self.opts, 'out')
        output.start()
        for iteration in xrange(1, 1 + self.opts.iters):
            self.bypass_iteration(particles)

            # Output phase.  (If freq is 5, output after iters 1, 6, 11, etc.)
            if not ((iteration-1) % output.freq):
                kwds = {}
                if 'iteration' in output.args:
                    kwds['iteration'] = iteration
                if 'particles' in output.args:
                    kwds['particles'] = particles
                if 'best' in output.args:
                    kwds['best'] = self.findbest(particles)
                output(**kwds)
        output.finish()

    def bypass_iteration(self, particles, swarmid=0):
        """Runs one iteration of PSO.

        The swarmid is used for seed initialization when this is used in
        subswarms.
        """
        # Update position and value.
        for p in particles:
            self.move_and_evaluate(p, swarmid)

        # Communication phase.
        comp = self.function.comparator
        for p in particles:
            self.set_neighborhood_rand(p, swarmid)
            for i in self.topology.iterneighbors(p):
                # Send p's information to neighbor i.
                neighbor = particles[i]
                neighbor.nbest_cand(p.pbestpos, p.pbestval, comp)
                if self.opts.transitive_best:
                    neighbor.nbest_cand(p.nbestpos, p.nbestval, comp)

    ##########################################################################
    # MapReduce Implementation

    def run(self, job):
        """Run Mrs PSO function
        
        This is run on the master.
        """
        if not self.cli_startup():
            return

        try:
            tty = open('/dev/tty', 'w')
        except IOError:
            tty = None

        # Perform the simulation in batches
        try:
            for batch in xrange(self.opts.batches):
                # Separate by two blank lines and a header.
                print
                print
                if (self.opts.batches > 1):
                    print "# Batch %d" % batch
                self.run_batch(job, batch, tty)
                print "# DONE" 
        except KeyboardInterrupt, e:
            print "# INTERRUPTED"

    def run_batch(self, job, batch, tty):
        """Performs a single batch of PSO using MapReduce.

        Compare to the bypass_batch method, which does the same thing without
        using MapReduce.
        """
        rand = self.initialization_rand(batch)
        init_particles = [(str(p.id), repr(p)) for p in
                self.topology.newparticles(batch, rand)]

        numtasks = self.opts.numtasks
        if not numtasks:
            numtasks = len(init_particles)
        new_data = job.local_data(init_particles, parter=self.mod_partition,
                splits=numtasks)

        output = param.instantiate(self.opts, 'out')
        output.start()

        # Perform iterations.  Note: we submit the next iteration while the
        # previous is being computed.  Also, the next PSO iteration depends on
        # the same data as the output phase, so they can run concurrently.
        last_swarm = new_data
        last_interm_data = None
        old_swarm = None
        last_out_data = None
        next_out_data = None
        last_iteration = 0
        for iteration in xrange(1, 1 + self.opts.iters):
            interm_data = job.map_data(last_swarm, self.pso_map,
                    splits=numtasks, parter=self.mod_partition)
            next_swarm = job.reduce_data(interm_data, self.pso_reduce,
                    splits=numtasks, parter=self.mod_partition)

            next_out_data = None
            if not ((iteration - 1) % output.freq):
                if 'particles' in output.args:
                    next_out_data = next_swarm
                elif 'best' in output.args:
                    # Create a new output_data MapReduce phase to find the
                    # best particle in the population.
                    collapsed_data = job.map_data(next_swarm,
                            self.collapse_map, splits=1)
                    next_out_data = job.reduce_data(collapsed_data,
                            self.findbest_reduce, splits=1)

            waitset = set()
            if iteration > 1:
                waitset.add(last_swarm)
            if last_out_data:
                waitset.add(last_out_data)
            while waitset:
                if tty:
                    ready = job.wait(timeout=1.0, *waitset)
                    if last_swarm in ready:
                        print >>tty, "Finished iteration", last_iteration
                else:
                    ready = job.wait(*waitset)

                # Download output data and store as `particles`.
                if last_out_data in ready:
                    if 'best' in output.args or 'particles' in output.args:
                        last_out_data.fetchall()
                        particles = []
                        for bucket in last_out_data:
                            for reduce_id, particle in bucket:
                                particles.append(Particle.unpack(particle))

                waitset -= set(ready)

            # Print out the results.
            if last_iteration > 0 and not ((last_iteration - 1) % output.freq):
                kwds = {}
                if 'iteration' in output.args:
                    kwds['iteration'] = last_iteration
                if 'particles' in output.args:
                    kwds['particles'] = particles
                if 'best' in output.args:
                    if len(particles) == 1:
                        best = particles[0]
                    else:
                        best = self.findbest(particles)
                    kwds['best'] = best
                output(**kwds)

            # Now that last_swarm is ready, last_interm_data and old_swarm can
            # be deleted.
            if old_swarm:
                old_swarm.close()
                old_swarm = None
            if last_interm_data:
                last_interm_data.close()
                last_interm_data = None

            # Set up for the next iteration.
            last_iteration = iteration
            old_swarm = last_swarm
            last_interm_data = interm_data
            last_swarm = next_swarm
            last_out_data = next_out_data

        output.finish()

    ##########################################################################
    # Primary MapReduce

    def pso_map(self, key, value):
        comparator = self.function.comparator
        particle = unpack(value)
        assert particle.id == int(key)
        self.move_and_evaluate(particle)

        # Emit the particle without changing its id:
        yield (key, repr(particle))

        # Emit a message for each dependent particle:
        message = particle.make_message(self.opts.transitive_best, comparator)
        self.set_neighborhood_rand(particle, 0)
        for dep_id in self.topology.iterneighbors(particle):
            yield (str(dep_id), repr(message))

    def pso_reduce(self, key, value_iter):
        comparator = self.function.comparator
        particle = None
        messages = []
        for value in value_iter:
            record = unpack(value)
            if isinstance(record, Particle):
                particle = record
            elif isinstance(record, Message):
                messages.append(record)
            else:
                raise ValueError

        assert particle, 'Missing particle %s in the reduce step' % key

        best = self.findbest(messages)
        if best:
            particle.nbest_cand(best.position, best.value, comparator)
        yield repr(particle)

    ##########################################################################
    # MapReduce to Find the Best Particle

    def collapse_map(self, key, value):
        yield '0', value

    def findbest_reduce(self, key, value_iter):
        particles = (Particle.unpack(value) for value in value_iter)
        best = self.findbest(particles)
        yield repr(best)

    ##########################################################################
    # Helper Functions (shared by bypass and mrs implementations)

    def move_and_evaluate(self, p, swarmid=0):
        """Moves, evaluates, and updates the given particle."""
        self.set_motion_rand(p, swarmid)
        if p.iters > 0:
            newpos, newvel = self.motion(p)
        else:
            newpos, newvel = p.pos, p.vel
        # TODO(?): value = self.function(newpos, p.rand)
        value = self.function(newpos)
        p.update(newpos, newvel, value, self.function.comparator)

    def findbest(self, candidates):
        """Returns the best particle or message from the given candidates."""
        comparator = self.function.comparator
        best = None
        for cand in candidates:
            if (best is None) or comparator(cand, best):
                best = cand
        return best

    def cli_startup(self):
        """Checks whether the repository is dirty and reports options.

        Returns True if startup succeeded, otherwise False.
        """
        import cli
        import sys

        # Check whether the repository is dirty.
        mrs_status = cli.GitStatus(mrs)
        amlpso_status = cli.GitStatus(sys.modules[__name__])
        if not self.opts.hey_im_testing:
            if amlpso_status.dirty:
                print >>sys.stderr, (('Repository amlpso (%s) is dirty!'
                        '  Use --hey-im-testing if necessary.') %
                        amlpso_status.directory)
                return False
            if mrs_status.dirty:
                print >>sys.stderr, (('Repository mrs (%s) is dirty!'
                        '  Use --hey-im-testing if necessary.') %
                        mrs_status.directory)
                return False

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
            for key, value in sorted(vars(self.opts).iteritems()):
                print '#   %s = %s' % (key, value)
            self.function.master_log()
            print ""
            sys.stdout.flush()

        return True

    ##########################################################################
    # Rand Setters

    # We define the rand offsets here, both for these rand setters, and for
    # those of all subclasses.  These really need to be unique, so let's put
    # them all in one place.
    MOTION_OFFSET = 1
    INITIALIZATION_OFFSET = 2
    SUBSWARM_OFFSET = 3
    NEIGHBORHOOD_OFFSET = 4

    def set_motion_rand(self, p, swarmid=0):
        """Makes a Random for the given particle and saves it to `p.rand`.

        This should be used just before performing motion on the particle.

        Note that the Random depends on the particle id, iteration, batch, and
        swarmid.  The swarmid is only needed in the subswarms case (to make
        sure that the particles in different subswarms have unique seeds), but
        it doesn't hurt the standardpso case to include it.
        """
        from mrs.impl import SEED_BITS
        base = 2 ** SEED_BITS
        offset = self.MOTION_OFFSET + base * (p.id + base * (p.iters + base *
            (p.batches + base * swarmid)))
        p.rand = self.random(offset)

    def set_neighborhood_rand(self, n, swarmid=0):
        """Makes a Random for the given node and saves it to `n.rand`.

        This should be used just before passing p to iterneighbors.  Note that
        depending on the PSO variant, node might be a particle, subswarm, or
        other object.

        Note that the Random depends on the particle id, iteration, batch, and
        swarmid.  The swarmid is only needed in the subswarms case (to make
        sure that the particles in different subswarms have unique seeds), but
        it doesn't hurt the standardpso case to include it.
        """
        from mrs.impl import SEED_BITS
        base = 2 ** SEED_BITS
        offset = self.NEIGHBORHOOD_OFFSET + base * (n.id + base * (n.iters +
            base * (n.batches + base * swarmid)))
        n.rand = self.random(offset)

    def initialization_rand(self, batch):
        """Returns a new Random for the given batch number.

        This ensures that each batch will have a unique initial swarm state.
        """
        from mrs.impl import SEED_BITS
        base = 2 ** SEED_BITS
        offset = self.INITIALIZATION_OFFSET + base * batch
        return self.random(offset)


##############################################################################
# Busywork

def update_parser(parser):
    """Adds PSO options to an OptionParser instance."""
    # Set the default Mrs implementation to Bypass (instead of MapReduce).
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
    parser.add_option('--hey-im-testing',
            dest='hey_im_testing', action='store_true',
            help='Ignore errors from uncommitted changes (for testing only!)',
            default=False,
            )

    return parser


if __name__ == '__main__':
    mrs.main(StandardPSO, update_parser)

# vim: et sw=4 sts=4
