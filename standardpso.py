#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import datetime
from six import b
import sys
import time

import mrs
from mrs import param
from particle import Particle, Message, PSOPickler

try:
    range = xrange
except NameError:
    pass

# TODO: allow the initial set of particles to be given


class StandardPSO(mrs.IterativeMR):
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
            return 1

        # Perform simulation
        try:
            self.output = param.instantiate(self.opts, 'out')
            self.output.start()
            self.bypass_run()
            self.output.finish()
        except KeyboardInterrupt as e:
            print("# INTERRUPTED")
        return 0

    def bypass_run(self):
        """Performs PSO without MapReduce.

        Compare to the producer/consumer methods, which use MapReduce to do
        the same thing.
        """
        # Create the Population.
        rand = self.initialization_rand()
        particles = list(self.topology.newparticles(rand))

        # Perform PSO Iterations.  The iteration number represents the total
        # number of function evaluations that have been performed for each
        # particle by the end of the iteration.
        for iteration in range(1, 1 + self.opts.iters):
            self.bypass_iteration(particles)

            # Output phase.  (If freq is 5, output after iters 1, 6, 11, etc.)
            if not ((iteration - 1) % self.output.freq):
                kwds = {}
                if 'iteration' in self.output.args:
                    kwds['iteration'] = iteration
                if 'particles' in self.output.args:
                    kwds['particles'] = particles
                if 'best' in self.output.args:
                    kwds['best'] = self.findbest(particles)
                self.output(**kwds)
                if self.stop_condition(particles):
                    self.output.success()
                    return

    def bypass_iteration(self, particles, swarmid=0):
        """Runs one iteration of PSO.

        The swarmid is used for seed initialization when this is used in
        subswarms.
        """
        self.randomize_function_center()

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
            return 1

        # Perform the simulation
        try:
            self.iteration = 0
            self.output = param.instantiate(self.opts, 'out')
            self.output.start()

            job.default_partition = self.mod_partition
            if self.opts.numtasks:
                job.default_reduce_tasks = self.opts.numtasks
            else:
                job.default_reduce_tasks = self.topology.num

            mrs.IterativeMR.run(self, job)
            self.output.finish()
            return 0
        except KeyboardInterrupt as e:
            print("# INTERRUPTED")
            return 1

    def producer(self, job):
        if self.iteration > self.opts.iters:
            return []

        elif self.iteration == 0:
            self.out_datasets = {}
            self.datasets = {}
            out_data = None

            rand = self.initialization_rand()
            init_particles = [(repr(p.id).encode('ascii'), p.__getstate__())
                    for p in self.topology.newparticles(rand)]
            start_swarm = job.local_data(init_particles)
            data = job.map_data(start_swarm, self.pso_map)
            start_swarm.close()

        elif (self.iteration - 1) % self.output.freq == 0:
            out_data = job.reduce_data(self.last_data, self.pso_reduce)
            if (self.last_data not in self.datasets and
                    self.last_data not in self.out_datasets):
                self.last_data.close()
            data = job.map_data(out_data, self.pso_map)

        else:
            out_data = None
            if self.opts.split_reducemap:
                interm = job.reduce_data(self.last_data, self.pso_reduce)
                data = job.map_data(interm, self.pso_map)
            else:
                data = job.reducemap_data(self.last_data, self.pso_reduce,
                        self.pso_map)

        self.iteration += 1
        self.datasets[data] = self.iteration
        self.last_data = data
        if out_data:
            self.out_datasets[out_data] = self.iteration
            return [out_data, data]
        else:
            return [data]

    def consumer(self, dataset):
        # Note that depending on the output class, a dataset could be both
        # in out_datasets and datasets.

        if dataset in self.datasets:
            iteration = self.datasets[dataset]
            del self.datasets[dataset]

            #self.output.print_to_tty("Finished iteration %s" % iteration)
            if dataset not in self.out_datasets and dataset != self.last_data:
                dataset.close()

        if dataset in self.out_datasets:
            iteration = self.out_datasets[dataset]
            del self.out_datasets[dataset]

            if 'best' in self.output.args or 'particles' in self.output.args:
                dataset.fetchall()
                particles = []
                for reduce_id, particle in dataset.data():
                    particles.append(Particle.unpack(particle))
            if dataset != self.last_data:
                dataset.close()
            kwds = {}
            if 'iteration' in self.output.args:
                kwds['iteration'] = iteration
            if 'particles' in self.output.args:
                kwds['particles'] = particles
            if 'best' in self.output.args:
                kwds['best'] = self.findbest(particles)
            self.output(**kwds)
            if self.stop_condition(particles):
                self.output.success()
                return False

        return True

    ##########################################################################
    # Primary MapReduce

    def pso_map(self, key, value):
        comparator = self.function.comparator
        particle = PSOPickler.loads(value)
        assert particle.id == int(key)

        before = datetime.datetime.now()

        self.randomize_function_center()
        self.move_and_evaluate(particle)
        after = datetime.datetime.now()
        delta = after - before
        seconds = (delta.days * 86400 + delta.seconds
                + delta.microseconds / 1000000)

        # Emit the particle without changing its id:
        yield (key, particle.__getstate__())

        # Emit a message for each dependent particle:
        message = particle.make_message(self.opts.transitive_best, comparator)
        self.set_neighborhood_rand(particle, 0)
        for dep_id in self.topology.iterneighbors(particle):
            yield (repr(dep_id).encode('ascii'), message.__getstate__())

    def pso_reduce(self, key, value_iter):
        comparator = self.function.comparator
        particle = None
        messages = []
        for value in value_iter:
            record = PSOPickler.loads(value)
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
        yield particle.__getstate__()

    ##########################################################################
    # MapReduce to Find the Best Particle

    def collapse_map(self, key, value):
        yield b('0'), value

    def findbest_reduce(self, key, value_iter):
        particles = [Particle.PSOPickler.loads(value) for value in value_iter]
        best = self.findbest(particles)
        yield best.__getstate__()

    ##########################################################################
    # Helper Functions (shared by bypass and mrs implementations)

    def move_and_evaluate(self, p, swarmid=0):
        """Moves, evaluates, and updates the given particle."""
        self.set_motion_rand(p, swarmid)
        if p.iters > 0:
            newpos, newvel = self.motion(p)
        else:
            newpos, newvel = p.pos, p.vel
        self.randomize_function_center()
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

    def stop_condition(self, candidates):
        """Determines whether the stopping criteria has been met.

        In other words, whether any particle has succeeded (e.g., at 0).
        """
        for cand in candidates:
            if self.function.is_opt(cand.value):
                return True
        return False

    def cli_startup(self):
        """Checks whether the repository is dirty and reports options.

        Returns True if startup succeeded, otherwise False.
        """
        import cli

        # Check whether the repository is dirty.
        mrs_status = cli.GitStatus(mrs)
        amlpso_status = cli.GitStatus(sys.modules[__name__])
        if not self.opts.hey_im_testing:
            if amlpso_status.dirty:
                print(('Repository amlpso (%s) is dirty!'
                        '  Use --hey-im-testing if necessary.')
                        % amlpso_status.directory, file=sys.stderr)
                return False
            if mrs_status.dirty:
                print(('Repository mrs (%s) is dirty!'
                        '  Use --hey-im-testing if necessary.')
                        % mrs_status.directory, file=sys.stderr)
                return False

        # Report command-line options.
        if not self.opts.quiet:
            now = time.localtime()
            date = time.strftime("%a, %d %b %Y %H:%M:%S +0000", now)
            print('#', sys.argv[0])
            print('# Date:', date)
            print('# Git Status:')
            print('#   amlpso:', amlpso_status)
            print('#   mrs:', mrs_status)
            print("# Options:")
            for key, value in sorted(vars(self.opts).items()):
                print('#   %s = %s' % (key, value))
            self.function.master_log()
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
    FUNCTION_OFFSET = 5

    def set_motion_rand(self, p, swarmid=0):
        """Makes a Random for the given particle and saves it to `p.rand`.

        This should be used just before performing motion on the particle.

        Note that the Random depends on the particle id, iteration, and
        swarmid.  The swarmid is only needed in the subswarms case (to make
        sure that the particles in different subswarms have unique seeds), but
        it doesn't hurt the standardpso case to include it.
        """
        from mrs.main import SEED_BITS
        base = 2 ** SEED_BITS
        offset = self.MOTION_OFFSET + base * (p.id + base * (p.iters + base *
            (base * swarmid)))
        p.rand = self.random(offset)

    def set_neighborhood_rand(self, n, swarmid=0):
        """Makes a Random for the given node and saves it to `n.rand`.

        This should be used just before passing p to iterneighbors.  Note that
        depending on the PSO variant, node might be a particle, subswarm, or
        other object.

        Note that the Random depends on the particle id, iteration, and
        swarmid.  The swarmid is only needed in the subswarms case (to make
        sure that the particles in different subswarms have unique seeds), but
        it doesn't hurt the standardpso case to include it.
        """
        from mrs.main import SEED_BITS
        base = 2 ** SEED_BITS
        offset = self.NEIGHBORHOOD_OFFSET + base * (n.id + base * (n.iters +
            base * (base * swarmid)))
        n.rand = self.random(offset)

    def initialization_rand(self):
        """Returns a new Random number.

        This ensures that each run will have a unique initial swarm state.
        """
        from mrs.main import SEED_BITS
        base = 2 ** SEED_BITS
        offset = self.INITIALIZATION_OFFSET + base
        return self.random(offset)

    def function_rand(self):
        """Returns a new Random number.

        This should be used just before performing motion on the particle.

        Note that the Random depends on the particle id, iteration, and
        swarmid.  The swarmid is only needed in the subswarms case (to make
        sure that the particles in different subswarms have unique seeds), but
        it doesn't hurt the standardpso case to include it.
        """
        from mrs.main import SEED_BITS
        base = 2 ** SEED_BITS
        offset = self.FUNCTION_OFFSET + base
        return self.random(offset)

    def randomize_function_center(self):
        """Sets a random function center."""
        rand = self.function_rand()
        self.function.randomize_center(rand)


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
    parser.add_option('-v', '--verbose',
            dest='verbose', action='store_true',
            help="Print out verbose error messages",
            default=False,
            )
    parser.add_option('-i', '--iters',
            dest='iters', type='int',
            help='Number of iterations',
            default=100,
            )
    parser.add_option('-f', '--func', metavar='FUNCTION',
            dest='func', action='extend', search=['amlpso.functions'],
            help='Function to optimize',
            default='sphere.Sphere',
            )
    parser.add_option('-m', '--motion',
            dest='motion', action='extend',
            search=['amlpso.motion.basic', 'amlpso.motion'],
            help='Particle motion type',
            default='Constricted',
            )
    parser.add_option('-t', '--top', metavar='TOPOLOGY',
            dest='top', action='extend', search=['amlpso.topology'],
            help='Particle topology/sociometry',
            default='Complete',
            )
    parser.add_option('-o', '--out', metavar='OUTPUTTER',
            dest='out', action='extend', search=['amlpso.output'],
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
    parser.add_option('--split-reducemap',
            dest='split_reducemap', action='store_true',
            help='Split ReduceMap into two separate operations',
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
