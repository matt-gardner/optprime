#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import copy
import datetime
import sys
import time

import mrs
from mrs import param

import cli
from particle import Particle, Message

try:
    range = xrange
except NameError:
    pass

# TODO: allow the initial set of particles to be given


class StandardPSO(mrs.GeneratorCallbackMR):
    def __init__(self, opts, args):
        """Mrs Setup (run on both master and slave)"""

        super(StandardPSO, self).__init__(opts, args)

        self.tmpfiles = None
        self.function = param.instantiate(opts, 'func')
        self.motion = param.instantiate(opts, 'motion')
        self.topology = param.instantiate(opts, 'top')

        self.function.setup(self.func_init_rand())
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
        particles = []
        for i in range(self.topology.num):
            rand = self.initialization_rand(i)
            p = self.topology.newparticle(i, rand)
            particles.append(p)

        # Perform PSO Iterations.  The iteration number represents the total
        # number of function evaluations that have been performed for each
        # particle by the end of the iteration.
        for iteration in range(1, 1 + self.opts.iters):
            self.bypass_iteration(particles)

            # Output phase.  (If freq is 5, output after iters 1, 6, 11, etc.)
            if self.output.freq and not ((iteration - 1) % self.output.freq):
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
        # Update position and value.
        for p in particles:
            self.move_and_evaluate(p, swarmid)

        # Communication phase.
        comp = self.function.comparator
        # With shuffled subswarms, the individual topology may be dynamic,
        # so adapt the swarm size of the topology if necessary.
        if len(particles) == self.topology.num:
            topology = self.topology
        else:
            topology = copy.copy(self.topology)
            topology.num = len(particles)
        for p in particles:
            rand = self.neighborhood_rand(p, swarmid)
            for i in topology.iterneighbors(p, rand):
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
            self.last_data = None
            self.output = param.instantiate(self.opts, 'out')
            self.output.start()

            job.default_partition = self.mod_partition
            if self.opts.numtasks:
                numtasks = self.opts.numtasks
            else:
                numtasks = self.topology.num
            job.default_reduce_tasks = numtasks
            job.default_reduce_splits = numtasks

            # Ensure that we submit enough tasks at a time.
            if self.output.freq:
                self.iterative_qmax = 2 * self.output.freq
            else:
                self.iterative_qmax = 2 * numtasks

            mrs.GeneratorCallbackMR.run(self, job)
            self.output.finish()
            return 0
        except KeyboardInterrupt as e:
            print("# INTERRUPTED")
            return 1

    def generator(self, job):
        self.out_datasets = {}
        out_data = None

        kvpairs = ((i, b'') for i in range(self.topology.num))
        start_swarm = job.local_data(kvpairs,
                key_serializer=self.int_serializer,
                value_serializer=self.raw_serializer)
        data = job.map_data(start_swarm, self.init_map)
        start_swarm.close()
        yield data, None
        self.last_data = data

        iteration = 1
        while iteration <= self.opts.iters:
            need_output = (self.output.freq and
                    (iteration - 1) % self.output.freq == 0)

            if need_output:
                num_reduce_tasks = getattr(self.opts, 'mrs__reduce_tasks', 1)
                swarm_data = job.reduce_data(self.last_data, self.pso_reduce,
                        affinity=True)
                if self.last_data not in self.out_datasets:
                    self.last_data.close()
                data = job.map_data(swarm_data, self.pso_map, affinity=True)
                if ('particles' not in self.output.args and
                        'best' not in self.output.args):
                    out_data = None
                    swarm_data.close()
                elif ('best' in self.output.args and num_reduce_tasks > 1):
                    interm = job.map_data(swarm_data, self.collapse_map,
                            splits=num_reduce_tasks)
                    swarm_data.close()
                    out_data = job.reduce_data(interm, self.findbest_reduce,
                            splits=1)
                    interm.close()
                else:
                    out_data = swarm_data

            else:
                out_data = None
                if self.opts.async:
                    async_r = {"async_start": True}
                    async_m = {"blocking_ratio": 0.75,
                            "backlink": self.last_data}
                else:
                    async_r = {}
                    async_m = {}
                if self.opts.split_reducemap:
                    swarm = job.reduce_data(self.last_data, self.pso_reduce,
                            affinity=True, **async_r)
                    if self.last_data not in self.out_datasets:
                        self.last_data.close()
                    data = job.map_data(swarm, self.pso_map, affinity=True,
                            **async_m)
                    swarm.close()
                else:
                    if self.opts.async:
                        async_rm = {"async_start": True, "blocking_ratio": 0.5,
                                    "backlink": self.last_data}
                    else:
                        async_rm = {}
                    data = job.reducemap_data(self.last_data, self.pso_reduce,
                            self.pso_map, affinity=True, **async_rm)
                    if self.last_data not in self.out_datasets:
                        self.last_data.close()

            iteration += 1
            self.last_data = data
            if out_data:
                self.out_datasets[out_data] = iteration
                yield out_data, self.output_dataset_handler

            yield data, None

    def output_dataset_handler(self, dataset):
        """Called when an output dataset is complete."""
        iteration = self.out_datasets[dataset]
        del self.out_datasets[dataset]

        if 'best' in self.output.args or 'particles' in self.output.args:
            dataset.fetchall()
            particles = [particle for _, particle in dataset.data()]
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

    @mrs.output_serializers(key=mrs.MapReduce.int_serializer)
    def init_map(self, particle_id, value):
        particle_id = int(particle_id)
        rand = self.initialization_rand(particle_id)
        p = self.topology.newparticle(particle_id, rand)

        for kvpair in self.pso_map(particle_id, p):
            yield kvpair

    @mrs.output_serializers(key=mrs.MapReduce.int_serializer)
    def pso_map(self, particle_id, particle):
        comparator = self.function.comparator
        assert particle.id == particle_id

        before = datetime.datetime.now()

        self.move_and_evaluate(particle)
        after = datetime.datetime.now()
        delta = after - before
        seconds = (delta.days * 86400 + delta.seconds
                + delta.microseconds / 1000000)

        # Emit the particle without changing its id:
        yield (particle_id, particle)

        # Emit a message for each dependent particle:
        message = particle.make_message(self.opts.transitive_best, comparator)
        rand = self.neighborhood_rand(particle, 0)
        for dep_id in self.topology.iterneighbors(particle, rand):
            yield (dep_id, message)

    def pso_reduce(self, key, value_iter):
        comparator = self.function.comparator
        particle = None
        messages = []
        for record in value_iter:
            if isinstance(record, Particle):
                particle = record
            elif isinstance(record, Message):
                messages.append(record)
            else:
                raise ValueError('Expected Particle or Message but got %s' %
                        type(record))

        best = self.findbest(messages)
        if particle is not None and best is not None:
            particle.nbest_cand(best.position, best.value, comparator)

        if particle is not None:
            yield particle
        else:
            yield best

    ##########################################################################
    # MapReduce to Find the Best Particle

    @mrs.output_serializers(key=mrs.MapReduce.int_serializer)
    def collapse_map(self, key, value):
        new_key = key % self.opts.mrs__reduce_tasks
        yield new_key, value

    def findbest_reduce(self, key, value_iter):
        particles = list(value_iter)
        assert len(particles) == self.topology.num, (
            'Only %s particles in findbest_reduce' % len(particles))

        best = self.findbest(particles)
        yield best

    ##########################################################################
    # Helper Functions (shared by bypass and mrs implementations)

    def move_and_evaluate(self, p, swarmid=0):
        """Moves, evaluates, and updates the given particle."""
        motion_rand = self.motion_rand(p, swarmid)
        if p.iters > 0:
            newpos, newvel = self.motion(p, motion_rand)
        else:
            newpos, newvel = p.pos, p.vel
        # TODO(?): value = self.function(newpos, p.rand)
        # Note that we cast to float because numpy returns numpy.float64. :(
        value = float(self.function(newpos))
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
        # Check whether the repository is dirty.
        mrs_status = cli.GitStatus(mrs)
        optprime_status = cli.GitStatus(sys.modules[__name__])
        if not self.opts.hey_im_testing:
            if optprime_status.dirty:
                print(('Repository optprime (%s) is dirty!'
                        '  Use --hey-im-testing if necessary.')
                        % optprime_status.directory, file=sys.stderr)
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
            print('#   optprime:', optprime_status)
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
    SUBITERS_OFFSET = 6

    def motion_rand(self, p, swarmid=0):
        """Makes a Random for the given particle and saves it to `p.rand`.

        This should be used just before performing motion on the particle.

        Note that the Random depends on the particle id, iteration, and
        swarmid.  The swarmid is only needed in the subswarms case (to make
        sure that the particles in different subswarms have unique seeds), but
        it doesn't hurt the standardpso case to include it.
        """
        return self.random(self.MOTION_OFFSET, p.id, p.iters, swarmid)

    def neighborhood_rand(self, n, swarmid=0):
        """Returns a Random for the given node.

        This should be used just before passing p to iterneighbors.  Note that
        depending on the PSO variant, node might be a particle, subswarm, or
        other object.

        Note that the Random depends on the particle id, iteration, and
        swarmid.  The swarmid is only needed in the subswarms case (to make
        sure that the particles in different subswarms have unique seeds), but
        it doesn't hurt the standardpso case to include it.
        """
        return self.random(self.NEIGHBORHOOD_OFFSET, n.id, n.iters, swarmid)

    def initialization_rand(self, i):
        """Returns a Random for the given particle id.

        This ensures that each run will have a unique initial swarm state.
        """
        return self.random(self.INITIALIZATION_OFFSET, i)

    def func_init_rand(self):
        """Returns a Random for function initialization."""
        return self.random(self.FUNCTION_OFFSET)

    @classmethod
    def update_parser(cls, parser):
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
                dest='func', action='extend', search=['optprime.functions'],
                help='Function to optimize',
                default='sphere.Sphere',
                )
        parser.add_option('-m', '--motion',
                dest='motion', action='extend',
                search=['optprime.motion.basic', 'optprime.motion'],
                help='Particle motion type',
                default='Constricted',
                )
        parser.add_option('-t', '--top', metavar='TOPOLOGY',
                dest='top', action='extend', search=['optprime.topology'],
                help='Particle topology/sociometry',
                default='Complete',
                )
        parser.add_option('-o', '--out', metavar='OUTPUTTER',
                dest='out', action='extend', search=['optprime.output'],
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
        parser.add_option('--async',
                dest='async', action='store_true',
                help='Run in asynchronous mode',
                default=False
                )
        parser.add_option('--hey-im-testing',
                dest='hey_im_testing', action='store_true',
                help='Ignore errors from uncommitted changes (testing only!)',
                default=False,
                )

        return parser


# vim: et sw=4 sts=4
