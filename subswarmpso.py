#!/usr/bin/env python

from __future__ import division
import collections
from itertools import chain
import operator
import sys

import mrs
from mrs import param
import standardpso
from particle import Swarm, Particle, Message, PSOPickler

try:
    range = xrange
except NameError:
    pass

# TODO: allow the initial set of particles to be given


class SubswarmPSO(standardpso.StandardPSO):
    def __init__(self, opts, args):
        """Mrs Setup (run on both master and slave)"""

        super(SubswarmPSO, self).__init__(opts, args)

        self.link = param.instantiate(opts, 'link')
        self.link.setup(self.function)

    ##########################################################################
    # Bypass Implementation

    def bypass_run(self):
        """Performs PSO without MapReduce.

        Compare to the run_batch method, which uses MapReduce to do the same
        thing.
        """
        comp = self.function.comparator

        # Create the Population.
        subswarms = []
        for swarm_id in range(self.link.num):
            rand = self.initialization_rand(swarm_id)
            swarm = Swarm(swarm_id, self.topology.newparticles(rand))
            subswarms.append(swarm)

        # Perform PSO Iterations.  The iteration number represents the total
        # number of function evaluations that have been performed for each
        # particle by the end of the iteration.
        output = param.instantiate(self.opts, 'out')
        output.start()
        outer_iters = self.opts.iters // self.opts.subiters
        for i in range(1, 1 + outer_iters):
            iteration = i * self.opts.subiters
            for swarm in subswarms:
                subiters = self.subiters(swarm.id, i)
                for j in range(subiters):
                    self.bypass_iteration(swarm, swarm.id)

            # Communication phase.
            if self.opts.shuffle:
                newswarms = collections.defaultdict(list)
                for swarm in subswarms:
                    self.set_swarm_rand(swarm)
                    neighbors = list(self.link.iterneighbors(swarm))
                    for shift, particle in enumerate(swarm.shuffled()):
                        # Convert to a global particle id to ensure determinism.
                        particle.id += swarm.id * self.link.num
                        # Pick a destination swarm.
                        dest_swarm = neighbors[shift % self.link.num]
                        newswarms[dest_swarm].append(particle)
                subswarms = []
                for sid, particles in newswarms.items():
                    particles.sort(key=lambda p: p.id)
                    for pid, particle in enumerate(particles):
                        particle.id = pid
                    swarm = Swarm(sid, particles)
                    subswarms.append(swarm)
            else:
                for swarm in subswarms:
                    self.set_swarm_rand(swarm)
                    if self.opts.send_best:
                        p = self.findbest(swarm)
                    else:
                        p = swarm[0]
                    for s_dep_id in self.link.iterneighbors(swarm):
                        neighbor_swarm = subswarms[s_dep_id]
                        swarm_head = neighbor_swarm[0]
                        self.set_neighborhood_rand(swarm_head, swarm.id)
                        for p_dep_id in self.topology.iterneighbors(swarm_head):
                            neighbor = neighbor_swarm[p_dep_id]
                            neighbor.nbest_cand(p.pbestpos, p.pbestval, comp)
                            if self.opts.transitive_best:
                                neighbor.nbest_cand(p.nbestpos, p.nbestval,
                                        comp)

            # Output phase.  (If freq is 5, output after iters 1, 6, 11, etc.)
            if output.freq and not ((i - 1) % output.freq):
                kwds = {}
                if 'iteration' in output.args:
                    kwds['iteration'] = iteration
                if 'particles' in output.args:
                    kwds['particles'] = particles
                if 'best' in output.args:
                    kwds['best'] = self.findbest(chain(*subswarms))
                output(**kwds)
                if self.stop_condition(chain(*subswarms)):
                    output.finish()
                    return

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
            self.last_data = None
            self.output = param.instantiate(self.opts, 'out')
            self.output.start()

            job.default_partition = self.mod_partition
            if self.opts.numtasks:
                numtasks = self.opts.numtasks
            else:
                numtasks = self.link.num
            job.default_reduce_tasks = numtasks
            job.default_reduce_splits = numtasks

            # Ensure that we submit enough tasks at a time.
            if self.output.freq:
                self.iterative_qmax = 2 * self.output.freq
            else:
                self.iterative_qmax = 2 * numtasks

            mrs.IterativeMR.run(self, job)
            self.output.finish()
            return 0
        except KeyboardInterrupt as e:
            print("# INTERRUPTED")
            return 1

    def producer(self, job):
        outer_iters = self.opts.iters // self.opts.subiters
        if self.iteration > outer_iters:
            return []

        elif self.iteration == 0:
            self.out_datasets = {}
            self.datasets = {}
            out_data = None

            kvpairs = ((str(i), '') for i in range(self.link.num))
            start_swarm = job.local_data(kvpairs)
            data = job.map_data(start_swarm, self.init_map,
                    format=mrs.ZipWriter)
            start_swarm.close()

        elif self.output.freq and (self.iteration - 1) % self.output.freq == 0:
            num_reduce_tasks = getattr(self.opts, 'mrs__reduce_tasks', 1)
            swarm_data = job.reduce_data(self.last_data, self.pso_reduce,
                    format=mrs.ZipWriter)
            if self.last_data not in self.out_datasets:
                self.last_data.close()
            data = job.map_data(swarm_data, self.pso_map, format=mrs.ZipWriter)
            if ('particles' not in self.output.args and
                    'best' not in self.output.args):
                out_data = None
                swarm_data.close()
            elif ('best' in self.output.args):
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
                async_m = {"blocking_percent": 0.75, "backlink": self.last_data}
            else:
                async_r = {}
                async_m = {}
            if self.opts.split_reducemap:
                interm = job.reduce_data(self.last_data, self.pso_reduce,
                        format=mrs.ZipWriter, **async_r)
                if self.last_data not in self.out_datasets:
                    self.last_data.close()

                data = job.map_data(interm, self.pso_map, format=mrs.ZipWriter,
                        **async_m)
                interm.close()
            else:
                if self.opts.async:
                    async_rm = {"async_start": True, "blocking_percent": 0.5,
                            "backlink": self.last_data}
                else:
                    async_rm = {}
                data = job.reducemap_data(self.last_data, self.pso_reduce,
                        self.pso_map, format=mrs.ZipWriter, **async_rm)
                if self.last_data not in self.out_datasets:
                    self.last_data.close()

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

        if dataset in self.out_datasets:
            iteration = self.out_datasets[dataset]
            del self.out_datasets[dataset]

            if 'best' in self.output.args or 'particles' in self.output.args:
                dataset.fetchall()
                particles = []
                for key, value in dataset.data():
                    particles.append(PSOPickler.loads(value))
                if 'particles' in self.output.args:
                    particles = list(chain(*particles))
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

    def init_map(self, key, value):
        swarm_id = int(key)
        rand = self.initialization_rand(swarm_id)
        swarm = Swarm(swarm_id, self.topology.newparticles(rand))

        for kvpair in self.pso_map(key, swarm.__getstate__()):
            yield kvpair

    def pso_map(self, key, value):
        swarm = PSOPickler.loads(value)
        assert swarm.id == int(key)
        subiters = self.subiters(swarm.id, swarm.iters())
        for i in range(subiters):
            self.bypass_iteration(swarm, swarm.id)

        self.set_swarm_rand(swarm)

        if self.opts.shuffle:
            neighbors = list(self.link.iterneighbors(swarm))
            for shift, particle in enumerate(swarm.shuffled()):
                # Convert to a global particle id to ensure determinism.
                particle.id += swarm.id * self.link.num
                # Pick a destination swarm.
                dest_swarm = neighbors[shift % self.link.num]
                yield(str(dest_swarm), particle.__getstate__())
        else:
            # Emit the swarm.
            yield (key, swarm.__getstate__())

            # Emit a message for each dependent swarm.
            if self.opts.send_best:
                particle = self.findbest(swarm)
            else:
                particle = swarm[0]
            message = particle.make_message(self.opts.transitive_best,
                    self.function.comparator)
            for dep_id in self.link.iterneighbors(swarm):
                yield (str(dep_id), message.__getstate__())

    def pso_reduce(self, key, value_iter):
        if self.opts.shuffle:
            particles = []
            for value in value_iter:
                record = PSOPickler.loads(value)
                if isinstance(record, Particle):
                    particles.append(record)
                elif isinstance(record, Swarm):
                    for particle in record:
                        particles.append(particle)
                else:
                    raise ValueError
            particles.sort(key=lambda p: p.id)
            for i, particle in enumerate(particles):
                particle.id = i
            swarm = Swarm(int(key), particles)
            yield swarm.__getstate__()
        else:
            swarm = None
            messages = []
            for value in value_iter:
                record = PSOPickler.loads(value)
                if isinstance(record, Swarm):
                    swarm = record
                elif isinstance(record, Message):
                    messages.append(record)
                else:
                    raise ValueError

            best = self.findbest(messages)
            if swarm is not None and best is not None:
                swarm_head = swarm[0]
                # TODO: Think about whether we're setting the particle's random
                # seed correctly.  Note that we normally take some random values
                # doing motion before we take random values for neighbors.
                self.set_neighborhood_rand(swarm_head, swarm.id)
                for dep_id in self.topology.iterneighbors(swarm_head):
                    neighbor = swarm[dep_id]
                    neighbor.nbest_cand(best.position, best.value,
                            self.function.comparator)

            if swarm is not None:
                yield swarm.__getstate__()
            else:
                yield best.__getstate__()

    ##########################################################################
    # MapReduce to Find the Best Particle

    def collapse_map(self, key, value):
        """Finds the best particle in the swarm and yields it with id 0."""
        swarm = PSOPickler.loads(value)
        best = self.findbest(swarm)
        yield '0', best.__getstate__()

    def findbest_reduce(self, key, value_iter):
        particles = [PSOPickler.loads(value) for value in value_iter]
        assert len(particles) == self.link.num, (
            'Only %s particles in findbest_reduce' % len(particles))

        best = self.findbest(particles)
        yield best.__getstate__()

    ##########################################################################
    # Helper Functions (shared by bypass and mrs implementations)

    def set_swarm_rand(self, s):
        """Makes a Random for the given particle and saves it to `s.rand`.

        Note that the Random depends on the swarm id and iteration.
        """
        s.rand = self.random(self.SUBSWARM_OFFSET, s.id, s.iters())

    def subiters_rand(self, swarmid, iteration):
        """Makes a Random for the given particle and saves it to `p.rand`.

        Note that the Random depends on the particle id, and iteration.
        """
        return self.random(self.SUBITERS_OFFSET, swarmid, iteration)

    def subiters(self, swarmid, iteration):
        """Return the number of subiterations to be performed."""
        if self.opts.subiters_stddev == 0:
            subiters = self.opts.subiters
        else:
            subiters_rand = self.subiters_rand(swarmid, iteration)
            sample = subiters_rand.normalvariate(self.opts.subiters,
                    self.opts.subiters_stddev)
            subiters = int(round(sample))
            if subiters <= 0:
                subiters = 1
        return subiters

##############################################################################
# Busywork

def update_parser(parser):
    """Adds PSO options to an OptionParser instance."""
    parser = standardpso.update_parser(parser)
    parser.add_option('-l', '--link', metavar='TOPOLOGY',
            dest='link', action='extend', search=['topology'],
            help='Topology/sociometry for linking subswarms',
            default='Complete',
            )
    parser.add_option('-s', '--subiters',
            dest='subiters', type='int',
            help='Number of iterations per subswarm between iterations',
            default=10,
            )
    parser.add_option('--subiters-stddev',
            dest='subiters_stddev', type='float',
            help='Variation in the number of subiters per subswarm',
            default=0,
            )
    parser.add_option('--shuffle',
            dest='shuffle', action='store_true',
            help='Shuffle particles between swarms (Dynamic Multi Swarm PSO)',
            )
    parser.add_option('--send-best',
            dest='send_best', action='store_true',
            help='Send the best particle from the swarm instead of the first',
            )
    return parser


if __name__ == '__main__':
    mrs.main(SubswarmPSO, update_parser)

# vim: et sw=4 sts=4
