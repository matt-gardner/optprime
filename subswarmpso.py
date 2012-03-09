#!/usr/bin/env python

from __future__ import division
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
                for j in range(self.opts.subiters):
                    self.bypass_iteration(swarm, swarm.id)

            # Communication phase.
            for swarm in subswarms:
                self.set_swarm_rand(swarm)
                # TODO: try "best in swarm" as an alternative approach.
                p = swarm[0]
                for s_dep_id in self.link.iterneighbors(swarm):
                    neighbor_swarm = subswarms[s_dep_id]
                    swarm_head = neighbor_swarm[0]
                    self.set_neighborhood_rand(swarm_head, swarm.id)
                    for p_dep_id in self.topology.iterneighbors(swarm_head):
                        neighbor = neighbor_swarm[p_dep_id]
                        neighbor.nbest_cand(p.pbestpos, p.pbestval, comp)
                        if self.opts.transitive_best:
                            neighbor.nbest_cand(p.nbestpos, p.nbestval, comp)
                    # FIXME: communication

            # Output phase.  (If freq is 5, output after iters 1, 6, 11, etc.)
            if not ((i - 1) % output.freq):
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
            self.output = param.instantiate(self.opts, 'out')
            self.output.start()

            job.default_partition = self.mod_partition
            if self.opts.numtasks:
                numtasks = self.opts.numtasks
            else:
                numtasks = self.link.num
            job.default_reduce_tasks = numtasks
            job.default_reduce_splits = numtasks

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
            data = job.map_data(start_swarm, self.init_map)
            start_swarm.close()

        elif (self.iteration - 1) % self.output.freq == 0:
            num_reduce_tasks = getattr(self.opts, 'mrs__reduce_tasks', 1)
            swarm_data = job.reduce_data(self.last_data, self.pso_reduce)
            if (self.last_data not in self.datasets and
                    self.last_data not in self.out_datasets):
                self.last_data.close()
            data = job.map_data(swarm_data, self.pso_map)
            if ('particles' not in self.output.args and
                    'best' not in self.output.args):
                out_data = None
            elif ('best' in self.output.args):
                interm = job.map_data(swarm_data, self.collapse_map,
                        splits=num_reduce_tasks)
                out_data = job.reduce_data(interm, self.findbest_reduce,
                        splits=1)
            else:
                out_data = swarm_data

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
        for i in range(self.opts.subiters):
            self.bypass_iteration(swarm, swarm.id)

        # Emit the swarm.
        yield (key, swarm.__getstate__())

        # Emit a message for each dependent swarm:
        self.set_swarm_rand(swarm)
        # TODO: try "best in swarm" as an alternative approach.
        particle = swarm[0]
        comparator = self.function.comparator
        message = particle.make_message(self.opts.transitive_best, comparator)
        for dep_id in self.link.iterneighbors(swarm):
            yield (str(dep_id), message.__getstate__())

    def pso_reduce(self, key, value_iter):
        comparator = self.function.comparator
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

        assert swarm, 'Missing swarm %s in the reduce step' % key

        best = self.findbest(messages)
        if best:
            swarm_head = swarm[0]
            # TODO: Think about whether we're setting the particle's random
            # seed correctly.  Note that we normally take some random values
            # doing motion before we take random values for neighbors.
            self.set_neighborhood_rand(swarm_head, swarm.id)
            for dep_id in self.topology.iterneighbors(swarm_head):
                neighbor = swarm[dep_id]
                neighbor.nbest_cand(best.position, best.value, comparator)
        yield swarm.__getstate__()

    ##########################################################################
    # MapReduce to Find the Best Particle

    def collapse_map(self, key, value):
        """Finds the best particle in the swarm and yields it with id 0."""
        swarm = PSOPickler.loads(value)
        best = self.findbest(swarm)
        yield '0', best.__getstate__()

    ##########################################################################
    # Helper Functions (shared by bypass and mrs implementations)

    def set_swarm_rand(self, s):
        """Makes a Random for the given particle and saves it to `p.rand`.

        Note that the Random depends on the particle id, and iteration.
        """
        from mrs.main import SEED_BITS
        base = 2 ** SEED_BITS
        offset = self.SUBSWARM_OFFSET + base * (s.id + base * (s.iters() + base))
        s.rand = self.random(offset)

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
    return parser


if __name__ == '__main__':
    mrs.main(SubswarmPSO, update_parser)

# vim: et sw=4 sts=4
