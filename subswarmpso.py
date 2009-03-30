#!/usr/bin/env python

from __future__ import division
import sys, optparse, operator
from itertools import chain

import mrs
from mrs import param
import standardpso
from particle import Swarm, Particle, Message, unpack


# TODO: allow the initial set of particles to be given


class SubswarmPSO(standardpso.StandardPSO):
    def __init__(self, opts, args):
        """Mrs Setup (run on both master and slave)"""

        super(SubswarmPSO, self).__init__(opts, args)

        self.link = param.instantiate(opts, 'link')
        self.link.setup(self.function)

    ##########################################################################
    # Bypass Implementation

    def bypass_batch(self, batch):
        """Performs a single batch of PSO without MapReduce.

        Compare to the run_batch method, which uses MapReduce to do the same
        thing.
        """
        self.setup()
        comp = self.function.comparator

        # Create the Population.
        rand = self.initialization_rand(batch)
        top = self.topology
        subswarms = [Swarm(i, top.newparticles(batch, rand, i * top.num))
                for i in xrange(self.link.num)]

        # Perform PSO Iterations.  The iteration number represents the total
        # number of function evaluations that have been performed for each
        # particle by the end of the iteration.
        output = param.instantiate(self.opts, 'out')
        output.start()
        outer_iters = self.opts.iters // self.opts.subiters
        for i in xrange(1, 1 + outer_iters):
            iteration = i * self.opts.subiters
            for swarm in subswarms:
                for j in xrange(self.opts.subiters):
                    self.bypass_iteration(swarm)

            # Communication phase.
            for swarm in subswarms:
                self.set_swarm_rand(swarm)
                # TODO: try "best in swarm" as an alternative approach.
                p = swarm[0]
                for s_dep_id in self.link.iterneighbors(swarm):
                    neighbor_swarm = subswarms[s_dep_id]
                    swarm_head = neighbor_swarm[0]
                    self.set_particle_rand(swarm_head)
                    for p_dep_id in self.topology.iterneighbors(swarm_head):
                        neighbor = neighbor_swarm[p_dep_id]
                        neighbor.nbest_cand(p.pbestpos, p.pbestval, comp)
                        if self.opts.transitive_best:
                            neighbor.nbest_cand(p.nbestpos, p.nbestval, comp)
                    # FIXME: communication

            # Output phase.  (If freq is 5, output after iters 1, 6, 11, etc.)
            if not ((i-1) % output.freq):
                kwds = {}
                if 'iteration' in output.args:
                    kwds['iteration'] = iteration
                if 'particles' in output.args:
                    kwds['particles'] = particles
                if 'best' in output.args:
                    kwds['best'] = self.findbest(chain(*subswarms), comp)
                output(**kwds)
        output.finish()

        self.cleanup()

    ##########################################################################
    # MapReduce Implementation

    def run_batch(self, job, batch, tty):
        """Performs a single batch of PSO using MapReduce.

        Compare to the bypass_batch method, which does the same thing without
        using MapReduce.
        """
        self.setup()

        # Create the Population.
        rand = self.initialization_rand(batch)
        subswarms = []
        for i in xrange(self.link.num):
            initid = i * self.topology.num
            particles = self.topology.newparticles(batch, rand, initid)
            swarm = Swarm(i, particles)
            kvpair = str(i), repr(swarm)
            subswarms.append(kvpair)

        numtasks = self.opts.numtasks
        if not numtasks:
            numtasks = len(subswarms)
        new_data = job.local_data(subswarms, parter=self.mod_partition,
                splits=numtasks)

        output = param.instantiate(self.opts, 'out')
        output.start()

        # Perform iterations.  Note: we submit the next iteration while the
        # previous is being computed.  Also, the next PSO iteration depends on
        # the same data as the output phase, so they can run concurrently.
        last_pso_data = new_data
        last_out_data = None
        next_out_data = None
        last_iteration = 0
        outer_iters = self.opts.iters // self.opts.subiters
        for iteration in xrange(1, 1 + outer_iters):
            interm_data = job.map_data(last_pso_data, self.pso_map,
                    splits=numtasks, parter=self.mod_partition)
            next_pso_data = job.reduce_data(interm_data, self.pso_reduce,
                    splits=numtasks, parter=self.mod_partition)

            next_out_data = None
            if not ((iteration - 1) % output.freq):
                if 'particles' in output.args:
                    next_out_data = next_pso_data
                if 'best' in output.args:
                    # Create a new output_data MapReduce phase to find the
                    # best particle in the population.
                    collapsed_data = job.map_data(next_pso_data,
                            self.collapse_map, splits=1)
                    next_out_data = job.reduce_data(collapsed_data,
                            self.findbest_reduce, splits=1)

            waitset = set()
            if iteration > 1:
                waitset.add(last_pso_data)
            if last_out_data:
                waitset.add(last_out_data)
            while waitset:
                if tty:
                    ready = job.wait(timeout=1.0, *waitset)
                    if last_pso_data in ready:
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
                        best = self.findbest(particles, comp)
                    kwds['best'] = best
                output(**kwds)

            # Set up for the next iteration.
            last_iteration = iteration
            last_pso_data = next_pso_data
            last_out_data = next_out_data

        output.finish()

    ##########################################################################
    # Primary MapReduce

    def pso_map(self, key, value):
        swarm = unpack(value)
        assert swarm.id == int(key)
        self.bypass_iteration(swarm)

        # Emit the swarm.
        yield (key, repr(swarm))

        # Emit a message for each dependent swarm:
        self.set_swarm_rand(swarm)
        # TODO: try "best in swarm" as an alternative approach.
        particle = swarm[0]
        message = particle.make_message(self.opts.transitive_best)
        # TODO: create a Random instance for the iterneighbors method.
        for dep_id in self.topology.iterneighbors(swarm):
            yield (str(dep_id), repr(message))

    def pso_reduce(self, key, value_iter):
        comparator = self.function.comparator
        swarm = None
        messages = []
        for value in value_iter:
            record = unpack(value)
            if isinstance(record, Swarm):
                swarm = record
            elif isinstance(record, Message):
                messages.append(record)
            else:
                raise ValueError

        assert swarm, 'Missing swarm %s in the reduce step' % key

        best = self.findbest(messages, comparator)
        if best:
            swarm_head = swarm[0]
            self.set_particle_rand(swarm_head)
            for dep_id in self.topology.iterneighbors(swarm_head):
                neighbor = swarm[dep_id]
                neighbor.nbest_cand(best.position, best.value, comparator)
        yield repr(swarm)

    ##########################################################################
    # MapReduce to Find the Best Particle

    def collapse_map(self, key, value):
        """Finds the best particle in the swarm and yields it with id 0."""
        comparator = self.function.comparator
        swarm = Swarm.unpack(value)
        best = self.findbest(swarm, comparator)
        yield '0', repr(best)

    ##########################################################################
    # Helper Functions (shared by bypass and mrs implementations)

    def set_swarm_rand(self, s):
        """Makes a Random for the given particle and saves it to `p.rand`.

        Note that the Random depends on the particle id, iteration, and batch.
        """
        from mrs.impl import SEED_BITS
        base = 2 ** SEED_BITS
        offset = 3 + base * (s.id + base * (s.iters() + base * s.batches()))
        s.rand = self.random(offset)


##############################################################################
# Busywork

def update_parser(parser):
    """Adds PSO options to an OptionParser instance."""
    parser = standardpso.update_parser(parser)
    parser.add_option('-l','--link', metavar='TOPOLOGY',
            dest='link', action='extend', search=['topology'],
            help='Topology/sociometry for linking subswarms',
            default='Complete',
            )
    parser.add_option('-s','--subiters',
            dest='subiters', type='int',
            help='Number of iterations per subswarm between iterations',
            default=10,
            )
    return parser


if __name__ == '__main__':
    mrs.main(SubswarmPSO, update_parser)

# vim: et sw=4 sts=4
