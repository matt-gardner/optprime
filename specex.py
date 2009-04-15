#!/usr/bin/env python

from __future__ import division
import sys, optparse, operator

import mrs
import standardpso
from mrs import param
from particle import Particle, Message, unpack, SEParticle, SEMessage


class SpecExPSO(standardpso.StandardPSO):

    def __init__(self, opts, args):
        """Mrs Setup (run on both master and slave)"""

        super(SpecExPSO, self).__init__(opts, args)

        if self.opts.reproduce_pso:
            self.specmethod = ReproducePSO
        else:
            self.specmethod = TakeBestChild

    ##########################################################################
    # MapReduce Implementation

    def run_batch(self, job, batch, tty):
        """Performs a single batch of Speculative Execution PSO using MapReduce.
        """
        self.setup()
        rand = self.initialization_rand(batch)

        particles = list(self.topology.newparticles(batch, rand))
        self.move_all(particles)
        first_iter_all = []
        for p in particles:
            first_iter_all.append(p)
            neighbors = list(particles[x] for x in 
                    self.topology.iterneighbors(p))
            for child in self.specmethod.generate_children(p, neighbors):
                first_iter_all.append(child)
        init_particles = [(str(p.id), repr(p)) for p in
                first_iter_all]

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
        last_out_data = None
        next_out_data = None
        last_iteration = 0
        for iteration in xrange(1, 1 + self.opts.iters):
            interm_data = job.map_data(last_swarm, self.sepso_map, 
                    splits=numtasks, parter=self.mod_partition)
            tmp_swarm = job.reduce_data(interm_data, self.sepso_reduce, 
                    splits=numtasks, parter=self.mod_partition)
            next_swarm = job.map_data(tmp_swarm, self.sepso_tmp_map, 
                    splits=numtasks, parter=self.mod_partition)
            return

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
                        best = self.findbest(particles, comp)
                    kwds['best'] = best
                output(**kwds)

            # Set up for the next iteration.
            last_iteration = iteration
            last_swarm = next_swarm
            last_out_data = next_out_data

        output.finish()

    ##########################################################################
    # Primary MapReduce

    def sepso_map(self, key, value):
        particle = unpack(value)
        assert particle.id == int(key)
        self.just_evaluate(particle)

        # Emit the particle without changing its id:
        yield (key, repr(particle))

        # Emit a message for each dependent particle:
        message = particle.make_message(self.opts.transitive_best)
        for dep_id in self.specmethod.itermessages(particle):
            yield (str(dep_id), repr(message))

    def sepso_reduce(self, key, value_iter):
        comparator = self.function.comparator
        particle = None
        neighbors = []
        children = []
        children_neighbors = []
        for value in value_iter:
            record = unpack(value)
            if type(record) == Particle:
                particle = record
            elif type(record) == Message:
                neighbors.append(record)
            elif type(record) == SEParticle:
                children.append(record)
            elif type(record) == SEMessage:
                children_neighbors.append(record)
            else:
                raise ValueError

        assert particle, 'Missing particle %s in the reduce step' % key

        newparticle = self.specmethod.pick_child(particle, neighbors, children)
        
        # Update the new particle's pbest.  The way we do updates, pbest gets
        # overwritten in the child if it's better than its previous pbest. But
        # what if the parent found the best position when it did its evaluation?
        # That's why we need this line.
        if Particle.isbetter(particle.pbestval, newparticle.pbestval,
                comparator):
            newparticle.pbestpos = particle.pbestpos
            newparticle.pbestval = particle.pbestval

        self.just_move(newneighbor)
        newneighbors = []
        for neighbor in neighbors:
            newneighbor = pick_neighbor_child(neighbor, children_neighbors)
            self.just_move(newneighbor)
            newneighbors.append(newneighbor)

        yield repr(newparticle)
        for child in self.specmethod.generate_children(self, newparticle, 
                newneighbors):
            yield repr(child)

    def sepso_reproduction_map2(self, key, value):
        particle = unpack(value)
        assert particle.id == int(key)

        # Emit the particle without changing its id:
        yield (key, repr(particle))

        # Emit a message for each dependent particle, but not if you're
        # speculative, because that information doesn't do any good in the
        # reproduction case.
        message = particle.make_message(self.opts.transitive_best)
        for dep_id in self.topology.iterneighbors(particle):
            yield (str(dep_id), repr(message))

    def sepso_reproduction_reduce2(self, key, value_iter):
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

        best = self.findbest(messages, comparator)
        if best:
            particle.nbest_cand(best.position, best.value, comparator)
        print 'Final particle after two iterations:',repr(particle)
        yield repr(particle)


    ##########################################################################
    # MapReduce to Find the Best Particle
    # TODO: make this work

    def collapse_map(self, key, value):
        yield '0', value

    def findbest_reduce(self, key, value_iter):
        comparator = self.function.comparator
        particles = (Particle.unpack(value) for value in value_iter)
        best = self.findbest(particles, comparator)
        yield repr(best)

    ##########################################################################
    # Helper Functions 

    def just_evaluate(self, p):
        """Evaluates the particle's position without moving it.  Updates the 
        pbest of the particle if necessary."""
        value = self.function(p.pos)
        p.update_value(value, self.function.comparator)

    def just_move(self, p):
        """Moves the particle without evaluating the function at the new
        position.  Updates the iteration count for the particle."""
        self.set_particle_rand(p)
        if p.iters > 0:
            newpos, newvel = self.motion(p)
        else:
            newpos, newvel = p.pos, p.vel
        p.update_pos(newpos, newvel, self.function.comparator)

    def move_all(self, particles):
        for p in particles:
            self.just_move(p)


##############################################################################
# Busywork

def update_parser(parser):
    """Adds PSO options to an OptionParser instance."""

    parser = standardpso.update_parser(parser)
    parser.set_default('mrs', 'Serial')
    parser.usage = parser.usage.replace('Bypass', 'Serial')

    parser.add_option('','--reproduce-pso',
            dest='reproduce_pso', action='store_true',
            help="Exactly reproduce PSO - don't use extra speculative "
                "information",
            default=False,
            )

    return parser


if __name__ == '__main__':
    mrs.main(SpecExPSO, update_parser)

# vim: et sw=4 sts=4