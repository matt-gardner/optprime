#!/usr/bin/env python

from __future__ import division
import sys, operator

import mrs
import standardpso
from mrs import param
from particle import *


class SpecExPSO(standardpso.StandardPSO):

    def __init__(self, opts, args):
        """Mrs Setup (run on both master and slave)"""

        super(SpecExPSO, self).__init__(opts, args)

        self.specmethod = param.instantiate(opts, 'spec')
        pruner = param.instantiate(opts, 'pruner')

        pruner.setup(self)
        self.specmethod.setup(self, pruner)
        if self.opts.min_tokens * self.topology.num > self.opts.tokens:
            raise ValueError("There aren't enough tokens to satisfy the min "
                    "token requirement.")

    ##########################################################################
    # MapReduce Implementation

    def run_batch(self, job, batch, tty):
        """Performs a single batch of Speculative Execution PSO using MapReduce.
        """
        rand = self.initialization_rand(batch)

        particles = list(self.topology.newparticles(batch, rand))
        self.move_all(particles)
        for particle in particles:
            particle.tokens = self.opts.min_tokens
        available_tokens = self.opts.tokens - (self.opts.min_tokens *
                len(particles))
        for i in range(available_tokens):
            particles[rand.randrange(len(particles))].tokens += 1
        init_particles = []
        for p in particles:
            init_particles.append((str(p.id),repr(p)))
            neighbors = []
            for n in particles:
                self.set_neighborhood_rand(n)
                if p.id in self.topology.iterneighbors(n):
                    neighbors.append(particles[n.id])
            children = self.specmethod.generate_children(p, neighbors)
            for i, child in enumerate(children):
                key = (i+1)*self.topology.num+p.id
                init_particles.append((str(key), repr(child)))

        numtasks = self.opts.numtasks
        if not numtasks:
            numtasks = len(init_particles)
        last_swarm = job.local_data(init_particles, parter=self.mod_partition,
                splits=numtasks)

        del init_particles
        del neighbors

        output = param.instantiate(self.opts, 'out')
        output.start()

        # Perform iterations.  Note: we submit the next iteration while the
        # previous is being computed.  Also, the next PSO iteration depends on
        # the same data as the output phase, so they can run concurrently.
        last_out_data = None
        next_out_data = None
        last_iteration = 0
        for iteration in xrange(1, 1 + self.opts.iters):
            interm_data = job.map_data(last_swarm, self.sepso_map,
                    splits=numtasks, parter=self.mod_partition)
            if last_swarm != last_out_data:
                last_swarm.close()
            tmp_swarm = job.reduce_data(interm_data, self.sepso_reduce,
                    splits=numtasks, parter=self.mod_partition)
            interm_data.close()
            next_swarm = job.map_data(tmp_swarm, self.sepso_tmp_map,
                    splits=numtasks, parter=self.mod_partition)
            tmp_swarm.close()

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
                    collapsed_data.close()

            waitset = set()
            if iteration > 1:
                waitset.add(last_swarm)
            if last_out_data is not None:
                waitset.add(last_out_data)
            while waitset:
                if tty:
                    ready = job.wait(timeout=1.0, *waitset)
                    if last_swarm in ready:
                        print >>tty, "Finished iteration", last_iteration*2
                else:
                    ready = job.wait(*waitset)

                # Download output data and store as `particles`.
                if last_out_data in ready:
                    if 'best' in output.args or 'particles' in output.args:
                        last_out_data.fetchall()
                        particles = []
                        for bucket in last_out_data:
                            for reduce_id, particle in bucket:
                                # Only output real particles, not SEParticles.
                                # Also, outputs that use current particle
                                # position and value will be off on the value,
                                # because it hasn't been evaluated yet.
                                record = unpack(particle)
                                if type(record) == Particle:
                                    particles.append(record)
                    last_out_data.close()
                    last_out_data = None

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
                if self.stop_condition(particles):
                    output.finish(True)
                    return
                del kwds

            # Set up for the next iteration.
            last_iteration = iteration
            last_swarm = next_swarm
            last_out_data = next_out_data

        output.finish(False)

    ##########################################################################
    # Primary MapReduce

    def sepso_map(self, key, value):
        particle = unpack(value)
        assert particle.id == int(key)%self.topology.num
        prev_val = particle.pbestval
        self.just_evaluate(particle)

        # If we didn't update our pbest, pass a token to someone else
        if (type(particle) == Particle and particle.pbestval == prev_val
                and particle.tokens > self.opts.min_tokens):
            particle.tokens -= 1
            self.set_neighborhood_rand(particle, 1)
            pass_to = particle.rand.randrange(self.topology.num)
            yield (str(pass_to), 'token')

        # Emit the particle without changing its id:
        yield (str(particle.id), repr(particle))

        # Emit a message for each dependent particle.  In this case we're just
        # sending around the whole particle as a message, because the
        # speculative stuff needs it.
        message = particle.make_message_particle()
        for dep_id in self.specmethod.message_ids(particle):
            yield (str(dep_id), repr(message))

    def sepso_reduce(self, key, value_iter):
        # Particles are at iteration 1, as are their messages, MessageParticles.
        # SEParticles and SEMessageParticles are at iteration 2.
        # You should only ever have one Particle in the reduce phase, and all
        # of the SEParticles send themselves (not as messages) to the same
        # Reduce task as their parent Particle, so they are children.
        # MessageParticles are messages from iteration 1 particles, and
        # SEMessageParticles are messages from iteration 2 particles.
        comparator = self.function.comparator
        particle = None
        it1messages = []
        children = []
        it2messages = []
        tokens = 0
        for value in value_iter:
            if value == 'token':
                tokens += 1
                continue
            record = unpack(value)
            if type(record) == Particle:
                particle = record
            elif type(record) == SEParticle:
                children.append(record)
            elif type(record) == MessageParticle:
                it1messages.append(record)
            elif type(record) == SEMessageParticle:
                it2messages.append(record)
            else:
                raise ValueError

        assert particle, 'Missing particle %s in the reduce step' % key
        particle.tokens += tokens

        newparticle = self.specmethod.pick_child(particle, it1messages,
                children)
        newparticle.tokens = particle.tokens

        # Specmethod.pick_child updates the child's pbest, so all you need to
        # finish the second iteration is to update the nbest.
        # To update nbest, you need a set of actual neighbors at the second
        # iteration - that's why we use newparticle instead of particle here.
        it2neighbors = self.specmethod.pick_neighbor_children(newparticle,
                it1messages, it2messages)
        best = self.findbest(it2neighbors)
        if newparticle.nbest_cand(best.pbestpos, best.pbestval, comparator):
            newparticle.lastbranch[1] = best.id
        else:
            newparticle.lastbranch[1] = -1

        # We now have a finished particle at the second iteration.  We move it
        # to the third iteration, then get its neighbors at the third iteration.
        self.just_move(newparticle)

        # In a dynamic topology, the neighbors at iteration three could be
        # different than the neighbors at iteration two, so find the neighbors
        # again (newparticle is now at iteration 3, so this will pick the right
        # neighbors), then move them.  In order to move correctly, they need to
        # update their nbest.
        it3neighbors = self.specmethod.pick_neighbor_children(newparticle,
                it1messages, it2messages)
        self.specmethod.update_neighbor_nbest(it3neighbors, it1messages,
                it2messages, it3=True)
        for neighbor in it3neighbors:
            self.just_move(neighbor)

        # In a dynamic topology, you might not already know your iteration 3
        # neighbors' iteration 2 pbest.  In order to speculate correctly, you
        # need that information.  Turns out we already have it, so update nbest
        # for your iteration 3 particle with your neighbors' pbest.
        best = self.findbest(it3neighbors)
        newparticle.nbest_cand(best.pbestpos, best.pbestval, comparator)

        # Generate and yield children.  Because we don't have a ReduceMap yet,
        # we tack on a key that will get separated in the sepso_tmp_map.
        yield key+'^'+repr(newparticle)
        nchildren = self.specmethod.generate_children(newparticle, it3neighbors)
        for i, child in enumerate(nchildren):
            newkey = (i+1)*self.topology.num+int(key)
            yield str(newkey)+'^'+repr(child)

    def sepso_tmp_map(self, key, value):
        newkey, newvalue = value.split('^', 1)
        yield (newkey, newvalue)

    ##########################################################################
    # MapReduce to Find the Best Particle

    def collapse_map(self, key, value):
        yield '0', value

    def findbest_reduce(self, key, value_iter):
        particles = []
        # All of the Particles and SEParticles are one or two iterations ahead
        # of the last iteration they have evaluated, and SEParticles don't have
        # anything interesting to say.  So, to get an accurate iteration count,
        # we decrement the iteration of best by 1.
        for value in value_iter:
            record = unpack(value)
            if type(record) == Particle:
                particles.append(record)
        best = self.findbest(particles)
        best.iters -= 1
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
        self.set_motion_rand(p)
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

    parser.add_option('-s','--specmethod', metavar='SPECMETHOD',
            dest='spec', action='extend', search=['specmethod'],
            help="Speculative execution method, such as reproduce PSO exactly "
                "or make some simplifying assumptions",
            default='ReproducePSO',
            )
    parser.add_option('-p','--pruner',metavar='PRUNER',
            dest='pruner', action='extend', search=['specmethod'],
            help='Pruning method for generating speculative children',
            default='OneCompleteIteration',
            )
    parser.add_option('','--total-tokens',
            dest='tokens', type='int',
            help='Number of tokens to use (only for the TokenPruner).  This is'
            ' the difference between the number of desired particles and the'
            ' number of available processors.',
            default=0,
            )
    parser.add_option('','--min-tokens',
            dest='min_tokens', type='int',
            help='The minimum number of tokens that each particle can have. '
            'This cannot be greater than the total number of tokens available.',
            default=0,
            )

    # There are some sticky issues involved with doing this speculatively
    # that I haven't worried about.  If we ever feel like we should do this,
    # we need make some changes to the code.  Until then, disabling it is
    # better than leaving it in and having it not work.
    parser.remove_option('--transitive-best')

    return parser


if __name__ == '__main__':
    mrs.main(SpecExPSO, update_parser)

# vim: et sw=4 sts=4
