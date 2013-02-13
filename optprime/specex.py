#!/usr/bin/env python

from __future__ import division
import sys, operator

import mrs
from mrs import param

from . import standardpso
from .particle import *

try:
    range = xrange
except NameError:
    pass


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
    def producer(self, job):
        if self.iteration > self.opts.iters:
            return []

        elif self.iteration == 0:
            self.out_datasets = {}
            self.datasets = {}
            out_data = None

            rand = self.initialization_rand()
            particles = list(self.topology.newparticles(rand))
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
                    rand = self.neighborhood_rand(n)
                    if p.id in self.topology.iterneighbors(n, rand):
                        neighbors.append(particles[n.id])
                children = self.specmethod.generate_children(p, neighbors)
                for i, child in enumerate(children):
                    key = (i+1)*self.topology.num+p.id
                    init_particles.append((str(key), repr(child)))
            self.numtasks = self.opts.numtasks
            if not self.numtasks:
                self.numtasks = len(init_particles)
            start_swarm = job.local_data(init_particles, splits=self.numtasks,
                    parter=self.mod_partition)
            data = job.map_data(start_swarm, self.sepso_map, splits=self.numtasks,
                    parter=self.mod_partition)
            start_swarm.close()  

            del init_particles
            del neighbors
            
        elif (self.iteration - 1) % self.output.freq == 0:
            out_data = job.reduce_data(self.last_data, self.sepso_reduce, 
                splits=self.numtasks, parter=self.mod_partition)
            if (self.last_data not in self.datasets and
                    self.last_data not in self.out_datasets):
                self.last_data.close()
            data = job.map_data(out_data, self.sepso_map, splits=self.numtasks,
                    parter=self.mod_partition)

        else:
            out_data = None
            if self.opts.split_reducemap:
                interm = job.reduce_data(self.last_data, self.sepso_reduce,
                        splits=self.numtasks, parter=self.mod_partition)
                data = job.map_data(interm, self.sepso_map,
                        splits=self.numtasks, parter=self.mod_partition)
            else:
                data = job.reducemap_data(self.last_data, self.sepso_reduce,
                        self.pso_map, splits=self.numtasks,
                        parter=self.mod_partition)

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
                for reduce_id, particle in dataset.iterdata():
                    # Only output real particles, not SEParticles.
                    # Also, outputs that use current particle
                    # position and value will be off on the value,
                    # because it hasn't been evaluated yet.
                    if type(value) == Particle:
                        particles.append(value)
            if dataset != self.last_data:
                dataset.close()
            kwds = {}
            if 'iteration' in self.output.args:
                kwds['iteration'] = last_iteration
            if 'particles' in self.output.args:
                kwds['particles'] = particles
            if 'best' in self.output.args:
                print "len=", len(particles)
                kwds['best'] = self.findbest(particles)
            self.output(**kwds)
            if self.stop_condition(particles):
                self.output.success()
                return False

        return True

    ##########################################################################
    # Primary MapReduce

    def sepso_map(self, key, particle):
        assert particle.id == int(key)%self.topology.num
        prev_val = particle.pbestval
        self.just_evaluate(particle)

        # If we didn't update our pbest, pass a token to someone else
        if (type(particle) == Particle and particle.pbestval == prev_val
                and particle.tokens > self.opts.min_tokens):
            particle.tokens -= 1
            rand = self.neighborhood_rand(particle, 1)
            pass_to = rand.randrange(self.topology.num)
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
            if type(value) == Particle:
                particle = value
            elif type(value) == SEParticle:
                children.append(value)
            elif type(value) == MessageParticle:
                it1messages.append(value)
            elif type(value) == SEMessageParticle:
                it2messages.append(value)
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
            if type(value) == Particle:
                particles.append(value)
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
        motion_rand = self.motion_rand(p)
        if p.iters > 0:
            newpos, newvel = self.motion(p, motion_rand)
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


# vim: et sw=4 sts=4
