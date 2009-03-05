from __future__ import division
from _base import _Base
from amlpso.Vector import Vector
from mrs.param import Param


class _FixedBase(_Base):
    # All fixed sociometries have this option
    _params = dict(
        selflink=Param(doc='Include self in neighborhood', default=True),
        )


class Ring(_FixedBase):
    _params = dict(
        double=Param(default=0, type='int', doc='Doubly linked ring'),
        neighbors=Param(default=1, type='int',
            doc='Number of neighbors to send to on each side'),
        )

    def iterneighbors(self, particle):
        idx = particle.idx
        num = len(self.particles)
        num_neighbors = self.neighbors
        for i in range(idx+1, idx+1+num_neighbors):
            yield i % num
        if self.double:
            for i in range(idx-num_neighbors, idx):
                yield i % num
        if self.selflink:
            yield idx


class Complete(_FixedBase):
    def iterneighbors(self, particle):
        # Yield all of the particles up to this one, and all after, then this
        # one last.
        idx = particle.idx
        num = len(self.particles)
        for i in xrange(0,idx):
            yield i
        for i in xrange(idx+1,num):
            yield i
        if self.selflink:
            yield idx


class Rand(_FixedBase):
    _params = dict(
        neighbors=Param(default=-1, type='int',
            doc='Number of neighbors to send to.  Default of -1 means all '+
            'particles'),
        )

    def iterneighbors(self, particle):
        from random import randint
        # Yield all of the particles up to this one, and all after, then this
        # one last, with probability equal to self.probability.
        idx = particle.idx
        num = len(self.particles)
        if (self.neighbors == -1):
            neighbors = num
        else:
            neighbors = self.neighbors
        for i in xrange(neighbors):
            yield randint(0,num-1)
        if self.selflink:
            yield idx


class Islands(_FixedBase):
    _params = dict(
        num_islands=Param(default=2, type='int',
            doc='Number of islands to use'),
        )

    def iterneighbors(self, particle):
        # Particles are grouped into n islands, and communicate with all members
        # on the island, and no one else
        idx = particle.idx
        num_particles = len(self.particles)
        islands = self.num_islands
        if num_particles % islands != 0:
            raise ValueError('Uneven split between islands! '+
            'num_particles % num_islands should be zero')
        step_size = int(num_particles/islands)
        for i in xrange(islands):
            if idx in xrange(i*step_size, i*step_size + step_size):
                for j in xrange(i*step_size, i*step_size + step_size):
                    yield j


class CommunicatingIslands(_FixedBase):
    _params = dict(
        num_islands=Param(default=4, type='int',
            doc='Number of islands to use'),
        iterations_per_communication=Param(default=50, type='int',
            doc='Number of iterations inbetween each inter-island communication'),
        type_of_communication=Param(default='Ring',
            doc='The sociometry to use at each inter-island communication (only'
            +' Ring and Random'),
        percent_communication=Param(default=1, type='float',
            doc='Percent of neighboring islands to communicate with (1 means '+
            'star)'),
        )

    def iterneighbors(self, particle):
        # Particles are grouped into n islands, and communicate with all members
        # on the island, and no one else
        idx = particle.idx
        iter = particle.iters
        num = len(self.particles)
        islands = self.num_islands
        if num % islands != 0:
            raise ValueError('Uneven split between islands! '+
            'num % num_islands should be zero')
        step_size = int(num/islands)
        if iter%self.iterations_per_communication == 0:
            # It's time to tell the other islands what's going on
            # There are probably smarter ways to do this communication, but
            # this is at least a temporary solution for the serial code
            if self.type_of_communication == 'Ring':
                 num_neighbors = int(self.percent_communication*num)
                 for i in range(idx+1, idx+num_neighbors+1):
                     yield i % num
                 if self.selflink:
                     yield idx
            elif self.type_of_communication == 'Random':
                for i in xrange(0,idx):
                    if random() < self.percent_communication: 
                        yield i
                for i in xrange(idx+1,num):
                    if random() < self.percent_communication: 
                        yield i
                if self.selflink:
                    yield idx
        else:
            for i in xrange(islands):
                if idx in xrange(i*step_size, i*step_size + step_size):
                    for j in xrange(i*step_size, i*step_size + step_size):
                        yield j





