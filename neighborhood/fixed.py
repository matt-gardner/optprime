from __future__ import division
from _base import _Base
from amlpso.Vector import Vector
from mrs.param import Param

#------------------------------------------------------------------------------

class _FixedBase(_Base):
    # All fixed sociometries have this option
    _params = dict(
            selflink=Param(doc='Include self in neighborhood', default=True),
            )

#------------------------------------------------------------------------------

class Ring(_FixedBase):
    _params = dict(
            double=Param(doc='Doubly linked ring (True/False)', default=False),
            neighbors=Param(doc='Percent of neighbors to send to on each side', default=.5),
            )
            #[
            #( 'double', False, 'Doubly linked ring (True/False)' ),
            #( 'neighbors', 1, 'How many neighbors to send to on one side' )
            #]

    def iterneighbors( self, particle ):
        idx = particle.idx
        num = len(self.particles)
        num_neighbors = int(self.neighbors*num)
        for i in range(idx+1, idx+num_neighbors+1):
            yield (idx+i) % num
        if self.double:
            for i in range(idx-num_neighbors, idx):
                yield (idx-1) % num
        if self.selflink:
            yield idx

#------------------------------------------------------------------------------

class Star(_FixedBase):
    def iterneighbors( self, particle ):
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

#------------------------------------------------------------------------------

class Wheel(_FixedBase):
    def iterneighbors( self, particle ):
        if particle.idx == 0:
            # If this is the leader, emit everyone else
            for i in xrange(1,len(self.particles)):
                yield i
        else:
            # Otherwise, only connect to the leader
            yield 0

        if self.selflink:
            yield particle.idx

#------------------------------------------------------------------------------

class Rand(_FixedBase):
    _params = dict(
            probability=Param(doc='Probability of sending a message to a given particle',
                default=.5),
            )
        #[( 'probability', .5, 'Probability of sending a message to a given '
        #+ 'particle' )]
    def iterneighbors( self, particle ):
        from random import random
        # Yield all of the particles up to this one, and all after, then this
        # one last, with probability equal to self.probability.
        idx = particle.idx
        num = len(self.particles)
        for i in xrange(0,idx):
            if random() < self.probability: 
                yield i
        for i in xrange(idx+1,num):
            if random() < self.probability: 
                yield i
        if self.selflink:
            yield idx

#------------------------------------------------------------------------------

class Islands(_FixedBase):
    _params = dict(
            num_islands=Param(doc='Number of islands to use', default=5),
            )
        #[( 'num_islands', 5, 'Number of islands to use')]

    def iterneighbors( self, particle ):
        # Particles are grouped into n islands, and communicate with all members
        # on the island, and no one else
        idx = particle.idx
        num_particles = len(self.particles)
        islands = self.num_islands
        if num_particles % islands != 0:
            raise IllegalArgumentError('Uneven split between islands!')
        step_size = int(num_particles/islands)
        for i in xrange(islands):
            if idx in xrange(i*step_size, i*step_size + step_size):
                for j in xrange(i*step_size, i*step_size + step_size):
                    yield j


#------------------------------------------------------------------------------

class Dynamic(_FixedBase):
    def iterneighbors( self, particle ):
        idx = particle.idx
        num = len(self.particles)

    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
