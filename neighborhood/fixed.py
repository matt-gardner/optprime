from __future__ import division
from _base import _Base
from amlpso.Vector import Vector

#------------------------------------------------------------------------------

class _FixedBase(_Base):
    # All fixed sociometries have this option
    _args = [
        ( 'selflink', True, 'Include self in neighborhood' ),
        ]

#------------------------------------------------------------------------------

class Ring(_FixedBase):
    _args = [( 'double', False, 'Doubly linked ring (True/False)' )]

    def iterneighbors( self, particle ):
        idx = particle.idx
        num = len(self.particles)
        yield (idx+1) % num
        if self.double:
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

class Random(_FixedBase):
    def iterneighbors( self, particle ):
        # Yield all of the particles up to this one, and all after, then this
        # one last.
        idx = particle.idx
        num = len(self.particles)
        for i in xrange(0,idx):
            #if random: continue
            yield i
        for i in xrange(idx+1,num):
            #if random: continue
            yield i
        if self.selflink:
            yield idx

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
