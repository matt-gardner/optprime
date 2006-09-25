from __future__ import division
from random import Random
from Vector import Vector
import operator

#------------------------------------------------------------------------------
class Particle(Vector):
    def __init__( self, pos, vel=None, val=None ):
        super(Particle, self).__init__( pos )

        self.dims = len(pos)
        self.pos = self

        if vel is None:
            vel = [0.0] * self.dims
        if val is None:
            val = 0.0

        self.vel = Vector(vel)
        self.val = val
        self.bestpos = Vector(self.pos)
        self.bestval = self.val

        self.stagnantcount = 0
        self.improvedcount = 0

    def copy( self ):
        p = Particle(self.pos, self.vel, self.val)
        p.bestpos = self.bestpos
        p.bestval = self.bestval
        return p

    def update( self, newpos, newvel, newval, isbetterfunc=operator.lt ):
        if newpos is None:
            self += newvel
        else:
            # We use slice notation to make sure to only change the list
            # elements
            self[:] = newpos
        self.vel = Vector(newvel)
        self.val = newval
        if isbetterfunc(self.val, self.bestval):
            self.stagnantcount = 0
            self.improvedcount += 1
            self.bestval = newval
            self.bestpos = newpos
        else:
            self.stagnantcount += 1
            self.improvedcount = 0

    def reset( self, pos, vel, val ):
        self[:] = pos
        self.vel = Vector(vel)
        self.val = val
        self.bestpos = Vector(pos)
        self.bestval = val
        self.resetcounts()

    def resetcounts( self ):
        self.stagnantcount = 0
        self.improvedcount = 0

    def __str__( self ):
        return "pos: %r; vel: %r; val: %r; bestpos: %r; bestval: %r" % (
                self.pos, self.vel, self.val, self.bestpos, self.bestval)

#------------------------------------------------------------------------------
