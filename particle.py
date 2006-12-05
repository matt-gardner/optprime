from __future__ import division
from random import Random
from Vector import Vector
import operator

#------------------------------------------------------------------------------
class Particle(object):
    """Particle for Particle Swarm Optimization.
    
    Initialization can be done by either a repr string or a position (with
    optional velocity, value, and particle id).

    Sample repr:
    14  0,1,2,3,4;4.3,-2.1;-0.4,0.7;0.0074;4.3,-2.1;0.0074;1.8,0.5;0.00023
    Interpretation of repr:
    id     deps     pos       vel    val     pbpos   pbval  gbpos   gbval

    Note: A particle's global best is only used in the MapReduce
    implementation--the neighborhood stuff for the normal PSO implementation
    needs to be converted over somehow.

    Also note: the old implementation had Particle inheriting from Vector,
    with the position and the particle being the same thing.  This is
    awkward, so we stopped doing that.  Hopefully other code won't depend
    on it too much.
    """
    def __init__( self, param, vel=None, val=None, pid=None ):
        repr_state = None
        gbest = None
        try:
            repr_id, repr_state = param.split('\t')
            pid = int(repr_id)
        except AttributeError:
            pos = param

        if repr_state is not None:
            dep_str, pos_str, vel_str, val_str, pbestpos_str, pbestval_str, \
                    gbestpos_str, gbestval_str = repr_state.split(';')
            deps = [int(field) for field in dep_str.split(',')]
            pos = [float(field) for field in pos_str.split(',')]
            vel = [float(field) for field in vel_str.split(',')]
            val = float(val_str)
            bestpos = [float(field) for field in pbestpos_str.split(',')]
            bestval = float(pbestval_str)
            gbestpos = [float(field) for field in gbestpos_str.split(',')]
            gbestval = float(gbestval_str)

        #super(Particle, self).__init__( pos )

        if pid is not None:
            self.id = pid
        else:
            self.id = self.next_id
            self.next_id += 1

        self.dims = len(pos)

        if vel is None:
            vel = [0.0] * self.dims
        if val is None:
            val = float('inf')

        if repr_state is None:
            deps = [self.id]
            bestpos = pos
            bestval = val
            gbest = self
        else:
            gbest = Particle(Vector(gbestpos), None, gbestval)

        self.deps = deps
        self.pos = Vector(pos)
        self.vel = Vector(vel)
        self.val = val
        self.bestpos = Vector(bestpos)
        self.bestval = bestval
        self.gbest = gbest

        self.stagnantcount = 0
        self.improvedcount = 0

    def copy( self ):
        p = Particle(self.pos, self.vel, self.val)
        p.bestpos = self.bestpos
        p.bestval = self.bestval
        p.gbestpos = self.gbestpos
        p.gbestval = self.gbestval
        return p

    def update( self, newpos, newvel, newval, isbetterfunc=operator.lt ):
        if newpos is None:
            self.pos += newvel
        else:
            # We use slice notation to make sure to only change the list
            # elements
            self.pos = Vector(newpos)
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
        self.pos = Vector(pos)
        self.vel = Vector(vel)
        self.val = val
        self.bestpos = Vector(pos)
        self.bestval = val
        # we should probably reset gbest here, too.
        self.resetcounts()

    def resetcounts( self ):
        self.stagnantcount = 0
        self.improvedcount = 0

    def __str__( self ):
        return "pos: %r; vel: %r; val: %r; bestpos: %r; bestval: %r" % (
                self.pos, self.vel, self.val, self.bestpos, self.bestval)

    def __repr__( self ):
        return '\t'.join((str(self.id),
                ';'.join((','.join(str(dep) for dep in self.deps),
                        ','.join(str(x) for x in self.pos),
                        ','.join(str(x) for x in self.vel),
                        str(self.val),
                        ','.join(str(x) for x in self.bestpos),
                        str(self.bestval),
                        ','.join(str(x) for x in self.gbest.bestpos),
                        str(self.gbest.bestval)))))

    next_id = 0


#------------------------------------------------------------------------------
