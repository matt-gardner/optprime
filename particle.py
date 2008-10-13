from __future__ import division
from random import Random
from Vector import Vector
import operator

#------------------------------------------------------------------------------
class Particle(object):
    """Particle for Particle Swarm Optimization.
    
    Initialization can be done by either a repr string or a position (with
    optional velocity, value).  If a particle id is not specified, one will be
    generated.

    Sample repr:
    0,1,2,3,4;4.3,-2.1;-0.4,0.7;0.0074;4.3,-2.1;0.0074;1.8,0.5;0.00023
    Interpretation of repr:
       deps     pos       vel    val     pbpos   pbval  gbpos   gbval

    Alternatively, the 'deps' may be given by a string like 'all-500', in
    which case the system algorithmically figures out who the neighbors are
    (to reduce space requirements).

    Note: A particle's global best is only used in the MapReduce
    implementation--the neighborhood stuff for the normal PSO implementation
    needs to be converted over somehow.

    Also note: the old implementation had Particle inheriting from Vector,
    with the position and the particle being the same thing.  This is
    awkward, so we stopped doing that.  Hopefully other code won't depend
    on it too much.
    """
    def __init__(self, pos=None, vel=None, val=None, pid=None, state=None):
        self.id = pid
        self.iters = 0

        # TODO: move all of the `state` grabbing stuff into a separate factory
        # function, which will clean up the __init__ function dramatically.
        if state is not None:
            dep_str, pos_str, vel_str, val_str, pbestpos_str, pbestval_str, \
                    gbestpos_str, gbestval_str = state.split(';')
            self.dep_str = dep_str
            if dep_str:
                try:
                    deps = [int(field) for field in dep_str.split(',')]
                except ValueError:
                    terms = dep_str.split('-')
                    dep_mode, total_particles = terms[0], int(terms[1])
                    if dep_mode == 'all':
                        deps = xrange(total_particles)
                    else:
                        raise Exception('Unknown deps mode: %s' % dep_mode)
            else:
                deps = []
            pos = [float(field) for field in pos_str.split(',')]
            vel = [float(field) for field in vel_str.split(',')]
            val = float(val_str)
            bestpos = [float(field) for field in pbestpos_str.split(',')]
            bestval = float(pbestval_str)
            gbestpos = [float(field) for field in gbestpos_str.split(',')]
            gbestval = float(gbestval_str)

        #super(Particle, self).__init__( pos )

        #print pos
        self.dims = len(pos)

        if vel is None:
            vel = [0.0] * self.dims
        if val is None:
            val = float('inf')

        if state is None:
            deps = [self.id]
            self.dep_str = str(self.id)
            bestpos = pos
            bestval = val
            gbestpos = pos
            gbestval = val

        self.deps = deps
        self.pos = Vector(pos)
        self.vel = Vector(vel)
        self.val = val
        self.bestpos = Vector(bestpos)
        self.bestval = bestval
        self.gbestpos = Vector(gbestpos)
        self.gbestval = gbestval

        self.stagnantcount = 0
        self.improvedcount = 0

    def copy(self):
        """Performs a deep copy and returns the new Particle."""
        p = Particle(self.pos, self.vel, self.val)
        p.bestpos = Vector(self.bestpos)
        p.bestval = self.bestval
        p.gbestpos = Vector(self.gbestpos)
        p.gbestval = self.gbestval
        p.deps = list(self.deps)
        p.dep_str = self.dep_str
        p.iters = self.iters
        return p

    def make_message(self):
        """Creates a pseudo-particle which will be sent to a neighbor.

        This is used only in the Mrs PSO implementation.
        """
        p = Particle(self.pos, self.vel, self.val)
        p.bestpos = Vector(self.bestpos)
        p.bestval = self.bestval
        # We send our personal best to contribute to their global best.
        p.gbestpos = Vector(self.bestpos)
        p.gbestval = self.bestval
        p.dep_str = ''
        p.deps = []
        p.iters = self.iters
        return p

    def update(self, newpos, newvel, newval, isbetterfunc=operator.lt):
        self.pos = Vector(newpos)
        self.vel = Vector(newvel)
        self.val = newval
        self.iters += 1
        if isbetterfunc(self.val, self.bestval):
            self.stagnantcount = 0
            self.improvedcount += 1
            self.bestval = newval
            self.bestpos = newpos
        else:
            self.stagnantcount += 1
            self.improvedcount = 0

    def gbest_cand(self, potential_pos, potential_val, comparator):
        """Update gbest if the given value is better than the current gbest."""
        if comparator(potential_val, self.gbestval):
            self.gbestpos = Vector(potential_pos)
            self.gbestval = potential_val
            return True
        else:
            return False

    def is_message(self):
        """Is this a message (True) or a full-fledged particle (False)"""
        return self.dep_str == ''

    def reset(self, pos, vel, val):
        self.pos = Vector(pos)
        self.vel = Vector(vel)
        self.val = val
        self.bestpos = Vector(pos)
        self.bestval = val
        self.gbestpos = Vector(pos)
        self.gbestval = val
        self.resetcounts()

    def resetcounts(self):
        self.stagnantcount = 0
        self.improvedcount = 0

    def __str__(self):
        return "pos: %r; vel: %r; val: %r; bestpos: %r; bestval: %r" % (
                self.pos, self.vel, self.val, self.bestpos, self.bestval)

    def __repr__(self):
        # Note: We don't set the dep_str from self.deps anymore.
        return ';'.join((self.dep_str,
                        ','.join(str(x) for x in self.pos),
                        ','.join(str(x) for x in self.vel),
                        str(self.val),
                        ','.join(str(x) for x in self.bestpos),
                        str(self.bestval),
                        ','.join(str(x) for x in self.gbest.bestpos),
                        str(self.gbest.bestval)))

