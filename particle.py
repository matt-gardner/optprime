from __future__ import division
from random import Random
from vector import Vector
import operator


class Particle(object):
    """Particle for Particle Swarm Optimization.
    
    Initialization can be done by either a repr string or a position (with
    optional velocity, value).  If a particle id is not specified, one will be
    generated.

    Sample repr:
    p:128;10;300;4.3,-2.1;-0.4,0.7;0.0074;4.3,-2.1;0.0074;1.8,0.5;0.00023
    Interpretation of repr:
     pid bat iters  pos      vel    value   pbpos   pbval  gbpos   gbval

    Note: A particle's global best is only used in the MapReduce
    implementation--the neighborhood stuff for the normal PSO implementation
    needs to be converted over somehow.

    Also note: the old implementation had Particle inheriting from Vector,
    with the position and the particle being the same thing.  This is
    awkward, so we stopped doing that.  Hopefully other code won't depend
    on it too much.
    """
    CLASS_ID = 'p'

    def __init__(self, pid, pos, vel, value=None):
        self.pid = pid
        self.batches = 0
        self.iters = 0

        if value is None:
            value = float('inf')

        self.pos = pos
        self.vel = vel
        self.value = value
        self.pbestpos = pos
        self.pbestval = value
        self.nbestpos = pos
        self.nbestval = value

        self.rand = None

    def copy(self, newpid):
        """Performs a deep copy and returns the new Particle.

        An id must be specified for the new particle.
        """
        p = Particle(newpid, self.pos, self.vel, self.value)
        p.pbestpos = self.pbestpos
        p.pbestval = self.pbestval
        p.nbestpos = self.nbestpos
        p.nbestval = self.nbestval
        p.batches = self.batches
        p.iters = self.iters
        return p

    def make_message(self):
        """Creates a pseudo-particle which will be sent to a neighbor.

        This is used only in the Mrs PSO implementation.
        """
        return Message(self.pid, self.pos, self.value)

    def update(self, newpos, newvel, newval, isbetterfunc=operator.lt):
        self.pos = Vector(newpos)
        self.vel = Vector(newvel)
        self.value = newval
        self.iters += 1
        if isbetterfunc(self.value, self.pbestval):
            self.pbestval = newval
            self.pbestpos = newpos

    def nbest_cand(self, potential_pos, potential_val, comparator):
        """Update nbest if the given value is better than the current nbest."""
        if comparator(potential_val, self.nbestval):
            self.nbestpos = Vector(potential_pos)
            self.nbestval = potential_val
            return True
        else:
            return False

    def reset(self, pos, vel, value):
        self.pos = Vector(pos)
        self.vel = Vector(vel)
        self.value = value
        self.pbestpos = pos
        self.pbestval = value
        self.nbestpos = pos
        self.nbestval = value

    def __str__(self):
        return "pos: %r; vel: %r; value: %r; pbestpos: %r; pbestval: %r" % (
                self.pos, self.vel, self.value, self.pbestpos, self.pbestval)

    def __repr__(self):
        # Note: We don't set the dep_str from self.deps anymore.
        fields = (self.pid, self.batches, self.iters, self.pos, self.vel,
                self.value, self.pbestpos, self.pbestval, self.nbestpos,
                self.nbestval)
        strings = (repr(field) for field in fields)
        return '%s:%s' % (self.CLASS_ID, ';'.join(strings))

    @classmethod
    def unpack(cls, state):
        """Unpacks a state string, returning a new Particle.

        The state string would have been created with repr(particle).
        """
        prefix = self.CLASS_ID + ':'
        assert state.startswith(prefix)
        state = state[len(prefix):]
        (batches, pid, iters, pos, vel, value, pbestpos, pbestval,
                nbestpos, nbestval) = state.split(';')
        pos = Vector.unpack(pos)
        vel = Vector.unpack(vel)
        value = float(value)
        p = cls(pid, pos, vel, value)
        p.batches = int(batches)
        p.iters = int(iters)
        p.pbestpos = Vector.unpack(pbestpos)
        p.pbestval = float(pbestval)
        p.nbestpos = Vector.unpack(nbestpos)
        p.nbestval = float(nbestval)
        return p


class Message(object):
    """Message used to update bests in Mrs PSO.

    Sample repr:
    m:128,4.3,-2.1;.7;0.0074
    Interpretation of repr:
    sender   pos      value

    Attributes:
        position: Position of the particle.
        value: Value of the particle at `position`.
        sender: ID of the sending particle.
    """
    CLASS_ID = 'm'

    def __init__(self, sender, position, value):
        self.sender = sender
        self.position = position
        self.value = value

    @classmethod
    def unpack(self, state):
        """Unpacks a state string, returning a new Message.

        The state string would have been created with repr(particle).
        """
        prefix = self.CLASS_ID + ':'
        assert state.startswith(prefix)
        state = state[len(prefix):]
        sender, pos, value = state.split(';')
        sender = int(sender)
        pos = Vector.unpack(pos)
        value = float(value)
        return cls(sender, pos, value)

    def __repr__(self):
        fields = (self.sender, self.position, self.value)
        strings = (repr(field) for field in fields)
        return '%s:%s' % (self.CLASS_ID, ';'.join(strings))


def unpack(state):
    """Unpacks a state string, returning a Particle or Message."""
    start, _ = state.split(':', 1)
    try:
        cls = CLASS_IDS[start]
    except KeyError:
        raise ValueError('Cannot unpack a state string of class "%s".' % start)
    return cls.unpack(state)


# Valid class identifiers and their corresponding classes.
CLASSES = (Particle, Message)
CLASS_IDS = dict((cls.CLASS_ID, cls) for cls in CLASSES)
