from __future__ import division

import math
import operator


try:
    from itertools import izip as zip
except ImportError:
    pass


SEQUENCETYPES = (list,tuple)
OVERRIDES = [
    '__add__',
    '__sub__',
    '__mul__',
    '__div__',
    '__truediv__',
    '__floordiv__',
    '__pow__',
    ]


class VectorSizeError(TypeError): pass


class Vector(tuple):
    """Immutable vector object."""

    @classmethod
    def from_state(cls, string):
        """Creates a new Vector from a string representing the state.

        Note that subclasses of tuple can't provide custom pickle protocol
        methods.
        """
        try:
            return cls(float(field) for field in string.split(','))
        except ValueError:
            raise RuntimeError('Vector could not unpack "%s".' % repr(string))

    def __getstate__(self):
        #return ','.join(repr(x) for x in self).encode('ascii')
        return ','.join(repr(x) for x in self)

    def __repr__(self):
        return 'Vector(%s)' % ','.join(repr(x) for x in self)

    def __neg__(self):
        return Vector(-x for x in self)

    def cross3d(self, other):
        """Cross this vector with the other vector."""
        assert len(self) == len(other) == 3
        return Vector((
                self[1]*other[2]-other[1]*self[2],
                other[0]*self[2]-self[0]*other[2],
                self[0]*other[1]-other[0]*self[1],
            ))

    def distance_to(self, other):
        return math.sqrt(sum((o-x)**2 for x,o in zip(self, other)))

    def lnorm(self, l):
        s = 0
        for v in self:
            s += v**l
        return s**(1/l)

    def __abs__(self):
        return math.sqrt(sum(x*x for x in self))

    def normalized(self, mag=None):
        if mag is None:
            mag = abs(self)
        if mag > 0:
            return Vector(x/mag for x in self)
        else:
            return self

    def dot(self, other):
        return sum(x * y for x, y in zip(self,other))


def make_binary_op(opname):
    op = getattr(operator, opname)
    def binary_op(self, other):
        if isinstance(other, SEQUENCETYPES):
            if len(self) != len(other):
                raise VectorSizeError
            return Vector([op(*pair) for pair in zip(self, other)])
        else:
            return Vector([op(x,other) for x in self])
    return binary_op


def make_rbinary_op(opname):
    op = getattr(operator, opname)
    def rbinary_op(self, other):
        if isinstance(other, SEQUENCETYPES):
            if len(self) != len(other):
                raise VectorSizeError
            return Vector([op(o,x) for (x,o) in zip(self,other)])
        else:
            return Vector([op(other,x) for x in self])
    return rbinary_op


# Add element-wise operators to the Vector class.
for opname in OVERRIDES:
    if hasattr(operator, opname):
        binary_op = make_binary_op(opname)
        rbinary_op = make_rbinary_op(opname)
        setattr(Vector, opname, binary_op)
        setattr(Vector, '__r' + opname[2:], rbinary_op)
