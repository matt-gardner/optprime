from __future__ import division
from itertools import izip, imap
import operator
import math
from math import sqrt
from array import array


SEQUENCETYPES = (list,tuple)
OVERRIDES = (
    '__add__',
    '__sub__',
    '__mul__',
    '__div__',
    '__truediv__',
    '__pow__',
    )


class VectorSizeError(TypeError): pass


class Vector(tuple):
    """Immutable vector object."""
    def __init__(self, *args, **kargs):
        super(Vector,self).__init__(*args)

    def __neg__(self):
        return Vector([-x for x in self])

    def cross3d(self, other):
        """Cross this vector with the other vector."""
        assert len(self) == len(other) == 3
        return Vector((
                self[1]*other[2]-other[1]*self[2],
                other[0]*self[2]-self[0]*other[2],
                self[0]*other[1]-other[0]*self[1],
            ))

    def distance_to(self, other):
        s = 0.0
        for i, (x,o) in enumerate(izip(self,other)):
            s += (o-x)**2
        return sqrt(s)

    def lnorm(self, l):
        s = 0
        for v in self:
            s += v**l
        return s**(1/l)

    def __abs__(self):
        return sqrt(sum(imap(operator.mul,self,self)))

    def normalized(self, mag=None):
        if mag is None:
            mag = abs(self)
        if mag > 0:
            return Vector([x/mag for x in self])
        else:
            return self

    def dot(self, other):
        return sum([x * y for x, y in izip(self,other)])

# Add element-wise operators to the Vector class.
for opname in OVERRIDES:
    op = getattr(operator, opname)

    def binary_op(self, other):
        if isinstance(other, SEQUENCETYPES):
            if len(self) != len(other):
                raise VectorSizeError
            return Vector([op(*pair) for pair in izip(self,other)])
        else:
            return Vector([op(x,other) for x in self])

    def rbinary_op(self, other):
        if isinstance(other, SEQUENCETYPES):
            if len(self) != len(other):
                raise VectorSizeError
            return Vector([op(o,x) for (x,o) in izip(self,other)])
        else:
            return Vector([op(other,x) for x in self])

    setattr(Vector, opname, binary_op)
    setattr(Vector, '__i' + opname[2:], binary_op)
    setattr(Vector, '__r' + opname[2:], rbinary_op)


