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
            return cls(float(field) for field in string.split(b','))
        except ValueError:
            raise RuntimeError('Vector could not unpack "%s".' % repr(string))

    def __getstate__(self, repr=repr):
        return b','.join((repr(x).encode('ascii')) for x in self)

    def __repr__(self):
        return 'Vector(%s)' % ','.join(repr(x) for x in self)

    def __neg__(self):
        return Vector(-x for x in self)

    def __abs__(self):
        return math.sqrt(sum(x*x for x in self))


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
