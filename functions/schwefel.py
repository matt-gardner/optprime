from __future__ import division
from . import _general
from math import sqrt, sin

try:
    from itertools import izip as zip
except ImportError:
    pass


class Schwefel(_general._Base):
    """The Schwefel benchmark function.

    There are other Schwefel benchmark functions; this one is often referred
    to as "the" Schwefel function.

    Note that this is a constrained optimization problem.  Going outside
    of the range [-512.03,511.97] is "cheating", so this function doesn't
    work very well with PSO.
    """

    def setup(self):
        super(Schwefel,self).setup()
        self._set_constraints(((-500,500),) * self.dims)

    def __call__(self, vec):
        s = sum([-x * sin(sqrt(abs(x))) for x in vec])
        return 418.9829 * self.dims + s


class Schwefel221(_general._Base):
    """The Schwefel 2.21 benchmark function.
    """

    def setup(self):
        super(Schwefel221,self).setup()
        self._set_constraints(((-500,500),) * self.dims)

    def __call__(self, vec):
        return max([abs(x-c) for x,c in zip(vec, self.abscenter)])
