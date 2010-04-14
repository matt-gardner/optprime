from __future__ import division
from itertools import izip
import _general
from math import sqrt, sin


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
