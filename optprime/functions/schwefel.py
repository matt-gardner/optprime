from __future__ import division

from math import sqrt, sin
from ._base import Benchmark

try:
    from itertools import izip as zip
except ImportError:
    pass


class Schwefel(Benchmark):
    """The Schwefel benchmark function.

    There are other Schwefel benchmark functions; this one is often referred
    to as "the" Schwefel function.

    Note that this is a constrained optimization problem.  Going outside
    of the range [-512.03,511.97] is "cheating", so this function doesn't
    work very well with PSO.
    """

    _each_constraints = (-500, 500)

    def _standard_call(self, vec):
        s = sum([-x * sin(sqrt(abs(x))) for x in vec])
        return 418.9829 * self.dims + s


class Schwefel221(Benchmark):
    """The Schwefel 2.21 benchmark function.
    """

    _each_constraints = (-500, 500)

    def _standard_call(self, vec):
        return max([abs(x) for x in vec])


class Schwefel12(Benchmark):
    """The Schwefel 1.2 benchmark function.

    Schwefel's Problem 1.2 is a naturally nonseparable function.  This
    function is part of the CEC 2010 benchmark suite.
    """
    _each_constraints = (-100, 100)

    def _standard_call(self, vec):
        return sum((sum(vec[:i]) ** 2) for i in xrange(self.dims))
