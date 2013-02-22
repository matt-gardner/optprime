from __future__ import division
from itertools import izip
from ._base import Benchmark

try:
    range = xrange
except NameError:
    pass


class Rosenbrock(Benchmark):
    """The Rosenbrock benchmark function

    Rosenbrock has a banana-shaped valley and is _extremely_ difficult
    to optimize.
    """
    _each_constraints = (-100, 100)

    def _standard_call(self, vec):
        part1 = ((v[1:] - v[:-1] ** 2) ** 2).sum()
        part2 = ((v[:-1] - 1) ** 2).sum()
        return 100 * part1 + part2
