from __future__ import division
from itertools import izip
from . import _general

try:
    range = xrange
except NameError:
    pass


class Rosenbrock(_general._Base):
    """The Rosenbrock benchmark function

    Rosenbrock has a banana-shaped valley and is _extremely_ difficult
    to optimize.
    """
    def setup(self):
        super(Rosenbrock,self).setup()
        self._set_constraints(((-100,100),) * self.dims)

    def __call__(self, vec):
        v = vec - self.abscenter
        part1 = ((v[1:] - v[:-1] ** 2) ** 2).sum()
        part2 = ((v[:-1] - 1) ** 2).sum()
        return 100 * part1 + part2
