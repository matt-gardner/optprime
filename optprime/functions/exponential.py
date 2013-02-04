from __future__ import division
from math import sqrt
from . import _general

try:
    from itertools import izip as zip
except ImportError:
    pass


class Exponential(_general._Base):
    def setup(self):
        super(Exponential, self).setup()
        self._set_constraints(((-50,50),) * self.dims)

    def __call__(self, vec):
        return 2**sqrt(sum([(x-c)**2 for x,c in zip(vec,self.abscenter)]))
