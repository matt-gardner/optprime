from __future__ import division
from math import cos, pi
from . import _general

try:
    from itertools import izip as zip
except ImportError:
    pass


class Rastrigin(_general._Base):
    def setup(self):
        super(Rastrigin,self).setup()
        self._set_constraints(((-5.12,5.12),) * self.dims)

    def __call__(self, vec):
        s= 0.0
        for v, c in zip(vec, self.abscenter):
            s += ((v-c)**2 - 10*cos(2*pi*(v-c)) + 10)
        return s
