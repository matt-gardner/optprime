from __future__ import division
from math import exp
from . import _general

try:
    range = xrange
except NameError:
    pass


class Quadratic(_general._Base):
    def setup(self):
        super(Quadratic,self).setup()
        self._set_constraints(((-100,100),) * self.dims)

    def __call__(self, vec):
        s = 0
        for i in range(self.dims):
            for j in range(self.dims):
                s += exp(-((vec[i] - vec[j])**2)) * vec[i] * vec[j]
        return s + sum(vec[:self.dims])
