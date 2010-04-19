from __future__ import division
from itertools import izip
import _general
from math import cos, pi

class Bohachevsky(_general._Base):
    def setup(self):
        super(Bohachevsky, self).setup()
        self._set_constraints(((-15,15),) * self.dims)

    def __call__(self, vec):
        sum = 0.0
        for j,x in enumerate(vec):
            if j == len(vec)-1:
                i = -1
            else:
                i = j
            sum += x**2 + 2*vec[i+1]**2-.3*cos(3*pi*x) - .4*cos(4*pi*vec[i+1])
            sum += .7
        return sum
