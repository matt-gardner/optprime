from __future__ import division
from itertools import izip
from math import cos, pi
import _general

class Rastrigin(_general._Base):
    def setup(self, dims):
        super(Rastrigin,self).setup(dims)
        self._set_constraints( ((-5.12,5.12),) * self.dims )

    def __call__( self, vec ):
        s= 0.0
        for v, c in izip(vec, self.abscenter):
            s += ((v-c)**2 - 10*cos(2*pi*(v-c)) + 10)
        return s
