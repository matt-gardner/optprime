from __future__ import division
from itertools import izip
from math import cos, pi
import _general

class Rastrigin(_general._Base):
    def __init__( self, *args, **kargs ):
        super(Rastrigin,self).__init__( *args, **kargs )
        self._set_constraints( ((-5.12,5.12),) * self.dims )

    def __call__( self, vec ):
        s= 0.0
        for v, c in izip(vec, self.abscenter):
            s += ((v-c)**2 - 10*cos(2*pi*(v-c)) + 10)
        return s
