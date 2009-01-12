from __future__ import division
from itertools import izip
from math import exp
import _general

class Quadratic(_general._Base):
    def setup(self, dims):
        super(Quadratic,self).setup(dims)
        self._set_constraints( ((-100,100),) * self.dims )

    def __call__( self, vec ):
        s = 0
        for i in xrange(self.dims):
            for j in xrange(self.dims):
                s += exp(-((vec[i] - vec[j])**2)) * vec[i] * vec[j]
        return s + sum(vec[:self.dims])
