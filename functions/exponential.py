from __future__ import division
from itertools import izip
from math import sqrt
import _general

class Exponential(_general._Base):
    def setup(self):
        super(Exponential, self).setup()
        self._set_constraints(((-50,50),) * self.dims)

    def __call__( self, vec ):
        return 2**sqrt(sum([(x-c)**2 for x,c in izip(vec,self.abscenter)]))
