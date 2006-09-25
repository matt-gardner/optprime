from __future__ import division
from itertools import izip
from math import sqrt
import _general

class Exponential(_general._Base):
    def __init__( self, *args, **kargs):
        super(Exponential, self).__init__( *args, **kargs )
        self._set_constraints( ((-50,50),) * self.dims )

    def __call__( self, vec ):
        return 2**sqrt(sum([(x-c)**2 for x,c in izip(vec,self.abscenter)]))
