from __future__ import division
from itertools import izip
import _general

class Sphere(_general._Base):
    def __init__( self, *args, **kargs):
        super(Sphere,self).__init__( *args, **kargs )
        self._set_constraints( ((-50,50),) * self.dims )

    def __call__( self, vec ):
        return sum([(x-c)**2 for x,c in izip(vec,self.abscenter)])
