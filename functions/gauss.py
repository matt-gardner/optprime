from __future__ import division
from math import sqrt, exp
from itertools import izip, imap
import _general

class Gauss(_general._Base):
    def __init__( self, *args, **kargs ):
        super(Gauss,self).__init__( *args, **kargs )
        self._set_constraints( ((-2,2),) * self.dims )

    def __call__( self, vec ):
        p = 1.0
        for x,c in izip(vec,self.abscenter):
            p *= exp( -(x-c)**2 )
        return 1-p
