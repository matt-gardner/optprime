from __future__ import division
from itertools import izip
import _general

class Distance(_general._Base):
    _args = [('norm', 2.0, 'Default norm (2 is Euclidian)'),]
    def __init__( self, *args, **kargs):
        super(Distance,self).__init__( *args, **kargs )
        self._set_constraints( ((-50,50),) * self.dims )
        #self._set_constraints( ((-2,2),) * self.dims )

    def __call__( self, vec ):
        s = sum([abs(x-c)**self.norm for x,c in izip(vec,self.abscenter)])
        return s**(1/self.norm)
