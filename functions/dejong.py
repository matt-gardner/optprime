from __future__ import division
from itertools import izip
import _general

class DeJongF4(_general._Base):
    def setup(self):
        super(DeJongF4,self).setup()
        self._set_constraints( ((-20,20),) * self.dims )

    def __call__( self, vec ):
        s = 0
        for i, (v,c) in enumerate(izip(vec,self.abscenter)):
            s += (i+1) * (v-c)**4
        return s
