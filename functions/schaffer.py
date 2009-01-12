from __future__ import division
from itertools import izip, imap
import _general
from math import sqrt, sin

class SchafferF6(_general._Base):
    def setup(self, dims):
        super(SchafferF6,self).setup(dims)
        self._set_constraints( ((-100,100),) * self.dims )

    def __call__( self, vec ):
        xsq = sum(imap(lambda x,c: (x-c)**2, vec, self.abscenter))
        return 0.5 + (sin(sqrt(xsq))**2 - 0.5)/((1 + 0.001*xsq)**2)

class SchafferF7(_general._Base):
    def setup(self, dims):
        super(SchafferF7,self).setup(dims)
        self._set_constraints( ((-100,100),) * self.dims )

    def __call__( self, vec ):
        xsq = sum(imap(lambda x,c: (x-c)**2, vec, self.abscenter))
        return xsq**0.25 * (sin(50*xsq**0.1)**2 + 1)
