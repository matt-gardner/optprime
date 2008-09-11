from __future__ import division
from itertools import izip
import _general
from math import sqrt, sin

class Schwefel(_general._Base):
    def setup(self, dims):
        super(Schwefel,self).setup(dims)
        self._set_constraints( ((-500,500),) * self.dims )

    def __call__( self, vec ):
        s = sum([x * sin(sqrt(abs(x))) for x in vec])
        return 418.9829 * self.dims + s
