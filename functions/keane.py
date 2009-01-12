from __future__ import division
from itertools import izip
from operator import mul
from math import cos, sqrt
import _general

class Keane(_general._Base):
    def setup(self, dims):
        super(Keane,self).setup(dims)
        self._set_constraints( ((0,10),) * self.dims )

    def __call__( self, vec ):
        if min(vec) < 0 or max(vec) > 10:
            v = 0
        elif reduce(mul, vec) < 0.75:
            v = 0
        elif sum(vec) > self.dims*7.5:
            v = 0
        else:
            s = 0
            sqs = 0
            p = 1
            for i, (v,c) in enumerate(izip(vec,self.abscenter)):
                # This has a non-symmetric feasible space, so we fix that.
                c -= 5
                # Now calculate stuff
                sqs += (i+1) * (v-c)**2
                s   += cos(v-c)**4
                p   *= cos(v-c)**2

            try:
                v = abs((s - 2*p)/sqrt(sqs))
            except ZeroDivisionError, e:
                v = 0

        return v
