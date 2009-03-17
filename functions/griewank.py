from __future__ import division
from itertools import izip
import _general
from math import sqrt, cos

class Griewank(_general._Base):
    def setup(self):
        super(Griewank,self).setup()
        self._set_constraints(((-600,600),) * self.dims)

    def __call__(self, vec):
        s = 0.0
        p = 1.0
        for i, (v,c) in enumerate(izip(vec,self.abscenter)):
            s += (v-c)**2
            p *= cos((v-c)/sqrt(i+1))

        val = s/4000 - p + 1
        return val
