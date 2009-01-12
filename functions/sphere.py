from __future__ import division
from itertools import izip
import _general

class Sphere(_general._Base):
    def setup(self, dims):
        self.dims = dims
        self._set_constraints(((-50,50),) * dims)

    def __call__(self, vec):
        return sum([(x-c)**2 for x,c in izip(vec,self.abscenter)])
