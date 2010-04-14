from __future__ import division
from itertools import izip
import _general

class Rosenbrock(_general._Base):
    """The Rosenbrock benchmark function

    Rosenbrock has a banana-shaped valley and is _extremely_ difficult
    to optimize.
    """
    def setup(self):
        super(Rosenbrock,self).setup()
        self._set_constraints(((-100,100),) * self.dims)

    def __call__(self, vec):
        s=0
        for i in xrange(self.dims-1):
            v = vec[i]
            v1 = vec[i+1]
            c = self.abscenter[i]
            c1 = self.abscenter[i+1]
            s += 100 * ((v1-c1) - (v-c)**2)**2 + ((v-c)-1)**2

        return s
