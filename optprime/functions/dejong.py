from __future__ import division
from . import _general

try:
    from itertools import izip as zip
except ImportError:
    pass


class DeJongF4(_general._Base):
    def setup(self):
        super(DeJongF4,self).setup()
        self._set_constraints( ((-20,20),) * self.dims )

    def __call__(self, vec):
        s = 0
        for i, (v,c) in enumerate(zip(vec,self.abscenter)):
            s += (i+1) * (v-c)**4
        return s
