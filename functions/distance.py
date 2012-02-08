from __future__ import division
from . import _general
from mrs.param import Param

try:
    from itertools import izip as zip
except ImportError:
    pass


class Distance(_general._Base):
    _params = dict(
                norm=Param(default=2.0, type='float', 
                    doc='Default norm (2 is Euclidian)'),
            )

    def setup(self):
        super(Distance,self).setup()
        self._set_constraints(((-50,50),) * self.dims)
        #self._set_constraints( ((-2,2),) * self.dims )

    def __call__(self, vec):
        s = sum([abs(x-c)**self.norm for x,c in zip(vec,self.abscenter)])
        return s**(1/self.norm)
