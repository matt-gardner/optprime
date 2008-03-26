from __future__ import division
from amlpso.varargs import VarArgs
from itertools import izip

class _Base(VarArgs):
    _args = [
        ('center',0.5,'Center of function (0,1), can be a tuple)',lambda x:x),
        ]
    def __init__( self, *args, **kargs ):
        super(_Base,self).__init__( *args, **kargs )
        self.dims = kargs['dims']
        self._set_constraints(((0,0),)*2)

    def _set_constraints( self, constraints ):
        if not isinstance(self.center, (tuple,list)):
            self.center = (self.center,) * self.dims
        self.constraints = tuple(constraints)
        self.abscenter = [
            c*(r-l) + l
            for c, (l,r) in izip(self.center, self.constraints)
            ]

    def __call__( self, vec ):
        return 0
