from __future__ import division
from amlpso.varargs import VarArgs
from mrs.param import ParamObj, Param
from itertools import izip

class _Base(ParamObj):
    _params = dict(
            center=Param(default='0.5',
                doc='Relative center of function, between 0 and 1.'),
            )

    def setup(self, dims):
        self._set_constraints(((0,0),)*2)

    def _set_constraints(self, constraints):
        self.dims = len(constraints)
        if ',' in self.center:
            center = [float(x) for x in self.center.split(',')]
        else:
            center = [float(self.center)] * self.dims
        self.constraints = tuple(constraints)
        self.abscenter = [
            c*(r-l) + l
            for c, (l,r) in izip(center, self.constraints)
            ]

    def __call__(self, vec):
        return 0
