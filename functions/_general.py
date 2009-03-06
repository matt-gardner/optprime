from __future__ import division
from mrs.param import ParamObj, Param
from itertools import izip

class _Base(ParamObj):
    _params = dict(
            dims=Param(default=2, type='int', shortopt='-d',
                doc='Number of dimensions'),
            center=Param(default='0.5', type='float',
                doc='Relative center of function, between 0 and 1.'),
            maximize=Param(type='bool',
                doc='Maximize the function instead of minimizing.'),
            )

    def setup(self):
        self._set_constraints(((0,0),)*self.dims)

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
