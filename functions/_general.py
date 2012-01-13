from __future__ import division

import operator

from mrs.param import ParamObj, Param

class _Base(ParamObj):
    _params = dict(
            dims=Param(default=2, type='int', shortopt='-d',
                doc='Number of dimensions'),
            # center is a string because it might be multidimensional
            center=Param(default='0.5', type='str',
                doc='Relative center of function, between 0 and 1 per dim.'),
            maximize=Param(type='bool',
                doc='Maximize the function instead of minimizing.'),
            success=Param(default=10.0**(-10), type='float',
                doc='Success value (the algorithm stops when reached)'),
            )

    def master_log(self):
        """print something after the master's logs"""
        pass

    def setup(self):
        self._set_constraints(((0,0),)*self.dims)
        if self.maximize:
            self.comparator = operator.gt
        else:
            self.comparator = operator.lt

    def _set_constraints(self, constraints):
        from itertools import izip
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

    def is_opt(self, value):
        """Determines whether the value is officially optimal.

        In other words, whether the value is sufficiently low/high.
        """
        return self.comparator(value, self.success)
