from __future__ import division

from . import Benchmark
from mrs.param import Param

try:
    from itertools import izip as zip
except ImportError:
    pass


class Distance(Benchmark):
    _params = dict(
            norm=Param(default=2.0, type='float',
                doc='Default norm (2 is Euclidian)'),
            )

    _each_constraints = (-50, 50)

    def _standard_call(self, vec):
        s = (abs(vec) ** self.norm).sum()
        return s ** (1/self.norm)
