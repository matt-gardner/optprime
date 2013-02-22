from __future__ import division
from ._base import Benchmark

try:
    from itertools import izip as zip
except ImportError:
    pass


class DeJongF4(Benchmark):
    _each_constraints = (-20, 20)

    def _standard_call(self, vec):
        s = 0
        for i, v in enumerate(vec):
            s += (i+1) * v**4
        return s
