from __future__ import division

from ._base import Benchmark
from math import sqrt, cos

try:
    from itertools import izip as zip
except ImportError:
    pass


class Griewank(Benchmark):
    _each_constraints = (-600, 600)

    def _standard_call(self, vec):
        s = 0.0
        p = 1.0
        for i, v in enumerate(vec):
            s += v**2
            p *= cos(v/sqrt(i+1))

        val = s/4000 - p + 1
        return val
