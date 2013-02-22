from __future__ import division
from math import sqrt, exp
from . import Benchmark

try:
    from itertools import izip as zip
except ImportError:
    pass


class Gauss(Benchmark):
    _each_constraints = (-2, 2)

    def _standard_call(self, vec):
        p = 1.0
        for x in vec:
            p *= exp(-x**2)
        return 1-p
