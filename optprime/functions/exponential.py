from __future__ import division
from math import sqrt
from ._base import BaseFunction

try:
    from itertools import izip as zip
except ImportError:
    pass


class Exponential(BaseFunction):
    _each_constraints = (-50, 50)

    def _standard_call(self, vec):
        return 2**sqrt(sum([x**2 for x in vec]))
