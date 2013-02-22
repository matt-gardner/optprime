from __future__ import division
from math import cos, pi

from ._base import Benchmark


class Bohachevsky(Benchmark):
    _each_constraints = (-15, 15)

    def _standard_call(self, vec):
        sum = 0.0
        for j,x in enumerate(vec):
            if j == len(vec)-1:
                i = -1
            else:
                i = j
            sum += x**2 + 2*vec[i+1]**2-.3*cos(3*pi*x) - .4*cos(4*pi*vec[i+1])
            sum += .7
        return sum
