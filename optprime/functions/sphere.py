from __future__ import division

from ._base import Benchmark
from mrs.param import Param

try:
    from numpy import array
except ImportError:
    from numpypy import array

try:
    from itertools import izip as zip
except ImportError:
    pass


class Sphere(Benchmark):
    _each_constraints = (-50, 50)

    def _standard_call(self, vec):
        return (vec ** 2).sum()


class SleepSphere(Sphere):
    _params = dict(
            sleep_time=Param(type='float', default=0.0),
            )

    def __call__(self, vec):
        from time import sleep
        val = super(SleepSphere, self).__call__(vec)
        sleep(self.sleep_time)
        return val
