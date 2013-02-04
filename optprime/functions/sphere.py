from __future__ import division
from . import _general
from mrs.param import Param

try:
    from itertools import izip as zip
except ImportError:
    pass


class Sphere(_general._Base):
    def setup(self):
        super(Sphere, self).setup()
        self._set_constraints(((-50,50),) * self.dims)

    def __call__(self, vec):
        return sum([(x-c)**2 for x,c in zip(vec,self.abscenter)])


class SleepSphere(Sphere):
    _params = dict(
            sleep_time=Param(type='float', default=0.0),
            )

    def __call__(self, vec):
        from time import sleep
        val = super(SleepSphere, self).__call__(vec)
        sleep(self.sleep_time)
        return val
