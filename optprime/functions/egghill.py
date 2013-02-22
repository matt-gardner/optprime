from math import sin, cos
from ._base import BaseFunction

from numpy import array

class egghill(BaseFunction):
    def setup(self, alpha=2.0, p=0.5):
        self.dims = 2
        self.constraints = array([[-10.0,10.0],[-10.0,10.0]])
        self.alpha = alpha
        self.p = p

    def __call__(self, vec):
        p = self.p
        a = self.alpha
        x, y = vec
        return a * ((1-p) * (sin(x) * cos(y)) - p * ((x/10)**2 + (y/10)**2))
