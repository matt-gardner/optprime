from math import sin, cos
from ._base import BaseFunction

class Egg(BaseFunction):
    def setup(self):
        super(Egg, self).setup()
        self.constraints = [[-10.0,10.0],[-10.0,10.0]]

    def __call__(self, vec):
        x, y = vec
        return sin(x) * cos(y)
