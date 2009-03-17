from math import sin, cos
import _general

class Egg(_general._Base):
    __slots__ = ['constraints']
    def setup(self):
        super(Egg, self).setup()
        self.constraints = [[-10.0,10.0],[-10.0,10.0]]

    def __call__(self, vec):
        x, y = vec
        return sin(x) * cos(y)
