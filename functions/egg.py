from math import sin, cos
import _general

class egg(_general._Base):
    __slots__ = ['constraints']
    def setup(self):
        self.constraints = [[-10.0,10.0],[-10.0,10.0]]

    def __call__( self, vec ):
        x, y = vec
        return sin(x) * cos(y)

f = egg()
