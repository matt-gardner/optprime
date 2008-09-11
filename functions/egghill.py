from math import sin, cos
import _general

class egghill(_general._Base):
    __slots__ = [
        'constraints',
        'alpha',
        'p'
        ]
    def setup(self, dims, alpha=2.0, p=0.5):
        self.constraints = [[-10.0,10.0],[-10.0,10.0]]
        self.alpha = alpha
        self.p = p

    def __call__( self, vec ):
        p = self.p
        a = self.alpha
        x, y = vec
        return a * ((1-p) * (sin(x) * cos(y)) - p * ((x/10)**2 + (y/10)**2))

f = egghill()
