from math import sin, cos
class egg(object):
    __slots__ = ['constraints']
    def __init__( self ):
        self.constraints = [[-10.0,10.0],[-10.0,10.0]]

    def __call__( self, vec ):
        x, y = vec
        return sin(x) * cos(y)

f = egg()
