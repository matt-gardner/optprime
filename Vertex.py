from __future__ import division
from copy import deepcopy
from math import sqrt
from itertools import izip
#---------------------------------------------------------------------------
class Vertex(list):
    # Limit the things we can change here
    __slots__ = ['value','cube']

    def train_val( self, newval ):
        """These allow us to defer to our containers when it makes sense to
        do so.
        """
        if hasattr( self, 'cube' ):
            self.cube.train_val( self, newval )
        else:
            self.value = newval

    def train_err( self, error ):
        """These allow us to defer to our containers when it makes sense to
        do so.
        """
        if hasattr( self, 'cube' ) and self.cube is not None:
            self.cube.train_err( self, error )
        else:
            self.value += error

    def dot( self, other ):
        s = 0
        for x1, x2 in izip( self, other ):
            s += x1 * x2
        return s

    def norm( self ):
        # Normalize this vector.
        div = self.magnitude()
        if div > 0:
            return self / div
        else:
            return self

    def magnitude( self ):
        return sqrt(self.dot(self))

    def __lt__( self, other ):
        return self.value < other.value

    def __eq__( self, other ):
        return self.value == other.value

    def __gt__( self, other ):
        return self.value > other.value

    def __le__( self, other ):
        return self.value <= other.value

    def __ge__( self, other ):
        return self.value >= other.value

    '''
    def __str__( self ):
        return " ".join(["%0.2f" % float(x) for x in self])
    '''

    def __init__( self, *args, **kargs ):
        # Call the original constructor (try not to mess list up too much)
        val = None
        cube = None
        if kargs.has_key('value'):
            val = kargs['value']
            del kargs['value']
        if kargs.has_key('cube'):
            cube = kargs['cube']
            del kargs['cube']

        super( Vertex, self ).__init__(*args, **kargs)

        # Add our own stuff to this
        if val is not None:
            self.value = val
        elif len(args) > 0 and isinstance(args[0], Vertex):
            self.value = args[0].value
        else:
            self.value = 0

        if cube is not None:
            self.cube = cube
        elif len(args) > 0 and isinstance(args[0], Vertex):
            self.cube = args[0].cube
        else:
            self.cube = None

    def __add__( self, other ):
        """Add two vertices
        
        other: another list (same length) or something to be added to each
            element
        """

        newvert = Vertex( self )
        if isinstance(other, (list, tuple)):
            for i, x in enumerate(newvert):
                newvert[i] += other[i]
        else:
            for i, x in enumerate(newvert):
                newvert[i] += other

        if isinstance(other, Vertex):
            newvert.value += other.value

        return newvert

    __radd__ = __add__

    def __iadd__( self, other ):
        new = self.__add__( other )
        self.value = new.value
        for i,v in enumerate(new):
            self[i] = v
        return self

    def __sub__( self, other ):
        """Subtract two vertices
        
        other: another list (same length) or something to be subtracted from
            each element
        """

        newvert = Vertex( self )
        if isinstance(other, (list, tuple)):
            for i, x in enumerate(newvert):
                newvert[i] -= other[i]
        else:
            for i, x in enumerate(newvert):
                newvert[i] -= other

        if isinstance(other, Vertex):
            newvert.value -= other.value

        return newvert

    def __isub__( self, other ):
        new = self.__sub__( other )
        self.value = new.value
        for i,v in enumerate(new):
            self[i] = v
        return self

    def __rsub__( self, other ):
        """Subtract two vertices
        
        other: another list (same length) or something to subtract each element
            from
        """

        newvert = Vertex( self )
        if isinstance(other, (list, tuple)):
            for i, x in enumerate(newvert):
                newvert[i] = other[i] - x
        else:
            for i, x in enumerate(newvert):
                newvert[i] = other - x

        if isinstance(other, Vertex):
            newvert.value = other.value - newvert.value

        return newvert

    def __neg__( self ):
        newvert = Vertex( self )
        for i, x in enumerate( newvert ):
            newvert[i] = -x

        return newvert

    def __div__( self, val ):
        newvert = deepcopy(self)
        if isinstance(val, (int, long, float)):
            for i, x in enumerate(self):
                newvert[i] = x / val
        else:
            raise RuntimeError, "Can't divide a vertex by anything but a scalar"
        return newvert

    __truediv__ = __div__
    def __idiv__( self, val ):
        newvert = self
        if isinstance(val, (int, long, float)):
            for i, x in enumerate(self):
                newvert[i] = x / val
        else:
            raise RuntimeError, "Can't divide a vertex by anything but a scalar"
        return newvert

    __itruediv__ = __idiv__

    def __pow__( self, val, mval ):
        newvert = deepcopy( self )
        if isinstance(val, (int, long, float)):
            for i, x in enumerate(self):
                newvert[i] = pow(x,val,mval)
        else:
            raise RuntimeError, "Can't use pow with anything but a scalar"
        return newvert

    def __ipow__( self, mval ):
        newvert = self
        if isinstance(val, (int, long, float)):
            for i, x in enumerate(self):
                newvert[i] = pow(x,val,mval)
        else:
            raise RuntimeError, "Can't use pow with anything but a scalar"
        return newvert

    def __mul__( self, val ):
        newvert = deepcopy(self)
        if isinstance(val, (int, long, float)):
            for i, x in enumerate(self):
                newvert[i] = x * val
        else:
            raise RuntimeError, "Can't multiply a vertex by anything but a scalar"
        return newvert

    def __imul__( self, val ):
        newvert = self
        if isinstance(val, (int, long, float)):
            for i, x in enumerate(self):
                newvert[i] = x * val
        else:
            raise RuntimeError, "Can't multiply a vertex by anything but a scalar"
        return newvert

    __rmul__ = __mul__
#---------------------------------------------------------------------------
