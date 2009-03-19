"""
cube.py

Defines a hypercube class, which should be subclassed.  The idea is that
structures that deal with various types of hypercubic interpolation (there
are many!) can just assume access to a class of this kind.
"""

from __future__ import division
from copy import deepcopy
from math import sqrt
from itertools import izip
from math import floor, ceil
import copy
from amlpso.vector import Vector


class Cube(object):
    def __init__( self, constraints, values=None ):
        """Creates a new cube object.  Requires constraints at a minimum,
        but also allows values (of unspecified format) to be passed in."""
        self.constraints = deepcopy( constraints )
        self.center = Vertex( [(cl + cr) / 2 for cl, cr in constraints] )
        self.lengths = [abs(cr - cl) for cl, cr in constraints]
        self.dims = len(constraints)
        self.vertices = self._init_vertices( values )

    def _init_vertices( self, values ):
        """This must be overridden so as to initialize and return the
        vertices of the cube.
        """
        return ()

    def itervertices( self ):
        """This must be overridden to return something akin to a vertex.
        If the cube has no concept of vertices (a regular CMAC, for
        example), then it may be viewed as having a single vertex in the
        center.
        """
        return iter(self.vertices)

    def itervertweights( self, vec ):
        """Returns an interator over pairs: (vertex index, weight) for
        interpolation.
        """
        raise NotImplementedError( "Cube.itervertweights" )

    def random_vec( self, robj ):
        """Return a random vector within the constraints of this cube.
        """
        return Vector(robj.uniform(*c) for c in self.constraints)

    def constrained_vec( self, vec, use_abs=False ):
        """Returns a new vector that is constrained to be in bounds.
        """
        vec = deepcopy(vec)
        self.constrain_vec( vec, use_abs )
        return vec

    def constrain_vec( self, vec, use_abs = False ):
        """Changes a vector to be within constraints"""
        for i, v in enumerate(vec):
            l, r = self.constraints[i]
            if use_abs:
                length = abs(l-r)
                vsize = abs(v)
                if vsize > length:
                    vec[i] = v * length/vsize
            else:
                if v < l: vec[i] = l
                if v > r: vec[i] = r

    def wrap( self, vec ):
        for i, (v, (l,r)) in enumerate(izip(vec,self.constraints)):
            span = r - l
            if v > r:
                extra = v - r
                wrap = extra - (extra // span) * span
                vec[i] = l + wrap
            elif v < l:
                extra = v - l
                # With the minus sign outside, the floor would become a ceil,
                # so we leave it inside.
                wrap = extra + (-extra // span) * span
                vec[i] = r + wrap

    def wrapped_vec( self, vec ):
        """Wraps a vector so that it appears to be on a torus.
        """
        vec = deepcopy(vec)
        self.wrap( vec )
        return vec

    def iterspace( self, resolution=None ):
        if resolution is None:
            resolution = 100
        if isinstance(resolution, (int, long)):
            resolution = (resolution,) * self.dims

        # The idea here is that we iterate over the space at the given
        # resolution, passing out exactly resolution[d] points for dimension d.

        v = [0] * self.dims
        vec = [cl for cl, cr in self.constraints]
        yield vec
        while True:
            for d in xrange(self.dims):
                v[d] += 1
                cl, cr = self.constraints[d]
                if v[d] < resolution[d]:
                    vec[d] = (v[d] / (resolution[d]-1)) * (cr - cl) + cl
                    yield vec
                    break
                else:
                    vec[d] = cl
                    v[d] = 0
            else:
                # We turned over.  We're through
                break

    def is_inside( self, vec ):
        for i, (cl, cr) in enumerate(self.constraints):
            if (vec[i] is not None) and (not (cl <= vec[i] <= cr)):
                return False
        return True

    def train_err( self, vec, valchange ):
        """Trains values based on a value error at the given location."""
        for vidx, w in self.itervertweights( vec ):
            self.vertices[vidx].value += valchange * w

    def value( self, vec ):
        """Returns the value at the location of the given vector."""
        val = 0.0
        for vidx, w in self.itervertweights(vec):
            val += self.vertices[vidx].value * w
        return val

    def __call__( self, vec ):
        # I can't just set this function here.  It would always call the
        # wrong one.
        return self.value( vec )

    def split( self, dim, distance=0.5 ):
        """Returns left and right children of this cube, given the dimension
        in which to split and the distance along that dimension (defaults to
        0.5, or halfway).
        """
        raise NotImplementedError( "Cube.split" )

    def __str__( self ):
        s = ""
        for v in self.itervertices():
            s = s + ("%r:%r\n" % (v,v.value))
        return s


class ConstantGradientExtremumMixin(object):
    def constrained_extremum( self, cvec, ismin=False ):
        con = self.constraints
        g = self.gradient()
        pt = Vertex(cvec)

        # The index of the constraint to use for max vs. min:
        # If not minimum (max), then we use the leftmost value for negative
        # gradient elements.  Otherwise, we use the rightmost.
        lesseridx = int(ismin)
        greateridx = int(not ismin)

        for i, c in enumerate(cvec):
            # If this is an unconstrained dimension, then we follow the
            # gradient in this dimension.
            # Otherwise, we leave this coordinate alone (it's constrained)
            if c is None:
                if g[i] < 0:
                    pt[i] = con[i][lesseridx]
                else:
                    pt[i] = con[i][greateridx]
        pt.value = self.value( pt )
        return pt


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
        newvert = deepcopy(self)
        if isinstance(val, (int, long, float)):
            for i, x in enumerate(self):
                newvert[i] = x * val
        else:
            raise RuntimeError, "Can't multiply a vertex by anything but a scalar"
        return newvert

    __rmul__ = __mul__

