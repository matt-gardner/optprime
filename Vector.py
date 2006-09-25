from __future__ import division
from itertools import izip, imap
import operator
import math
from math import sqrt
from array import array

range3 = range(3)

class VectorSizeError(TypeError): pass

def MakeDoubleVector3D():
    class DoubleVector3D(array):
        def __new__( cls, *args ):
            if args:
                if isinstance(args[0],(array,cls)):
                    args = (args[0].tolist(),)
            else:
                args = ((0,0,0),)
            return array.__new__(cls, 'd', *args)

        def __neg__( self ):
            return self.__class__((-self[0],-self[1],-self[2]))

        def cross( self, other ):
            # Cross this vector with the other vector.
            return self.__class__((
                    self[1]*other[2]-other[1]*self[2],
                    other[0]*self[2]-self[0]*other[2],
                    self[0]*other[1]-other[0]*self[1],
                ))

        def distance_to( self, other ):
            return sqrt(sum([(s-o)**2 for s,o in zip(self,other)]))

        def copy( self ):
            return self.__class__(self)

        def lnorm( self, l ):
            return (self[0]**l + self[1]**l + self[2]**l)**(1/l)

        def magnitude( self ):
            return sqrt(self[0]**2 + self[1]**2 + self[2]**2)
        __abs__=magnitude

        def norm( self, mag=None ):
            if mag is None:
                mag = self.magnitude()
            s = self.copy()
            s.normalize(mag)
            return s

        def normalize( self, mag=None ):
            if mag is None:
                mag = self.magnitude()
            self[0] /= mag
            self[1] /= mag
            self[2] /= mag

    def make_scalar_opt( op, opname ):
        def isopt( self, other ):
            self[0] = op(self[0],other)
            self[1] = op(self[1],other)
            self[2] = op(self[2],other)
        setattr(DoubleVector3D, '%sscalar' % opname, isopt)

    def make_vec_opt( op, opname, kind='forward' ):
        if kind == 'forward':
            def fvopt( self, other ):
                s = self.__class__()
                s[0] = op(self[0],other[0])
                s[1] = op(self[1],other[1])
                s[2] = op(self[2],other[2])
                return s
            setattr(DoubleVector3D, '__%s__' % opname, fvopt)
        elif kind == 'reverse':
            def rvopt( self, other ):
                s = self.__class__()
                s[0] = op(other[0],self[0])
                s[1] = op(other[1],self[1])
                s[2] = op(other[2],self[2])
                return s
            setattr(DoubleVector3D, '__r%s__' % opname, rvopt)
        elif kind == 'inplace':
            def ivopt( self, other ):
                self[0] = op(self[0],other[0])
                self[1] = op(self[1],other[1])
                self[2] = op(self[2],other[2])
                return self
            setattr(DoubleVector3D, '__i%s__' % opname, ivopt)

    overrides = (
        'add',
        'sub',
        'mul',
        'div',
        'truediv',
        )

    for opname in overrides:
        op = getattr(operator, '__%s__' % opname)
        for kind in ('forward', 'reverse', 'inplace'):
            make_vec_opt( op, opname, kind )
        make_scalar_opt( op, opname )

    return DoubleVector3D

def MakeVector():
    class Vector(list):
        SEQUENCETYPES=(list,tuple,dict)

        def __init__( self, *args, **kargs ):
            super(Vector,self).__init__(*args)

        def __neg__( self ):
            return Vector([-x for x in self])

        def cross3d( self, other ):
            # Cross this vector with the other vector.
            return Vector((
                    self[1]*other[2]-other[1]*self[2],
                    other[0]*self[2]-self[0]*other[2],
                    self[0]*other[1]-other[0]*self[1],
                ))

        def distance_to( self, other ):
            s = 0.0
            for i, (x,o) in enumerate(izip(self,other)):
                s += (o-x)**2
            return sqrt(s)

        def lnorm( self, l ):
            s = 0
            for v in self:
                s += v**l
            return s**(1/l)

        def __abs__( self ):
            return sqrt(sum(imap(operator.mul,self,self)))

        magnitude = __abs__

        def norm( self, mag=None ):
            if mag is None:
                mag = self.magnitude()
            if mag > 0:
                return Vector( [x/mag for x in self] )
            else:
                return Vector( self )

        def normalize( self, mag=None ):
            if mag is None:
                mag = self.magnitude()
            if mag > 0:
                for i, x in enumerate(self):
                    self[i] = x/mag
            return self

        def copy( self ):
            return self.__class__(self)

        def dot( self, other ):
            return sum([x * y for x, y in izip(self,other)])

    def make_binary_op( klass, opname, op ):
        seqtypes = klass.SEQUENCETYPES
        def bop( self, other ):
            if isinstance(other,seqtypes):
                if len(self) != len(other):
                    raise VectorSizeError()
                return Vector([op(*pair) for pair in izip(self,other)])
            else:
                return Vector([op(x,other) for x in self])
        setattr( klass, opname, bop )

    def make_ibinary_op( klass, opname, op ):
        seqtypes = klass.SEQUENCETYPES
        def bop( self, other ):
            if isinstance(other,seqtypes):
                if len(self) != len(other):
                    raise VectorSizeError()
                for i, pair in enumerate(izip(self,other)):
                    self[i] = op(*pair)
            else:
                for i, x in enumerate(self):
                    self[i] = op(x,other)
            return self
        setattr( klass, opname[:2] + 'i' + opname[2:], bop )

    def make_rbinary_op( klass, opname, op ):
        seqtypes = klass.SEQUENCETYPES
        def bop( self, other ):
            if isinstance(other,seqtypes):
                if len(self) != len(other):
                    raise VectorSizeError()
                return Vector([op(o,x) for (x,o) in izip(self,other)])
            else:
                return Vector([op(other,x) for x in self])
        setattr( klass, opname[:2] + 'r' + opname[2:], bop )

    overrides = (
        '__add__',
        '__sub__',
        '__mul__',
        '__div__',
        '__truediv__',
        '__pow__',
        )

    for opname in overrides:
        op = getattr(operator, opname)
        make_binary_op( Vector, opname, op )
        make_ibinary_op( Vector, opname, op )
        make_rbinary_op( Vector, opname, op )

    return Vector

Vector = MakeVector()
DoubleVector3D = MakeDoubleVector3D()
