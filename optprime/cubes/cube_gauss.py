"""cube_gauss.py

Defines a hypercube that uses a Gaussian Radial Basis Function for its values.
"""
from __future__ import division

from cube import Cube, Vertex
from math import exp, sqrt
from itertools import izip

class CubeGauss(Cube):
    """Cube with Gaussian RBF as its basis function."""
    def __init__( self, *args, **kargs ):
        super(CubeGauss,self).__init__( *args, **kargs )

        sdevs = kargs.get('sdevs',None)
        if sdevs is None:
            sdevs = [l/2 for l in self.lengths]
        elif not isinstance(sdevs,(tuple,list)):
            sdevs = [sdevs] * self.dims
        else:
            sdevs = list(sdevs)

        self.sdevs = sdevs

    def _init_vertices( self, values ):
        if values is None:
            values = [0]
        self.center.value = values[0]
        self.center.cube = self
        return [self.center]

    def itervertweights( self, vec ):
        yield (0, 1.0)

    def value( self, vec ):
        p = self.center.value
        for v,c,s in izip(vec,self.center,self.sdevs):
            p *= exp(-((v-c)/s)**2)
        return p
