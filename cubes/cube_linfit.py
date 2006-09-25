"""
cube_linfit.py

Defines a hypercube that does a sort of linear fitting algorithm.  There are
two vertices per dimension, each in the center of a facet.  Splits are
easily done, interpolation is easily performed, and training is fairly
simple, too.  All algorithms are linear in time and space complexity,
including projections.
"""

from __future__ import division

from cube import Cube, Vertex, ConstantGradientExtremumMixin
from copy import deepcopy

class CubeLinFit(Cube, ConstantGradientExtremumMixin):
    def __init__( self, *args, **kargs ):
        super( CubeLinFit, self ).__init__( *args, **kargs )

    def _init_vertices( self, values ):
        """Initialize the vertices of this cube.  The vertices are located
        at the center of the facets.
        """
        # Each vertex is created from the values list, which is a list of
        # pairs, two values for each dimension.
        if values is None:
            values = [0] * (2*self.dims)
        vertices = []
        for i in range(self.dims):
            vleft, vright = values[2*i:2*i+2]
            vtx = [(cmin + cmax) / 2 for cmin, cmax in self.constraints]
            vtleft = Vertex( vtx, value=vleft, cube=self )
            vtright = Vertex( vtx, value=vright, cube=self )

            cmin, cmax = self.constraints[i]
            vtleft[i] = cmin
            vtright[i] = cmax

            vertices.extend( (vtleft, vtright) )
        return vertices

    def itervertpairs( self ):
        for i in range(0,2*self.dims,2):
            yield self.vertices[i:i+2]

    def itervertweights( self, vec ):
        wts = self.weights(vec)
        for i, w in enumerate(wts):
            yield 2*i,(1-w)/self.dims # applies to the left vertex
            yield 2*i+1, w/self.dims     # applies to the right vertex

    def split( self, dim, distance=0.5 ):
        """Split this cube into left and right children, given the dimension
        in which to split and a distance along that dimension.  Distance
        defaults to 0.5 (halfway).
        """
        values = [deepcopy(v.value) for v in self.itervertices()]
        constraints = list(self.constraints)

        vidx = 2*dim

        # Children differ in constraints and a single value.  All values are
        # shared except the new boundary value, which is the average over
        # the split dimension.
        oldvals = values[vidx:vidx+2]
        midval = distance * oldvals[1] + (1-distance) * oldvals[0]

        oldconst = constraints[dim]
        midconst = distance * oldconst[1] + (1-distance) * oldconst[0]

        values[vidx:vidx+2] = oldvals[0], midval
        constraints[dim] = oldconst[0], midconst

        lchild = CubeLinFit( constraints, deepcopy(values) )

        values[vidx:vidx+2] = midval, oldvals[1]
        constraints[dim] = midconst, oldconst[1]

        rchild = CubeLinFit( constraints, deepcopy(values) )

        return lchild, rchild

    #-----------------------------------------------------------------------
    # Not inherited stuff
    #-----------------------------------------------------------------------

    def gradient( self ):
        d = self.dims
        l = self.lengths
        return [
            (vr.value - vl.value) / (d * l[i])
            for i, (vl, vr) in enumerate( self.itervertpairs() )
            ]

    def weights( self, vec ):
        """Returns a vector of weights.  The weights apply to the right
        value, and 1-weight applies to the left value."""
        vt = self.itervertpairs()
        return [
            (vec[i]-vtl[i])/(vtr[i]-vtl[i]) for i, (vtl,vtr) in enumerate(vt)
            ]

#---------------------------------------------------------------------------
