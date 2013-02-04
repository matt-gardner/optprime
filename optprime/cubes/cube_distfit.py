"""
cube_distfit.py

We generate interpolation by using 1/d^2
"""

from __future__ import division

from cube import Cube, Vertex
from math import sqrt
from itertools import izip
from copy import deepcopy

class CubeDistFit(Cube):
    def __init__( self, *args, **kargs ):
        super( CubeDistFit, self ).__init__( *args, **kargs )

    def _init_vertices( self, values ):
        """Initialize the vertices of this cube.  The vertices are located
        at the center of the facets.
        """
        # Each vertex is created from the values list, 2 per dimension in a
        # flat list
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

    def itervertexweights( self, vec ):
        return izip(self.itervertices(),self.weights( vec ))

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

        lchild = CubeDistFit( constraints, deepcopy(values) )

        values[vidx:vidx+2] = midval, oldvals[1]
        constraints[dim] = midconst, oldconst[1]

        rchild = CubeDistFit( constraints, deepcopy(values) )

        return lchild, rchild

    #-----------------------------------------------------------------------
    # Not inherited stuff
    #-----------------------------------------------------------------------

    def distance( self, v1, v2 ):
        return sqrt(reduce(lambda x,y: x+(y[0]-y[1])**2, izip(v1,v2),0))
    
    def manhattan( self, v1, v2 ):
        return (reduce(lambda x,y: x+abs(y[0]-y[1]), izip(v1,v2),0))

    def weights( self, vec ):
        """Returns a vector of weights."""
        wts = [0] * (2*self.dims)
        for i, v in enumerate(self.itervertices()):
            d = self.distance( v, vec )
            #d = self.manhattan( v, vec )
            if d == 0.0:
                wts = [0] * (2 * self.dims)
                wts[i] = 1.0
                return wts
            wts[i] = 1/(d**2)

        s = sum(wts)
        return [w/s for w in wts]

#---------------------------------------------------------------------------
