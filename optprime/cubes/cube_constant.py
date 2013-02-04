"""
cube_constant.py

Defines a hypercube that has constant values everywhere.
"""

from __future__ import division

from cube import Cube, Vertex, ConstantGradientExtremumMixin
from copy import deepcopy

class CubeConstant(Cube):
    def __init__( self, *args, **kargs ):
        super( CubeConstant, self ).__init__( *args, **kargs )
        self._gradient = [0] * self.dims

    def _init_vertices( self, values ):
        """Initialize the vertices of this cube.  The vertices are located
        at the center of the facets.
        """
        if values is None:
            values = [0]
        self.center.value = values[0]
        self.center.cube = self
        return [self.center]

    def itervertweights( self, vec ):
        yield (0, 1.0)

    def split( self, dim, distance=0.5 ):
        """Split this cube into left and right children, given the dimension
        in which to split and a distance along that dimension.  Distance
        defaults to 0.5 (halfway).
        """
        constraints = list(self.constraints)

        # Children differ in constraints.
        oldconst = constraints[dim]
        midconst = distance * oldconst[1] + (1-distance) * oldconst[0]

        constraints[dim] = oldconst[0], midconst

        lchild = CubeConstant( constraints, [deepcopy(self.center.value)] )

        constraints[dim] = midconst, oldconst[1]

        rchild = CubeConstant( constraints, [deepcopy(self.center.value)] )

        return lchild, rchild

    #-----------------------------------------------------------------------
    # Not inherited stuff
    #-----------------------------------------------------------------------

    def constrained_extremum( self, cvec, ismin=False ):
        return Vertex( self.center )

    def gradient( self ):
        return [0] * self.dims

#---------------------------------------------------------------------------
