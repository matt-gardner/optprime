"""
cube_vecspace.py

Defines a hypercube that is interpolated purely from d+1 basis vectors, all
eminating from the ``origin'' of the cube.  This is a very simple
interpolation method that requires no triangulation.
"""

from __future__ import division

from cube import Cube, Vertex, ConstantGradientExtremumMixin

class CubeVecSpace(Cube, ConstantGradientExtremumMixin):
    def __init__( self, *args, **kargs ):
        super( CubeVecSpace, self ).__init__( *args, **kargs )
        self.origin = self.vertices[0]

    def _init_vertices( self, values ):
        """Initialize the vertices of this cube.  They are the origin, and
        then d additional vertices, one for each dimension.
        """

        if values is None:
            values = [0] * (self.dims + 1)

        origin = [cl for cl, cr in self.constraints]

        # Get the first value -- it belongs to the origin
        vals = iter(values)
        oval = vals.next()

        # Create the first vertex
        verts = [Vertex( origin, value=oval, cube=self )]

        # Because vals is an iterator, we continue where we left off!
        for i, val in enumerate(vals):
            v = Vertex( origin, value=val, cube=self )
            v[i] = self.constraints[i][1]
            verts.append( v )

        return verts

    def train_err( self, vec, valchange ):
        if isinstance(vec,Vertex) and hasattr(vec,'cube') and self is vec.cube:
            vec.value += valchange
        else:
            raise ValueError( "Can't train a vertex not in the cube" )

    def train_val( self, vec, newval ):
        if isinstance(vec,Vertex) and hasattr(vec,'cube') and self is vec.cube:
            vec.value = newval
        else:
            raise ValueError( "Can't train a vertex not in the cube" )

    def value( self, vec ):
        """Get the value at this location."""
        if isinstance(vec,Vertex) and hasattr(vec,'cube') and self is vec.cube:
            return vec.value

        # Interpolation is done by subtracting the origin from this point and
        # dividing by the scaling factors.  This yields a vector of weights for
        # the value differences, which are used to perform interpolation.
        wpoint = vec - self.origin
        for i, x in enumerate( wpoint ):
            wpoint[i] = x / self.lengths[i]

        verts = self.itervertices()
        oval = verts.next().value
        val = oval
        for d, vtx in enumerate( verts ):
            val += wpoint[d] * (vtx.value - oval)

        return val

    def split( self, dim, distance=0.5 ):
        """Split this cube into left and right children, given the dimension
        in which to split and a distance along that dimension.  Distance
        defaults to 0.5 (halfway).
        """
        # Now we create two new children in the specified dimension

        con = list(self.constraints)

        # In the dimension in question, we change the constraint
        pair = con[dim]
        mid = distance * pair[1] + (1-distance) * pair[0]
        lpair = pair[0], mid
        rpair = mid, pair[1]

        con[dim] = lpair
        lcon = tuple(con)
        con[dim] = rpair
        rcon = tuple(con)

        # Now we have the right constraints.  With those we can generate
        # all of the needed values for the new set of nodes.
        oval = self.origin.value
        vval = self.vertices[dim+1].value
        rorigin_val = distance * vval + (1-distance) * oval

        lvalues = [v.value for v in self.vertices]
        lvalues[dim+1] = rorigin_val

        # For the "right" child values, we change every value *except* the one
        # in the dimension in which we are splitting, because that one is
        # already correct.
        vdelta = rorigin_val - self.origin.value
        rvalues = [v.value + vdelta for v in self.vertices]
        rvalues[dim+1] = self.vertices[dim+1].value

        lchild = CubeVecSpace( lcon, lvalues )
        rchild = CubeVecSpace( rcon, rvalues )

        return lchild, rchild

    #-----------------------------------------------------------------------
    # Not inherited stuff
    #-----------------------------------------------------------------------

    def gradient( self ):
        d = self.dims
        l = self.lengths
        verts = iter(self.vertices)
        oval = verts.next().value

        return [
            (v.value - oval) / l[i]
            for i, v in enumerate( verts )
            ]

#---------------------------------------------------------------------------
