"""
cube_kuhn.py

Defines a hypercube that does Kuhn triangulation for interpolation
between its vertices.
"""

from __future__ import division

from cube import Cube, Vertex, ConstantGradientExtremumMixin
from itertools import izip
from copy import deepcopy

try:
    range = xrange
except NameError:
    pass


class CubeKuhn(Cube):
    def __init__( self, *args, **kargs ):
        super( CubeKuhn, self ).__init__( *args, **kargs )
        self.origin = Vertex([cl for cl, cr in self.constraints], cube=self)

    def iterdefaultvals( self ):
        for x in range(2**self.dims):
            yield 0

    def _init_vertices( self, values ):
        """Initialize the vertices of this cube.  The vertices are located
        at the corners.  We expect to get all 2**d of them in binary
        decomposition order. (least significant bit is in the 0 position
        of the array, which seems backwards, but is correct)
        """
        if values is None:
            values = self.iterdefaultvals()
        vertices = []
        cons = self.constraints
        for vi, val in enumerate( values ):
            m = self.dims - 1
            # To switch the order of decomposition, just replace the
            # 'x' with 'self.dims - 1 - x'
            # Currently the most significant bit is in the zeroth
            # element, which works well for later sorting of stuff.
            vtx = [c[ (vi >> x) & 1 ] for x, c in enumerate(cons)]
            vtx = Vertex( vtx, value=val, cube=self )
            vertices.append( vtx )
        return vertices

    def itervertexweights( self, vec ):
        #ki = self.kuhn_info( vec )
        #return iter([(self.vertices[i],w) for i, w in izip(*ki)])
        verts = list(self.itervertices())
        weights = [0] * len(verts)
        for i, v in enumerate(verts):
            d = reduce(lambda x,y: x+(y[0]-y[1])**2, izip(vec, v),0)
            if d == 0.0:
                weights = [0] * len(verts)
                weights[i] = 1
                break
            weights[i] = 1/d
        s = sum(weights)
        weights = [w/s for w in weights]
        return izip(verts,weights)
        #ki = self.kuhn_info( vec )
        #indices, coords = ki
        #verts = [self.vertices[i] for i in indices]
        #weights = [0] * len(verts)
        #for i, v in enumerate(verts):
        #    d = reduce(lambda x,y: x+(y[0]-y[1])**2, izip(vec, v),0)
        #    if d == 0.0:
        #        weights = [0] * len(verts)
        #        weights[i] = 1
        #        break
        #    weights[i] = 1/d
        #s = sum(weights)
        #weights = [w/s for w in weights]
        #return izip(verts,weights)

    def value( self, vec ):
        """Get the value at this location."""
        val = 0
        for v, w in self.itervertexweights( vec ):
            val += v.value * w

        return val

    def split( self, dim, distance=0.5 ):
        """Split this cube into left and right children, given the dimension
        in which to split and a distance along that dimension.  Distance
        defaults to 0.5 (halfway).
        """
        # We have to create a bunch of new vertices around the midpoint of
        # things.  The two new cubes created will have their own copy of things
        # just to make everything simpler.
        cl, cr = self.constraints[dim]
        mid = cr * distance + cl * (1-distance)

        leftconstraints = list( self.constraints )
        rightconstraints = list( self.constraints )

        leftconstraints[dim] = cl, mid
        rightconstraints[dim] = mid, cr

        leftvalues = [deepcopy(x.value) for x in self.vertices]
        rightvalues = deepcopy(leftvalues)

        # Now, we iterate over all numbers 0 ... 2**d-1.  If something is on
        # the "left" of the appropriate dimension, we change the vertex of the
        # right cube.  If something is on the "right", we change the vertex of
        # the left.
        dmask = 1 << dim
        for vidx, vert in enumerate(self.vertices):
            # We always change one of left and right, never both.
            if vidx & dmask:
                # This index represents a vertex to the "right" of the divide,
                # so we change it in the "left" list to be the middle value,
                # thus chopping it off at the split.
                lv = Vertex(vert)
                lv[dim] = mid
                # We also need to set its value, so we set it to the original
                # interpolated value.
                leftvalues[vidx] = self.value( lv )
            else:
                # On the left, so chop off the right vertex coordinate.
                rv = Vertex(vert)
                rv[dim] = mid
                rightvalues[vidx] = self.value( rv )

        lchild = CubeKuhn( leftconstraints, leftvalues )
        rchild = CubeKuhn( rightconstraints, rightvalues )

        return lchild, rchild

    #-----------------------------------------------------------------------
    # Not inherited stuff
    #-----------------------------------------------------------------------
    def kuhn_info( self, vec ):
        """Returns the vertex indices in question and the barycentric
        coordinates of this point."""
        dims = self.dims
        dsize = self.lengths

        # Get this vector into the space defined by this cube (subtract
        # the origin)
        relvec = vec - self.origin

        # We need the index so that we know where stuff got sorted to
        # We also need everything to be normalized to some value between
        # 0 and 1.
        relvec = [(v/dsize[i],i) for i, v in enumerate(relvec)]

        # The triangle is found by moving in the largest direction
        # first, then the next largest, etc.
        relvec.sort()
        relvec.reverse()

        # Now find the triangle vertex indices
        indices = [0] # The corner is always involved
        vertex_idx = 0 # Start at the origin
        bary_coords = []
        bary_val = 1.0
        for x, idx in relvec:
            # Set the bit corresponding to this dimension to '1'.  This
            # constitutes walking in one dimension.
            vertex_idx |= 0x1 << idx
            indices.append( vertex_idx )
            bary_coords.append( bary_val - x )
            bary_val = x

        bary_coords.append( relvec[-1][0] )
        
        return indices, bary_coords

#---------------------------------------------------------------------------
