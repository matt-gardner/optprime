"""kdtrie.py

Requires Python 2.3

This implements the kd-trie structure in such a way as to do simple
interpolation over the interior of a hypercube (no triangulation).
"""

from __future__ import division
from cubes.cube import Vertex
from copy import deepcopy

try:
    range = xrange
except NameError:
    pass


class kdTrie(object):
    def __init__( self, constraints, cubecls, initvals ):
        """Initialize the kdTrie object

        arguments:
            constraints -- a list of constraint pairs
            cubecls -- a factory for creating cubes
        """
        self.dims = len(constraints)
        self.constraints = tuple(constraints)

        rootcube = cubecls( constraints, initvals )

        self.root = kdNode( rootcube )

    def gridify( self, splits, node=None ):
        """Split to create a grid, given how many times to split each
        dimension.

        arguments:
            splits -- either a scalar or an array of per-dim splits
            node=None -- defaults to root.  Defines which node to gridify
        """
        if type(splits) in (int, long):
            splits = [splits] * self.dims

        if node is None:
            node = self.root

        # Now we have a node and some splits.  Pick the first dimension
        # in which we should do splits, and go (recursively)
        for d in range(self.dims):
            if splits[d] > 1:
                dim = d
                break
        else:
            # We're done.  No useful splits left to do.
            return

        if 0 == splits[dim] % 2:
            # If divisible by 2, what to do is simple: we split
            # Create a copy, and modify that copy
            splits = list(splits)
            splits[dim] /= 2

            # Now split and call again
            node.split( dim )
            self.gridify( splits, node.lchild )
            self.gridify( splits, node.rchild )
        else:
            # If not divisible by 2, things are a bit trickier:
            lsplits = list(splits)
            rsplits = list(splits)
            lsplits[dim] //= 2
            rsplits[dim] = lsplits[dim] + 1

            # Now we are going to pull some magic out of our hats here.
            # The splits don't have to occur in the middle, so we calculate
            # where they do occur.  The distance is a percentage from left
            # to right, so it's easy to calculate as a ratio:
            dist = lsplits[dim] / splits[dim]

            node.split( dim, dist )
            self.gridify( lsplits, node.lchild )
            self.gridify( rsplits, node.rchild )

    def iterleaves( self, node=None ):
        """iterator generator for dumping out each leaf node."""
        if node is None:
            node = self.root

        if node.splitdim is None:
            yield node
        else:
            for x in self.iterleaves( node.lchild ):
                yield x
            for x in self.iterleaves( node.rchild ):
                yield x

    def iternodes( self, node=None ):
        if node is None:
            node = self.root

        if node.lchild is not None:
            for x in self.iternodes( node.lchild ):
                yield x
        yield node
        if node.rchild is not None:
            for x in self.iternodes( node.rchild ):
                yield x

    def itervertices( self ):
        """iterator generator for dumping the vertices of this tree
        in order."""
        for node in self.iterleaves():
            for v in node.cube.itervertices():
                yield v

    def iternodequery( self, query ):
        """iterator generator for a range query.  Returns each child node that
        satisfies the given constraint in turn.
        """
        nodes_to_try = [self.root]
        nodes = []

        while nodes_to_try:
            node = nodes_to_try.pop()

            if node.splitdim is None:
                yield( node )
            else:
                # If this is not a leaf, we need to determine where we will
                # continue our search.  If the split dimension is a wildcard,
                # we'll continue our search at both children, otherwise we'll
                # continue with the appropriate child.
                if query[node.splitdim] is None:
                    nodes_to_try.append( node.lchild )
                    nodes_to_try.append( node.rchild )
                else:
                    if query[node.splitdim] < node.splitval:
                        nodes_to_try.append( node.lchild )
                    else:
                        nodes_to_try.append( node.rchild )

    def node_at( self, vtx ):
        """Returns exactly one matching node."""

        # NOTE: This is NOT implemented as a call to iternodequery because it
        # is a very common special case and we want it to fail if a wildcard is
        # passed in.  Having this special-cased allows us to optimize for the
        # common case and to eliminate a bunch of tests for wildcards.
        node = self.root

        while node.splitdim is not None:
            if vtx[node.splitdim] < node.splitval:
                node = node.lchild
            else:
                node = node.rchild

        return node

#------------------------------------------------------------------------------
class kdNode(object):
    dranges = {}
    def __init__( self, cube, depth=0 ):
        """Initialize the kdNode object, given constraints and a cube class

        arguments:
            constraints -- bounds of this node
            cube -- a hypercube, which contains all of the relevant
                information for interpolation
        """

        self.depth = depth
        self.cube = cube
        self.dims = cube.dims
        self.constraints = cube.constraints

        self.splitdim = None
        self.splitval = None
        self.lchild = None
        self.rchild = None

        if not kdNode.dranges.has_key( self.dims ):
            kdNode.dranges[self.dims] = range(self.dims)

        self.drange = kdNode.dranges[self.dims]

    def is_split( self ):
        return self.splitdim is not None

    def split( self, dim, distance=0.5 ):
        """Split this node in the appropriate dimension.
        
        dim: dimension in which to split
        """
        self.lchild, self.rchild = self.child_nodes( dim, distance )
        self.splitdim = dim
        cl, cr = self.cube.constraints[dim]
        self.splitval = distance * cr + (1-distance) * cl
        return self.lchild, self.rchild

    def child_nodes( self, dim, distance=0.5 ):
        """Create two children from splitting this node along the indicated
        dimension.
        
        dim: dimension in which to split the node.
        """
        # Now we create two new children in the specified dimension

        lcube, rcube = self.cube.split( dim, distance )
        lchild = kdNode( lcube, self.depth+1 )
        rchild = kdNode( rcube, self.depth+1 )

        return lchild, rchild

#------------------------------------------------------------------------------
