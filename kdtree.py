from __future__ import division

from copy import deepcopy
from sets import Set
from operator import mul
from cubes.cube import Cube

#------------------------------------------------------------------------------
    
#------------------------------------------------------------------------------
class KDTree(object):
    def __init__( self, constraints ):
        """Initialize the KDTree object

        arguments:
            constraints -- a list of constraint pairs
            cubecls -- a factory for creating cubes
        """
        self.dims = len(constraints)
        self.constraints = tuple(constraints)
        self.lengths = [abs(cr-cl) for cl,cr in constraints]

        self.root = KDNode( constraints )

    def frompoints(
            cls,
            constraints,
            points,
            enforce_constraints=True,
            truemedian=False,
            ):
        """Create a KDTree that indexes the given points
        
        arguments:
            constraints -- list of spatial constraints for the root node
            points -- list of position vectors
            enforce_constraints -- Only index points within the constraints
            truemedian -- Use the true median (halfway for even lists)
        """
        self = cls( constraints )
        if enforce_constraints:
            points = list(self.root.iter_inside_points( points ))
        if len(points) > 0:
            self.root.index_points( points, truemedian )
        else:
            self.root.point = None
        return self
    frompoints = classmethod(frompoints)

    def fromgrid(cls, constraints, splits ):
        """Create a KDTree that splits the world into a grid given the splits
        
        arguments:
            constraints -- list of spatial constraints for the root node
            splits -- list of splits (or a scalar) for each dimension
        """
        self = cls( constraints )
        self._gridify( splits )
        return self
    fromgrid = classmethod(fromgrid)

    def _gridify( self, splits, node=None ):
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
        for d in xrange(self.dims):
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

    def add_point( self, point, enforce_constraints=True, minsizes=None ):
        """Find the cube containing this point and split it to handle the
        original point and this one."""

        leaf = self.node_at( point )

        if not hasattr(leaf,'point') or leaf.point is None:
            leaf.point = point
        elif enforce_constraints and leaf.cube.is_inside( point ):
            # Now we do the splitting, always in the longest dimension
            dim = leaf.longest_dim()
            length = leaf.lengths[dim]

            if minsizes is None or minsizes[dim] < length:
                # Get the point from the leaf node.
                points = [point, leaf.point]
                del leaf.point

                # Now just index this leaf node
                leaf.index_points( points )

    def find_neighbors( self, point ):
        """Finds the neighbors of this point by looking for nearest adjacent
        cubes.

        arguments:
            point -- the query point in question
        """

        leaf = self.node_at( point )
        query = point

        # Now we have the node.  From here we look for neighbors.  This is done
        # by searching backward in the tree for a node that splits in a
        # particular dimension.  Then we go the other way.
        neighbors = {} # Keyed on dimension and direction (dim,dir)
        curnode = leaf
        while len(neighbors) < (self.dims * 2):
            parent = curnode.parent
            if parent is None:
                break
            # Is the current node a left or right child, and in which dimension?
            splitdim = parent.splitdim
            if parent.lchild is curnode:
                # This node is on the left, so we search down the right side for
                # another leaf node.
                leafsearch = parent.rchild
                direction = "R"
            else:
                leafsearch = parent.lchild
                direction = "L"

            k = (splitdim,direction)
            # If we have seen this kind of a move before, and we decided to
            # keep it, then forget it.  We already have a close neighbor.
            if not neighbors.has_key( k ):
                while leafsearch.is_split():
                    # If we are not randomly selecting things, then we just
                    # pick the one that fits best with the query point.
                    if query[leafsearch.splitdim] <= leafsearch.splitval:
                        leafsearch = leafsearch.lchild
                    else:
                        leafsearch = leafsearch.rchild

                # Now we have the leaf node corresponding to a particular
                # neighbor.  Store it in the dictionary
                neighbors[k] = leafsearch.point
            curnode = parent
        return neighbors.values()

    def iterpoints( self, node=None ):
        for node in self.iterleaves():
            if hasattr(node,'point'):
                yield node.point

    def iterleaves( self ):
        """iterator generator for dumping out each leaf node."""
        for node in self.iternodes():
            if node.splitdim is None:
                yield node

    def iternodes( self ):
        """Dumps out each node in an inorder traversal"""
        stack = [(self.root, 0)]
        while stack:
            node, direction = stack.pop()

            if direction == 0:
                stack.append( (node, 1) )
                if node.lchild is not None:
                    stack.append( (node.lchild, 0) )
            else:
                # The left side was already searched and we came back to
                # this node, so emit it and search the right side.
                yield node
                if node.rchild is not None:
                    stack.append( (node.rchild, 0) )

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
                    if query[node.splitdim] <= node.splitval:
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
            if vtx[node.splitdim] <= node.splitval:
                node = node.lchild
            else:
                node = node.rchild

        return node

#------------------------------------------------------------------------------
class KDNode(object):
    def __init__( self, constraints, depth=0, parent=None ):
        """Initialize the KDNode object, given constraints and a cube class

        arguments:
            constraints -- bounds of this node
            depth -- Position in the tree
        """

        self.depth = depth
        self.parent = parent
        self.dims = len(constraints)
        self.constraints = tuple(constraints)
        self.cube = Cube(self.constraints)
        self.lengths = [abs(cr-cl) for cl,cr in constraints]

        self.splitdim = None
        self.splitval = None
        self.lchild = None
        self.rchild = None

    def is_split( self ):
        return self.splitdim is not None

    def iter_inside_points( self, points ):
        for p in points:
            if self.cube.is_inside( p ):
                yield p

    def partition( self, points, dim, truemedian=True ):
        if len(points) <= 1:
            return None

        points.sort(lambda x,y:cmp(x[dim],y[dim]))

        # Split on the median (or the point closest to it)
        npoints = len(points)
        pidx = npoints//2 - 1
        splitval = points[pidx][dim]
        if truemedian and len(points) % 2 == 0:
            splitval += points[pidx+1][dim]
            splitval /= 2
        return splitval, points[:pidx+1], points[pidx+1:]


    def index_points( self, points, truemedian=True ):
        """Used by the point indexing routine to index a subset of the points
        that fit within a particular node.
        """
        # Now we analyze the points, splitting along the longest dimension at
        # the median point.

        if len(points) == 1:
            # If we have only a single point in the list, there is no need to
            # split.  We can just set the point for this node and move on.
            self.point = points[0]
            return

        # Partition the points and split the node
        longest = self.longest_dim()
        splitval, leftpoints, rightpoints = \
                self.partition( points, longest, truemedian )

        lnode, rnode = self.split_at_val( longest, splitval )

        # Recursive calls
        lnode.index_points( leftpoints, truemedian )
        rnode.index_points( rightpoints, truemedian )

    def split_at_val( self, dim, val ):
        """Split this node at the specified place (absolute coordinates)

        arguments:
            dim -- dimension in which to split
            val -- coordinate in that dimension where split will occur
        """
        self.splitdim = dim
        self.splitval = val

        clnew = []
        crnew = []
        for i, (cl,cr) in enumerate(self.constraints):
            if i == dim:
                clnew.append( (cl, self.splitval) )
                crnew.append( (self.splitval, cr) )
            else:
                clnew.append( (cl, cr) )
                crnew.append( (cl, cr) )

        self.lchild = KDNode( clnew, self.depth+1, self )
        self.rchild = KDNode( crnew, self.depth+1, self )

        return self.lchild, self.rchild

    def split_at_dist( self, dim, dist=0.5 ):
        """Split this node at the specified distance (0.0 to 1.0)

        arguments:
            dim -- dimension in which to split
            dist -- number between 0.0 and 1.0, defaults to 0.5
        """
        cl, cr = self.constraints[dim]
        splitval = distance * cr + (1-distance) * cl

        return self.split_at_val( dim, splitval )

    def lebesgue_measure( self ):
        return reduce( mul, self.lengths )

    volume = lebesgue_measure

    def longest_dim( self ):
        nodeenum = enumerate(self.lengths)
        bestdim, bestlength = nodeenum.next()
        for i, length in nodeenum:
            if length > bestlength:
                bestlength = length
                bestdim = i
        return bestdim

    split = split_at_dist # splits default to middle based on distance
#------------------------------------------------------------------------------
