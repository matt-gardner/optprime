from __future__ import division

try:
    from numpy import array
except ImportError:
    from numpypy import array

try:
    from itertools import izip as zip
except ImportError:
    pass

try:
    range = xrange
except NameError:
    pass


class Cube(object):
    def __init__(self, constraints):
        """Creates a new cube object.  Requires constraints at a minimum,
        but also allows values (of unspecified format) to be passed in."""
        self.constraints = array(constraints)
        self.center = (self.constraints[:, 1] - self.constraints[:, 0]) / 2
        self.lengths = abs(self.constraints[:, 1] - self.constraints[:, 0])
        self.dims = len(constraints)

    def random_vec(self, rand):
        """Return a random vector within the constraints of this cube.
        """
        return array([rand.uniform(*c) for c in self.constraints])

    def constrain_vec(self, vec, use_abs=False):
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

