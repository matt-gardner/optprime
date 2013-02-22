#!/usr/bin/env python2.2

import math
from numpy import array
from . import BaseFunction

class DisjointEgg(BaseFunction):
    def setup(self, rand):
        self.dims = 2
        self.constraints = array([(0,5), (0,5)])

    def __call__(self, vec):
        x, y = vec

        alpha = 0.71 * int(y) + 1.0

        return ((2.5-x)**2 + (2.5-y)**2)/10 + math.sin(x*alpha) * math.cos(y)

