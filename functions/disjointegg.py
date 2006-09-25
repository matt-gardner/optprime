#!/usr/bin/env python2.2

import math

class disjointegg(object):
    __slots__ = 'constraints'
    def __init__( self ):
        self.constraints = ((0,5), (0,5))

    def __call__( self, input ):
        x, y = input

        alpha = 0.71 * int(y) + 1.0

        return ((2.5-x)**2 + (2.5-y)**2)/10 + math.sin(x*alpha) * math.cos(y)

f = disjointegg()
