#!/usr/bin/env python2.2

class step2d(object):
    def __init__( self ):
        self.constraints = ((0,5), (0,5))

    def __call__( self, input ):
        x, y = input

        if x ** 2 + y ** 2 < 10:
            return 1
        else:
            return -1

f = step2d()
