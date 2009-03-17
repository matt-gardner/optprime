#!/usr/bin/env python2.2
import _general

class step2d(_general._Base):
    def setup(self):
        super(step2d, self).setup()
        self.constraints = ((0,5), (0,5))

    def __call__(self, input):
        x, y = input

        if x ** 2 + y ** 2 < 10:
            return 1
        else:
            return -1

f = step2d()
