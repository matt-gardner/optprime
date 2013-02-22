from __future__ import division

from math import sqrt, exp, cos, sin, pi
from . import Benchmark

try:
    from itertools import izip as zip
except ImportError:
    pass


class TwoGaussians(Benchmark):
    _each_constraints = (-50, 50)

    def __call__(self, vec):
        magsq = sum([(x-c)**2 for x,c in zip(vec,self.abscenter)])
        mag = sqrt(magsq)
        outersdev = 30
        innersdev = 1
        oosdev = 50.0
        oogaussian = 1.0
        outergaussian = 1.0
        innergaussian = 1.0
        for x, c in zip(vec, self.abscenter):
            outergaussian *= exp((-(x-c)**2)/(2*outersdev**2))
            innergaussian *= exp((-(x-c)**2)/(2*innersdev**2))
            oogaussian *= exp((-(x-c)**2)/(2*oosdev**2))

        return (1-oogaussian) + 0.4*(outergaussian - innergaussian)

class ValleyNeedle(Benchmark):
    _each_constraints = (-50, 50)

    def __call__(self, vec):
        magsq = sum([(x-c)**2 for x,c in zip(vec,self.abscenter)])
        mag = sqrt(magsq)
        flatradius=30
        flatheight=5
        if mag > flatradius:
            # Outside of a certain radius?  Be a cone.
            return mag - flatradius + flatheight
        else:
            sdev = flatradius / 100
            prod = 1.0
            for x, c in zip(vec,self.abscenter):
                prod *= exp((-(x-c)**2)/(2*sdev**2))
            return flatheight*(1 - prod)

class AsymmetricCone(Benchmark):
    _each_constraints = (-50, 50)

    def __call__(self, vec):
        cvec = self.abscenter

        x = vec[0] - cvec[0]

        const = sum([(v-c)**2 for v,c in zip(vec,cvec)]) - x**2
        angle = pi/8

        ca = cos(2*angle)
        sa = sin(2*angle)

        return (-x*sa - sqrt(x**2 + const*ca)) / -ca
