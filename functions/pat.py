from __future__ import division

from math import exp
import _general


class Pat(_general._Base):
    def __init__( self, *args, **kargs ):
        super( Pat, self ).__init__( *args, **kargs )
        self._set_constraints( ((0.0,3.0),) * self.dims )

        self.alpha = kargs.get( 'alpha', 8.0 )
        self.beta = kargs.get( 'beta', -1.5 )
        self.p11 = kargs.get( 'p11', 0.11136187058385827 )
        self.p21 = kargs.get( 'p21', -0.04998457596588335 )
        self.p12 = kargs.get( 'p12', -0.04998457596588338 )
        self.p22 = kargs.get( 'p22', 0.1221573729448382 )
        self.k = kargs.get( 'k', 75.2 )
        self.m = kargs.get( 'm', 1.4020053175883656 )

    def __call__( self, vec ):
        pt = vec[0]

        alpha = self.alpha
        beta = self.beta
        p11 = self.p11
        p21 = self.p21
        p12 = self.p12
        p22 = self.p22
        k = self.k
        m = self.m

        s1 = pt * exp(alpha + pt * beta) * m
        numerator = (pt*p22)**2 + pt*p22*(p21 + p12) + p12*p21
        denominator = pt**2 * p22 + pt*(p12+p21) + p11 + 1

        return s1 + k*(p22 - numerator/denominator)
