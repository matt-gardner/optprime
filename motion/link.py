from __future__ import division

import math
import basic
from itertools import izip
from Vector import Vector

#------------------------------------------------------------------------------
class Link1(basic._Base):
    """This particle is the first linear kalman approximation -- the one that
    makes mathematical sense"""

    _args = [
        ( 'variance', 0.6, 'Sample variance' ),
        ( 'weight', 0.45, 'Weight applied to predictions' ),
        ( 'usepbest', False, 'Use pbest' ),
        ( 'pweight', 0.1, 'Weight applied to pbest/gbest combination' ),
        ( 'dimdiv', False, 'Divide variance by number of dimensions' ),
        ( 'norm', True, 'Normalize vectors instead of just combining' ),
        ( 'vecsdev', False, 'Use a vector of standard deviations, not just a constant' ),
        ]

    def __init__( self, *args, **kargs ):
        super( Link1, self ).__init__( *args, **kargs )

        if self.dimdiv:
            self.sdev = math.sqrt( self.variance / self.dims )
        else:
            self.sdev = math.sqrt( self.variance )

    def __call__( self, particle, neighbor ):
        # Here we get the new gnorm dependent on whether we include pbest or not
        if self.usepbest:
            gnorm1 = neighbor.bestpos - particle.pos
            gnorm2 = particle.bestpos - particle.pos

            pw = self.pweight
            gw = 1 - pw

            gnorm = gw*gnorm1 + pw*gnorm2
        else:
            gnorm = neighbor.bestpos - particle.pos

        pnorm = Vector(particle.vel)
        gval = neighbor.bestval
        gmag = abs(gnorm)
        pmag = abs(pnorm)

        if self.norm:
            pnorm.normalize(pmag)
            gnorm.normalize(gmag)

        # The question is how much we should believe ourselves versus the other
        # vector.
        if self.weight == -1:
            try:
                # Applies to self -- when minimizing...
                weight = neighbor.bestval / (particle.bestval+neighbor.bestval)
                # ...when maximizing
                if self.comparator(1,0):
                    weight = 1 - weight
            except ZeroDivisionError, e:
                weight = 0
        else:
            weight = self.weight
        pw = weight
        gw = 1 - pw

        gauss = self.rand.gauss
        sdev = self.sdev

        nvel = pw * pnorm + gw * gnorm

        if self.norm:
            # We create a new vector and then give it the right
            # magnitude by combining the original magnitudes of the other
            # vectors.

            # Here we have to do a test to see if the magnitude of the vector
            # is very small.  If so, then we may do any number of things.  For
            # example, we may decide to go back to the best value we know.
            if nvel.magnitude() < 1.0e-4:
                if self.comparator(gval, particle.val):
                    nvel = gnorm
                else:
                    nvel = pnorm

            nvel.normalize()
            nvel = Vector([gauss(x,sdev) for x in nvel])
            nvel.normalize()
            nvel *= (pmag * pw + gmag * gw)
        else:
            if self.vecsdev:
                nvel = Vector([gauss(x,math.sqrt(abs(x)*sdev)) for x in nvel])
            else:
                nmag = abs(nvel)
                nvel = Vector([gauss(x,nmag*sdev) for x in nvel])

        if self.restrictvel:
            self.vcube.constrain_vec( nvel, True )

        return None, nvel

#------------------------------------------------------------------------------
class Link2(basic._Base):
    """This particle is the second linear kalman approximation -- the one that
    is a terrible hack"""

    _args = [
        ( 'variance', 0.6, 'Sample variance' ),
        ( 'weight', 0.45, 'Weight applied to predictions' ),
        ( 'craziness', 0.1, 'Variance multiplier for craziness' ),
        ( 'usepbest', False, 'Use pbest' ),
        ( 'pweight', 0.45, 'Weight applied to pbest/gbest combination' ),
        ]

    def __init__( self, *args, **kargs ):
        super( Link2, self ).__init__( *args, **kargs )

        self.sdev = math.sqrt( self.variance )
        self.sizes = [abs(r-l) for l,r in self.constraints]

    def __call__( self, particle, neighbor ):
        if self.usepbest:
            gnorm1 = neighbor.bestpos - particle.pos
            gnorm2 = particle.bestpos - particle.pos
            gmag1 = abs(gnorm1)
            gmag2 = abs(gnorm2)

            gnorm1.normalize(gmag1)
            gnorm2.normalize(gmag2)

            if self.pweight == -1:
                try:
                    # Minimizing -- weight applies to self
                    weight = neighbor.bestval / (particle.bestval + neighbor.bestval)
                    # Maximizing -- invert the weight -- still applies to self
                    if self.comparator(1,0):
                        weight = 1 - weight

                except ZeroDivisionError, e:
                    weight = 0
            else:
                weight = self.weight
            pw = self.rand.gauss(weight, self.sdev)
            gw = 1 - pw

            gnorm = gw * gnorm1 + pw * gnorm2
            if gnorm.magnitude() < 1.0e-4:
                if self.comparator(neighbor.bestval,particle.bestval):
                    gnorm = gnorm1
                else:
                    gnorm = gnorm2

            gnorm.normalize()
            gnorm *= (gmag1 * gw + gmag2 * pw)
        else:
            gnorm = neighbor.bestpos - particle.pos

        pnorm = Vector(particle.vel)

        gval = neighbor.bestval

        gmag = abs(gnorm)
        pmag = abs(pnorm)

        gnorm.normalize(gmag)
        pnorm.normalize(pmag)

        # The question is how much we should believe ourselves versus the other
        # vector.
        if self.weight == -1:
            try:
                weight = neighbor.bestval / (particle.bestval+neighbor.bestval)
                if self.comparator(1,0):
                    weight = 1 - weight
            except ZeroDivisionError, e:
                weight = 0
        else:
            weight = self.weight
        pw = self.rand.gauss( weight, self.sdev )
        gw = 1 - pw

        # We create a new normal vector and then give it the right magnitude by
        # combining the original magnitudes of the other vectors.
        nvel = pw * pnorm + gw * gnorm

        # Here we have to do a test to see if the magnitude of the vector is
        # very small.  If so, then we may do any number of things.  For
        # example, we may decide to go back to the best value we know.
        if nvel.magnitude() < 1.0e-4:
            if self.comparator(gval, particle.val):
                nvel = gnorm
            else:
                nvel = pnorm

        # Pick a random element of the vector and randomize it.
        elidx = self.rand.randrange(0,len(nvel))
        if self.craziness == -1:
            nvel = Vector([self.rand.gauss(x,max(abs(x)/2,s*0.00001)) for x,s in izip(nvel,self.sizes)])
        else:
            sdev = self.sdev * self.craziness
            nvel[elidx] = self.rand.gauss( nvel[elidx], sdev )


        nvel.normalize()
        nvel *= (pw * pmag + gw * gmag)

        if self.restrictvel:
            self.cube.constrain_vec( nvel, True )

        return None, nvel

#------------------------------------------------------------------------------
class Link3(basic._Base):
    """This is a very simple hack on the Kalman stuff, making the new position
    a weighted average of the old position, rather than an average over
    velocities."""

    _args = [
        ( 'cfac', 0.0001, 'Covariance factor' ),
        ( 'weight', 0.45, 'Weight applied to predictions' ),
        ( 'craziness', 0.5, 'How much of the velocity to apply to sampling' ),
        ]

    def __init__( self, *args, **kargs ):
        super( Link3, self ).__init__( *args, **kargs )

        self.sizes = [abs(r-l) for l,r in self.constraints]
        self.sdevs = [math.sqrt(self.cfac * s) for s in self.sizes]

    def __call__( self, particle, neighbor ):
        predpos = particle.pos + particle.vel
        goodpos = neighbor.bestpos

        diffpos = predpos - goodpos

        #pw = self.rand.gauss( self.weight, self.cfac )
        if self.weight == -1:
            try:
                weight = neighbor.bestval / (particle.bestval + neighbor.bestval)
            except ZeroDivisionError, e:
                weight = 0
        else:
            weight = self.weight
        pw = weight
        gw = 1 - pw

        meanpos = pw * predpos + gw * goodpos
        meanvel = particle.pos - meanpos
        #sdevs = self.sdevs
        #sdevs = [max(math.sqrt(self.cfac * abs(s)),0.001) for s in diffpos]
        sdevs = [max(self.craziness*s,sdev) for s,sdev in izip(diffpos,self.sdevs)]

        gauss = self.rand.gauss
        newpos = Vector([gauss(x,sdev) for x,sdev in izip(meanpos,sdevs)])
        newvel = newpos - particle.pos

        return None, newvel

#------------------------------------------------------------------------------
class Link4(basic._Base):
    """This is a very simple hack on the Kalman stuff, making the new position
    a weighted average of the old position, rather than an average over
    velocities."""

    _args = [
        ( 'variance', 0.6, 'Weight variance' ),
        ( 'weight', 0.45, 'Weight applied to predictions' ),
        ]

    def __init__( self, *args, **kargs ):
        super( Link4, self ).__init__( *args, **kargs )

        self.sizes = [abs(r-l) for l,r in self.constraints]
        self.sdev = math.sqrt(self.variance)

    def __call__( self, particle, neighbor ):
        predpos = particle.pos + particle.vel
        goodpos = neighbor.bestpos

        if self.weight == -1:
            try:
                weight = neighbor.bestval / (particle.bestval + neighbor.bestval)
            except ZeroDivisionError, e:
                weight = 0
        else:
            weight = self.weight
        pw = self.rand.gauss( weight, self.sdev )
        gw = 1 - pw

        newpos = pw * predpos + gw * goodpos
        newvel = newpos - particle.pos

        return None, newvel

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
