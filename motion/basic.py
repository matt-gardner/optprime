from __future__ import division
import random
from aml.opt.varargs import VarArgs
from aml.opt.Vector import Vector
from math import sqrt
from aml.opt.cubes.cube import Cube

#------------------------------------------------------------------------------
class _Base(VarArgs):
    __slots__ = ['rand']

    _args = [
        ( 'restrictvel', False, 'Restrict velocities' ),
        ]

    def __init__( self, comparator, constraints, *args, **kargs ):
        super(_Base,self).__init__( *args, **kargs )
        self.comparator = comparator
        self.rand = random.Random()
        self.constraints = constraints
        self.dims = len(constraints)
        self.cube = Cube( constraints )
        sizes = [abs(cr-cl) for cl,cr in constraints]
        vconstraints = [(-s,s) for s in sizes]
        self.vcube = Cube( vconstraints )
        self.sign = 1.0

    def pre_batch( self, soc ):
        pass

    def post_batch( self, soc ):
        pass

#------------------------------------------------------------------------------
class Basic(_Base):
    _args = [
        ( 'm1', 1.0, 'Momentum start' ),
        ( 'm2', 1.0, 'Momentum stop' ),
        ( 'mstep', 0.0, 'Momentum step' ),
        ( 'phi1', 2.05, "Max of phi_1" ),
        ( 'phi2', 2.05, "Max of phi_2" ),
        ( 'kappa', 1.0, "Clerc's Kappa value, always in (0,1)" ),
        ( 'randvecs', False, "Use random vectors instead of random constants "),
        ( 'arpso', False, "Use ARPSO diversity guided behavior" ),
        ( 'constricted', True, "Use constricted PSO, per Clerc" ),
        ( 'arpso_high', 0.25, "High water mark for ARPSO" ),
        ( 'arpso_low', 5.0e-6, "Low water mark for ARPSO" ),
        ( 'arpso_rate', 1.0, "Learning rate for ARPSO" ),
        ]
    def __init__( self, *args, **kargs ):
        super(Basic, self).__init__( *args, **kargs )

        self.msmall = min(self.m1, self.m2)
        self.mbig = max(self.m1, self.m2)

        self.current_arpso_low = self.arpso_low
        self.current_arpso_high = self.arpso_high

        self.momentum = self.m1

        p1, p2 = [Vector(x) for x in zip( *self.cube.constraints )]
        self.diaglength = abs(p1 - p2)

    def pre_batch( self, soc ):
        # Calculate diversity and set the sign appropriately if needs be.
        if not self.arpso or not soc.is_initialized:
            return

        L = self.diaglength
        S = soc.numparticles()
        piter = soc.iterparticles()
        pmean = Vector(piter.next())
        for p in piter:
            pmean += p
        pmean /= S

        diversity = sum([abs(p-pmean) for p in soc.iterparticles()]) / (S*L)

        # Set the sign accordingly
        if diversity > self.current_arpso_high:
            self.sign = 1
        elif diversity < self.current_arpso_low:
            self.sign = -1
            self.current_arpso_low *= self.arpso_rate
            self.current_arpso_high *= self.arpso_rate

    def __call__( self, particle, neighbor ):
        """Get the next velocity from this particle given a particle that it
        should be moving toward"""

        phi1 = self.phi1
        phi2 = self.phi2
        s = 1.0
        if self.constricted:
            phi = phi1 + phi2
            kappa = self.kappa
            if phi > 4:
                s = 2*kappa/abs(2 - phi - sqrt(phi**2 - 4*phi))
            else:
                s = kappa
                
        if self.randvecs:
            r1 = Vector([self.rand.uniform(0,phi1) for x in xrange(self.dims)])
            r2 = Vector([self.rand.uniform(0,phi2) for x in xrange(self.dims)])
        else:
            r1 = self.rand.uniform(0,phi1)
            r2 = self.rand.uniform(0,phi2)
        m = self.momentum

        grel = neighbor.bestpos - particle.pos
        prel = particle.bestpos - particle.pos

        newvel = s * (particle.vel*m + self.sign*(grel*r1 + prel*r2))

        if self.restrictvel:
            self.vcube.constrain_vec( newvel )

        # Apply the momentum schedule before doing anything else
        m += self.mstep
        m = min(self.mbig, m)
        m = max(self.msmall, m)

        self.momentum = m

        return particle.pos + newvel, newvel

    def _setsign( self, soc ):
        if not self.arpso:
            return

        dh = self.arpso_high
        dl = self.arpso_low

#------------------------------------------------------------------------------
class BasicAdaptive(_Base):
    _args = [
        ('k', -0.5, 'Friction coefficient -- usually should be negative'),
        ('dt', 1.0, 'Initial time step for each particle'),
        ('step_inc', 1.5, 'Amount to increment the timestep (multiply)' ),
        ('step_dec', 0.5, 'Amount to decrement the timestep (multiply)' ),
        ('max_err_inc', 0.1, 'Maximum allowed error' ),
        ('c1', 2.0, 'first random constant multiplier' ),
        ('c2', 2.0, 'second random constant multiplier' ),
        ]
    def __init__( self, *args, **kargs ):
        super(BasicAdaptive, self).__init__( *args, **kargs )

    def __call__( self, particle, neighbor ):
        """Adaptation of the APSO (Tsou and MacNish) -- this actually always
        keeps the position whether we liked it or not (easier with this code
        base) and performs the step calculations right before diving into the
        next calculation.
        """

        if not hasattr(particle, 'prev_val'):
            particle.prev_val = particle.val

        if not hasattr(particle, 'dt'):
            particle.dt = self.dt

        # Now decide whether to decrement the time step or not.

        dims, k, c1, c2 = self.dims, self.k, self.c1, self.c2

        r1 = Vector([self.rand.uniform(0,c1) for x in xrange(dims)])
        r2 = Vector([self.rand.uniform(0,c2) for x in xrange(dims)])

        pos, vel, dt = particle.pos, particle.vel, particle.dt

        # This needs to be changed to allow for maximization and negative
        # values If the current value is better than the last, then we increase
        # the time step, otherwise if we're within an allowable range, we
        # decrease it.
        if self.comparator(particle.val, particle.prev_val):
            particle.dt *= self.step_inc
        elif particle.val / particle.prev_val >= self.max_err_inc:
            particle.dt *= self.step_dec
            vel *= self.step_dec

        grel = neighbor.bestpos - pos
        prel = particle.bestpos - pos

        acc = r1 * grel + r2 * prel + k * vel

        newvel = vel + acc * dt
        newpos = pos + vel * dt + (acc * dt**2)/2

        # Store the value before the state change for future use
        particle.prev_val = particle.val

        return newpos, newvel

#------------------------------------------------------------------------------
class BasicGauss(_Base):
    _args = []
    def __init__( self, *args, **kargs ):
        super(BasicGauss, self).__init__( *args, **kargs )

    def __call__( self, particle, neighbor ):
        """Get the next velocity from this particle given a particle that it
        should be moving toward"""

        gauss = self.rand.gauss

        phi = 2/0.97225 # per Clerc's 2003 TRIBES paper
        chi = 1/(phi - 1 + sqrt(phi**2 - 2*phi))

        grel = neighbor.bestpos - particle.pos
        prel = particle.bestpos - particle.pos

        # Generate a Gaussian around the velocity vectors according to Clerc's
        # paper.
        gvel = Vector([gauss(x,abs(x)/2) for x in grel])
        pvel = Vector([gauss(x,abs(x)/2) for x in prel])

        newvel = chi * (particle.vel + gvel + pvel)

        return None, newvel

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
