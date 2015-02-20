from __future__ import division, print_function

import itertools
import math
import numpy
import random

from optprime.dirstats import make_basic_bingham_wishart_model
from optprime.linalg import degrees_between_unsigned

SEED = 42
DIMS = 5
UNNORMED_TARGET = numpy.array([1, -0.2, 0.2, -0.2, 0.2])
#UNNORMED_TARGET = numpy.array([1, -0.4, 0.4, -0.2, 0.2])
#UNNORMED_TARGET = numpy.array([1, 0.2, -0.05, 0.025, -0.0125])

# Note: Increasing either the KAPPA or the DOF parameter will increase
# the concentration of samples.
#KAPPA = 7
KAPPA = 5
DOF = 5
ITERS = 100000

# Parameters that determine the probability of success from the angle from
# a sampled vector to the target vector.
#PROB_CONCENTRATION = 10
PROB_CONCENTRATION = 100

def normalized(v):
    return v / numpy.linalg.norm(v)

def angles(v):
    """Print the angles from u to the x-axis and target respectively (in
    degrees).
    """
    return ('x-axis angle: %s, target angle: %s' %
        (degrees_between_unsigned(v, x_axis),
        degrees_between_unsigned(v, target)))

def prob(u, v):
    """Compute a success probability from sampled and target vectors."""
    angle = math.acos(abs(numpy.dot(u, v)))
    return math.exp(-PROB_CONCENTRATION * angle ** 2)

x1 = normalized(numpy.array([1, 0]))
x2 = normalized(numpy.array([3, 1]))
x3 = normalized(numpy.array([0, 1]))
print(degrees_between_unsigned(x1, x1), prob(x1, x1))
print(degrees_between_unsigned(x1, x2), prob(x1, x2))
print(degrees_between_unsigned(x1, x3), prob(x1, x3))

rand = random.Random(SEED)
target = normalized(UNNORMED_TARGET)
x_axis = numpy.zeros(DIMS)
x_axis[0] = 1.0


print('Target:', target)
print('Angle from x-axis to target:', degrees_between_unsigned(x_axis, target))

model = make_basic_bingham_wishart_model(DIMS, KAPPA, DOF, rand)

print(model.inv_scale())

print()
print('Samples from the prior predictive distribution:')
for _ in range(20):
    # TODO: For each sample, print the angle from the x-axis and the angle
    # from the target.
    v = model.sample_success(rand)
    print(angles(v))
print()

successes = 0
for i in range(ITERS):
    #print('iter:', i)
    #print('inv_scale_L:', model._bingham_wishart._inv_scale_L)
    x = model.sample_success(rand)
    #print('Sampled x:', angles(x))

    # Set the probability to be proportional to cos(theta)^2
    p = prob(x, target)
    if rand.random() < p:
        #print('Success at x: %s (p=%s)' % (angles(x), p))
        model = model.posterior_success(x)
        successes += 1
    else:
        #print('Failure (p=%s)' % p)
        pass

print('Total successes:', successes)

print()
print('Samples from the posterior predictive distribution:')
for _ in range(10):
    v = model.sample_success(rand)
    #print(v)
    print(angles(v))

print(model.inv_scale())

# vim: et sw=4 sts=4
