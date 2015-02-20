from __future__ import division, print_function

import itertools
import math
import numpy
import random

from optprime.dirstats import make_basic_bingham_wishart_model
from optprime.linalg import degrees_between

SEED = 42
DIMS = 5
UNNORMED_TARGET = numpy.array([1, -0.2, 0.2, -0.2, 0.2])
#UNNORMED_TARGET = numpy.array([1, 0.2, -0.05, 0.025, -0.0125])
TARGET = UNNORMED_TARGET / numpy.linalg.norm(UNNORMED_TARGET)

# Note: Increasing either the KAPPA or the DOF parameter will increase
# the concentration of samples.
KAPPA = 7
DOF = 20
ITERS = 100

rand = random.Random(SEED)

print('Target:', TARGET)
x_axis = numpy.zeros(DIMS)
x_axis[0] = 1.0
print('Angle from x-axis to target:', degrees_between(x_axis, TARGET))

model = make_basic_bingham_wishart_model(DIMS, KAPPA, DOF, rand)

def angles(v):
    """Print the angles from u to the x-axis and target respectively (in
    degrees).
    """
    return ('x-axis angle: %s, target angle: %s' %
        (degrees_between(v, x_axis), degrees_between(v, TARGET)))

print()
print('Samples from the prior predictive distribution:')
for _ in range(20):
    # TODO: For each sample, print the angle from the x-axis and the angle
    # from the target.
    v = model.sample_success(rand)
    print(angles(v))
print()

for i in range(ITERS):
    #print('iter:', i)
    #print('inv_scale_L:', model._bingham_wishart._inv_scale_L)
    x = model.sample_success(rand)
    print('Sampled x:', angles(x))

    # Set the probability to be proportional to cos(theta)^2
    p = numpy.dot(x, TARGET) ** 2
    if rand.random() < p:
        print('Success (p=%s)' % p)
        model = model.posterior_success(x)
    else:
        print('Failure (p=%s)' % p)
        pass

print()
print('Samples from the posterior predictive distribution:')
for _ in range(5):
    v = model.sample_success(rand)
    print(v)

# vim: et sw=4 sts=4
