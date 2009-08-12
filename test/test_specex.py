#!/usr/bin/env python

import unittest
import optparse
from specex import SpecExPSO

class TestSpecEx(unittest.TestCase):

    def setUp(self):
        self.opts = optparse.Values()
        self.opts.verbose = False
        self.opts.func_center = '0.5'
        self.opts.mrs__verbose = False
        self.opts.top__num = 10
        self.opts.out = 'output.Basic'
        self.opts.mrs__seed = '2'
        self.opts.batches = 1
        self.opts.top__noselflink = False
        self.opts.top = 'topology.Ring'
        self.opts.out__freq = 1
        self.opts.motion__restrictvel = False
        self.opts.spec = 'specmethod.ReproducePSO'
        self.opts.func__dims = 2
        self.opts.func_maximize = False
        self.opts.iters = 100
        self.opts.top__initoffset = 0.0
        self.opts.top__initscale = 1.0
        self.opts.motion__phi2 = 2.05
        self.opts.motion__phi1 = 2.05
        self.opts.pruner = 'specmethod.OneCompleteIteration'
        self.opts.hey_im_testing = True
        self.opts.numtasks = 0
        self.opts.motion__Kappa = 1
        self.opts.mrs__debug = False
        self.opts.quiet = False
        self.opts.mrs = 'mrs.impl.Serial'
        self.opts.motion = 'motion.basic.Constricted'
        self.opts.transitive_best = False
        self.opts.func = 'functions.sphere.Sphere'

    def test_sepso_map(self):
        specex = SpecExPSO(self.opts, [])
        rand = specex.initialization_rand(0)
        particles = list(specex.topology.newparticles(0, rand))
        key = str(particles[2].id)
        value = repr(particles[2])
        emitted_messages = list(specex.sepso_map(key, value))
        specex.just_evaluate(particles[2])
        message = particles[2].make_message_particle()
        expected_messages = [('2', repr(particles[2])),
                ('0', repr(message)),
                ('1', repr(message)),
                ('2', repr(message)),
                ('3', repr(message)),
                ('4', repr(message))]
        for expected_message in expected_messages:
            self.assert_(expected_message in emitted_messages)

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestSpecEx)

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())

# vim: et sw=4 sts=4
