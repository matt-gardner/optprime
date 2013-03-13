#!/usr/bin/env python

from copy import deepcopy
import unittest

from optprime import specmethod
from optprime.specex import SpecExPSO
from optprime.particle import Particle, SEParticle, Dummy

import default_opts


class Test_SpecMethod(unittest.TestCase):

    def setUp(self):
        self.opts = default_opts.default_specex_opts()
        self.specex = SpecExPSO(self.opts, [])
        self.rand = self.specex.initialization_rand(0)

    def test_unimplemented_errors(self):
        spec_method = specmethod._SpecMethod()
        self.assertRaises(NotImplementedError, spec_method.pick_child, None, 
                None, None)
        self.assertRaises(NotImplementedError,
                spec_method.pick_neighbor_children, None, None, None)
        self.assertRaises(NotImplementedError,
                spec_method.update_neighbor_nbest, None, None, None)
        self.assertRaises(NotImplementedError, spec_method.message_ids, None)

    def test_get_neighbors(self):
        spec_method = specmethod._SpecMethod()
        spec_method.setup(self.specex, None)
        particles = list(self.specex.topology.newparticles(0, self.rand))
        self.assertEquals(spec_method.get_neighbors(particles[0], []), [])
        self.assertEquals(spec_method.get_neighbors(particles[0], particles),
                [particles[0], particles[1], particles[9]])


class TestOneCompleteIteration(unittest.TestCase):

    def setUp(self):
        self.opts = default_opts.default_specex_opts()
        self.specex = SpecExPSO(self.opts, [])
        self.pruner = specmethod.OneCompleteIteration()
        self.pruner.setup(self.specex)
        self.rand = self.specex.initialization_rand(0)

    def test_generate_children(self):
        particles = list(self.specex.topology.newparticles(0, self.rand))
        particle = particles[4]
        neighbors = [particles[3]]
        children = set(self.pruner.generate_children(particle, neighbors))
        expected_children = set()
        child = SEParticle(particle, specpbest=False, specnbestid=-1)
        self.specex.just_move(child)
        expected_children.add(child)
        child = SEParticle(particle, specpbest=False,
                specnbestid=particles[3].id)
        child.nbestpos = particles[3].pos
        self.specex.just_move(child)
        expected_children.add(child)
        child = SEParticle(particle, specpbest=True, specnbestid=-1)
        child.pbestpos = particle.pos
        self.specex.just_move(child)
        expected_children.add(child)
        child = SEParticle(particle, specpbest=True,
                specnbestid=particles[3].id)
        child.pbestpos = particle.pos
        child.nbestpos = particles[3].pos
        self.specex.just_move(child)
        expected_children.add(child)
        for child in expected_children:
            self.assert_(self.set_contains_item(children, child))
        self.assertEquals(len(children), len(expected_children))

    def set_contains_item(self, set, item):
        for cand in set:
            if repr(cand) == repr(item):
                return True
        else:
            return False


class TestReproducePSO(unittest.TestCase):

    def setUp(self):
        self.opts = default_opts.default_specex_opts()
        self.specex = SpecExPSO(self.opts, [])
        self.pruner = specmethod.OneCompleteIteration()
        self.pruner.setup(self.specex)
        self.rand = self.specex.initialization_rand(0)
        self.smethod = specmethod.ReproducePSO()
        self.smethod.setup(self.specex, self.pruner)

    def test_bad_pruner(self):
        pruner = specmethod._Pruner()
        self.assertRaises(ValueError, self.smethod.setup, self.specex, pruner)

    def test_message_ids_with_particle(self):
        opts = self.opts
        opts.top = 'topology.Rand'
        opts.top__neighbors = 1
        opts.top__num = 100
        opts.top__noselflink = True
        specex = SpecExPSO(opts, [])
        particles = list(specex.topology.newparticles(0, self.rand))
        particle = particles[3]
        specex.set_neighborhood_rand(particle)
        n = list(specex.topology.iterneighbors(particle))[0]
        ndummy = Dummy(n, particle.iters+1, 0)
        specex.set_neighborhood_rand(ndummy)
        n2 = list(specex.topology.iterneighbors(ndummy))[0]
        n2dummy = Dummy(n2, particle.iters+2, 0)
        specex.set_neighborhood_rand(n2dummy)
        n3 = list(specex.topology.iterneighbors(n2dummy))[0]
        expected_message_ids = [n, n2, n3]
        expected_message_ids.sort()

        self.smethod.setup(specex, self.pruner)
        message_ids = list(self.smethod.message_ids(particle))
        message_ids.sort()
        self.assertEquals(expected_message_ids, message_ids)

    def test_message_ids_with_separticle(self):
        opts = self.opts
        opts.top = 'topology.Rand'
        opts.top__neighbors = 1
        opts.top__num = 100
        opts.top__noselflink = True
        specex = SpecExPSO(opts, [])
        particles = list(specex.topology.newparticles(0, self.rand))
        particle = SEParticle(particles[3], specpbest=True, specnbestid=-1)
        specex.set_neighborhood_rand(particle)
        n = list(specex.topology.iterneighbors(particle))[0]
        ndummy = Dummy(n, particle.iters+1, 0)
        specex.set_neighborhood_rand(ndummy)
        n2 = list(specex.topology.iterneighbors(ndummy))[0]
        expected_message_ids = [n, n2]
        expected_message_ids.sort()

        self.smethod.setup(specex, self.pruner)
        message_ids = list(self.smethod.message_ids(particle))
        message_ids.sort()
        self.assertEquals(expected_message_ids, message_ids)

    def test_pick_child(self):
        opts = self.opts
        opts.func__dims = 1
        specex = SpecExPSO(opts, [])
        self.smethod.setup(specex, self.pruner)
        particle = Particle(4, 1, 2, 5)
        neighbors = [Particle(3, 2, 3, 4), Particle(5, 3, 4, 3)]
        children = list(self.smethod.generate_children(particle, neighbors))
        for child in children:
            if child.specpbest == True and child.specnbestid == 5:
                real_child = child
        child = self.smethod.pick_child(particle, neighbors, children)
        real_child.nbestval = 3
        self.assertEquals(repr(real_child.make_real_particle()), repr(child))

    def test_pick_child_doesnt_modify_state(self):
        particles = list(self.specex.topology.newparticles(0, self.rand))
        neighbors = [particles[n]
                for n in self.specex.topology.iterneighbors(particles[3])]
        children = list(self.smethod.generate_children(particles[3], neighbors))
        messages = [n.make_message_particle() for n in neighbors]
        old_messages = deepcopy(messages)
        self.smethod.pick_child(particles[3], messages, children)
        self.assertEquals(old_messages, messages)

    def test_no_child_in_pick_child(self):
        particles = list(self.specex.topology.newparticles(0, self.rand))
        neighbors = [particles[n]
                for n in self.specex.topology.iterneighbors(particles[3])]
        children = []
        messages = [n.make_message_particle() for n in neighbors]
        self.assertRaises(RuntimeError, self.smethod.pick_child,
                particles[3], messages, children)

    # I'm leaving out tests for pick_neighbor_children and update_neighbor_nbest
    # because it would take a lot of work to get correct tests for them, and
    # they mostly just rely on pick_child anyway.  So they will be indirectly
    # tested in some bigger tests.  If the bigger tests break and none of the
    # smaller ones do, try looking at those two methods that I'm skipping.

    def test_update_child_bests(self):
        particle = Particle(1, 1, 1, 1)
        neighbor = Particle(2, 2, 2, 2)
        child = Particle(3, 3, 3, 3)
        self.smethod.update_child_bests(particle, neighbor, child)
        self.assertEquals(2, child.nbestval)
        self.assertEquals(1, child.pbestval)


class TestPickBestChild(unittest.TestCase):

    def setUp(self):
        self.opts = default_opts.default_specex_opts()
        self.specex = SpecExPSO(self.opts, [])
        self.pruner = specmethod.OneCompleteIteration()
        self.pruner.setup(self.specex)
        self.rand = self.specex.initialization_rand(0)
        self.smethod = specmethod.PickBestChild()
        self.smethod.setup(self.specex, self.pruner)

    def test_pick_child(self):
        opts = self.opts
        opts.func__dims = 1
        specex = SpecExPSO(opts, [])
        self.smethod.setup(specex, self.pruner)
        particle = Particle(4, 1, 2, 5)
        neighbors = [Particle(3, 2, 3, 4), Particle(5, 3, 4, 3)]
        children = [SEParticle(Particle(1, 1, 1, 1), True, -1),
                SEParticle(Particle(2, 2, 2, 2), False, -1),
                SEParticle(Particle(3, 3, 3, 3), False, -1),
                SEParticle(Particle(4, 4, 4, 4), False, -1)]
        for child in children:
            child.nbestval = None
        child = self.smethod.pick_child(particle, neighbors, children)
        self.assertEquals(1, child.id)
        self.assertEquals(1, child.pbestval)
        self.assertEquals(3, child.nbestval)
        
    def test_pick_child_doesnt_modify_state(self):
        particles = list(self.specex.topology.newparticles(0, self.rand))
        neighbors = [particles[n]
                for n in self.specex.topology.iterneighbors(particles[3])]
        children = list(self.smethod.generate_children(particles[3], neighbors))
        messages = [n.make_message_particle() for n in neighbors]
        old_messages = deepcopy(messages)
        self.smethod.pick_child(particles[3], messages, children)
        self.assertEquals(old_messages, messages)


class Test_Pruner(unittest.TestCase):

    def test_not_implemented_methods(self):
        pruner = specmethod._Pruner()
        self.assertRaises(NotImplementedError,
                pruner.generate_children, None, None)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(
        TestOneCompleteIteration))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(
        Test_SpecMethod))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(
        TestReproducePSO))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(
        TestPickBestChild))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(
        Test_Pruner))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())

# vim: et sw=4 sts=4
