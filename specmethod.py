#!/usr/bin/env python

from mrs.param import ParamObj, Param
from particle import Particle, SEParticle, Dummy

class _SpecMethod(ParamObj):
    """A method for performing speculative execution.
    
    Things such as how to pick the speculative branch to accept, how much
    pruning to do, and how to determine which neighbors to accept are defined
    here.
    """

    def setup(self, specex, pruner):
        self.specex = specex
        self.pruner = pruner
    
    def pick_child(self, particle, neighbors, children):
        """Given a particle, its neighbors, and its children, decide which 
        child to return."""
        raise NotImplementedError

    def pick_neighbor_children(self, particle, neighbors, neighbor_children):
        """Given a set of neighbors and their children and neighbors, return a
        set of children, one for each neighbor.  In some cases having the
        particle is also necessary to determine which neighbors are actually
        neighbors."""
        raise NotImplementedError
        

    def itermessages(self, particle):
        """Given a particle, return the set of particles to whom it must
        send messages."""
        raise NotImplementedError

    def generate_children(self, particle, neighbors):
        """Yield the speculative children of a particle given it and its
        neighbors.  Often uses an external pruning class."""
        return self.pruner.generate_children(particle, neighbors, 
                self.specex.just_move)

class ReproducePSO(_SpecMethod):
    """Perform speculative execution in such a way that the original PSO is
    reproduced exactly, just two iterations at a time.  A lot of communication
    is required to make sure the behavior is exactly the same as the original
    PSO, and any good information from speculative particles is ignored except
    on the branch that standard PSO would have taken.
    """

    def setup(self, specex, pruner):
        if not isinstance(pruner, OneCompleteIteration):
            raise ValueError('ReproducePSO must have OneCompleteIteration as '
                    'its pruner!')
        self.pruner = pruner
        self.specex = specex

    def pick_child(self, particle, neighbors, children):
        comparator = self.specex.function.comparator
        realneighbors = self.get_actual_neighbors(particle, neighbors)
        # Look at the messages to see which branch you actually took
        best = self.specex.findbest(realneighbors, comparator)
        particle.nbest_cand(best.pbestpos, best.pbestval, comparator)

        # If you updated your pbest, then your current value will be your
        # pbestval
        updatedpbest = particle.pbestval == particle.value

        if particle.nbestval == best.pbestval:
            bestid = best.id
        else:
            bestid = -1

        # Look through the children and pick the child that corresponds to the
        # branch you took
        for child in children:
            if child.specpbest == updatedpbest and \
                child.specnbestid == bestid:
                    return child.make_real_particle()
        raise RuntimeError("Didn't find a child that matched the right branch!")

    def itermessages(self, particle):
        neighbors = set()
        self.specex.set_neighborhood_rand(particle, swarmid=0)
        for n in self.specex.topology.iterneighbors(particle):
            neighbors.add(n)
            ndummy = Dummy(n, particle.iters, particle.batches)
            self.specex.set_neighborhood_rand(ndummy, swarmid=0)
            for n2 in self.specex.topology.iterneighbors(ndummy):
                neighbors.add(n2)
                if type(particle) == Particle:
                    n2dummy = Dummy(n2, particle.iters, particle.batches)
                    self.specex.set_neighborhood_rand(n2dummy, swarmid=0)
                    for n3 in self.specex.topology.iterneighbors(n2dummy):
                        neighbors.add(n3)
        if type(particle) == Particle:
            particle.iters += 1
            for n in self.specex.topology.iterneighbors(particle):
                neighbors.add(n)
                ndummy = Dummy(n, particle.iters, particle.batches)
                self.specex.set_neighborhood_rand(ndummy, swarmid=0)
                for n2 in self.specex.topology.iterneighbors(ndummy):
                    neighbors.add(n2)
                    n2dummy = Dummy(n2, particle.iters, particle.batches)
                    self.specex.set_neighborhood_rand(n2dummy, swarmid=0)
                    for n3 in self.specex.topology.iterneighbors(n2dummy):
                        neighbors.add(n3)

            particle.iters -= 1
        return neighbors

    def pick_neighbor_children(self, particle, neighbors, all_children):
        self.update_children_pbest(neighbors, all_children)
        # Find the actual children for all particles in neighbors
        n2_all = []
        for neighbor in neighbors:
            neighborsneighbors = self.get_actual_neighbors(neighbor, neighbors)
            neighborchildren = []
            for child in all_children:
                if child.id == neighbor.id:
                    neighborchildren.append(child)
            if (len(neighborchildren) > 0):
                realchild = self.pick_child(neighbor, neighborsneighbors, 
                    neighborchildren)
                n2_all.append(realchild)

        # Get the particle's actual neighbors, update their nbest, return them
        n2_actual = self.get_actual_neighbors(particle, n2_all)
        children = []
        for n2 in n2_actual:
            neighborsneighbors = self.get_actual_neighbors(n2, n2_all)
            comparator = self.specex.function.comparator
            best = self.specex.findbest(neighborsneighbors, comparator)
            n2.nbest_cand(best.pbestpos, best.pbestval, comparator)
            children.append(n2)
        return children

    def get_actual_neighbors(self, particle, neighbors):
        realneighbors = []
        for cand in neighbors:
            #print cand.id
            self.specex.set_neighborhood_rand(cand, swarmid=0)
            if particle.id in self.specex.topology.iterneighbors(cand):
                realneighbors.append(cand)
        #print particle.id, particle.iters, [x.id for x in realneighbors]
        return realneighbors

    def update_children_pbest(self, particles, children):
        for particle in particles:
            for child in children:
                if particle.id == child.id:
                    if Particle.isbetter(particle.pbestval, child.pbestval,
                            self.specex.function.comparator):
                        child.pbestpos = particle.pbestpos
                        child.pbestval = particle.pbestval


class _Pruning(ParamObj):
    """Speculative Pruning
    
    You have an infinite tree of possible executions.  This class defines
    one method, generate_children, that decides how far to go down the 
    tree, and in what way.
    """

    def generate_children(self, particle, neighbors, just_move):
        """Given a particle and its neighbors, generate the particle's
        speculative children."""
        raise NotImplementedError

class OneCompleteIteration(_Pruning):

    def generate_children(self, particle, neighbors, just_move):
        """Given the particle and neighbors at iteration i, yield all possible
        speculative children for iteration i+1."""
        # Create children guessing that pbest was not updated
        child = SEParticle(particle, specpbest=False, specnbestid=-1)
        just_move(child)
        yield child
        for n in neighbors:
            if n.id != particle.id:
                child = SEParticle(particle, specpbest=False, specnbestid=n.id)
                child.nbestpos = n.pos
                just_move(child)
                yield child

        # Create children guessing that pbest was updated
        child = SEParticle(particle, specpbest=True, specnbestid=-1)
        child.pbestpos = particle.pos
        just_move(child)
        yield child
        for n in neighbors:
            child = SEParticle(particle, specpbest=True, specnbestid=n.id)
            child.nbestpos = n.pos
            child.pbestpos = particle.pos
            just_move(child)
            yield child

# vim: et sw=4 sts=4
