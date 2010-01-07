"""A library for reading output from amlpso."""


from __future__ import division

class Batch(object):
    def __init__(self):
        self.keys = []
        self.map = {}

    def add(self, key, value):
        self.keys.append(key)
        self.map[key] = value

    def __getitem__(self, key):
        return self.map[key]

    def __iter__(self):
        return iter(self.keys)

    def last(self):
        return self.keys[-1]

    def lastitem(self):
        key = self.last()
        return key, self.map[key]


class PSOData(object):
    def __init__(self, infile):
        self.headers = []
        self.batches = []

        for line in infile:
            if line[0:7] == '# Batch':
                self.batches.append(Batch())
                continue
            if line.isspace():
                continue
            if (line[0] == '#'):
                if line.split()[1] not in ('Batch', 'DONE'):
                    self.headers.append(line)
                continue
            iteration, value = line.split()
            iteration = int(iteration)
            value = float(value)
            lastbatch = self.batches[-1]
            lastbatch.add(iteration, value)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

    def average(self, key):
        """Finds the average for a given key."""
        values = [batch[key] for batch in self.batches]
        return sum(values) / len(values)

    def statistics(self, key, trim):
        """Finds low, median, and high values after trimming outliers.
        
        Note that if trim is 2, and we have 20 samples, this is the 10th,
        50th, and 90th percentiles.
        """
        values = [batch[key] for batch in self.batches]
        values.sort()
        trimmed = values[trim:-1-trim]
        midpoint = int(len(values) / 2)
        med = (values[midpoint] + values[-midpoint]) / 2
        return trimmed[0], med, trimmed[-1]


# vim: et sw=4 sts=4
