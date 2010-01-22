"""A library for reading output from amlpso."""


from __future__ import division

class Batch(object):
    def __init__(self):
        self.keys = []
        self.map = {}
        self.done = False

    def _set_done(self):
        """Records that the batch had a terminating "DONE" statement."""
        assert not self.done
        self.done = True

    def add(self, key, value):
        self.keys.append(key)
        self.map[key] = value

    def __getitem__(self, key):
        return self.map[key]

    def __iter__(self):
        return iter(self.keys)

    def __len__(self):
        return len(self.keys)

    def last(self):
        return self.keys[-1]

    def lastitem(self):
        key = self.last()
        return key, self.map[key]


class PSOData(object):
    def __init__(self, infile):
        self.options = {}
        self.batches = []
        in_options = False

        for line in infile:
            lastbatch = self.batches[-1] if self.batches else None

            if line.isspace():
                in_options = False
                continue
            if (line[0] == '#'):
                if line.startswith('# Batch'):
                    in_options = False
                    self.batches.append(Batch())
                elif line.startswith('# DONE'):
                    assert lastbatch
                    lastbatch._set_done()
                elif line.startswith('# Options'):
                    in_options = True
                elif in_options:
                    self._add_option_line(line)
                continue

            outputter = self.options['out']
            if outputter == 'output.Basic':
                outfreq = int(self.options['out__freq'])
                iteration = 1 + outfreq * len(lastbatch)
                value = float(line.strip())
            elif outputter == 'output.Pair':
                iteration, value = line.split()
                iteration = int(iteration)
                value = float(value)
            lastbatch.add(iteration, value)

        if not lastbatch.done:
            print 'WARNING: ignoring last batch (incomplete)'
            del self.batches[-1]

    def _add_option_line(self, line):
        assert line[0] == '#'
        line = line[1:]
        if '=' not in line:
            print line
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip()
        self.options[key] = value

    def __getitem__(self, index):
        return self.batches[index]

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
