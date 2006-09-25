from __future__ import division

class statval(object):
    def __init__( self ):
        self.variance = 0
        self.mean = 0
        self.consistency = 1
        self.num = 0

        self._running_sum = 0
        self._running_sum_abs = 0
        self.oldval = 0

    def newval( self, val ):
        # When adding a new value, we update the mean and variance

        delta = val - self.oldval

        oldmean = self.mean

        self.mean = (self.num * self.mean + val) / (self.num + 1)
        if self.num == 0:
            self.variance = 0
        else:
            self.variance = (1.0 - 1.0/self.num) * self.variance + \
                (self.num + 1) * ((self.mean-oldmean)*(self.mean-oldmean))

        self._running_sum = self._running_sum + delta
        self._running_sum_abs = self._running_sum_abs + abs(delta)

        if self._running_sum_abs > 0:
            self.consistency = abs(self._running_sum) / self._running_sum_abs

        self.num += 1
        self.oldval = val

    def __str__( self ):
        return "m: %r, v: %r, n: %d" % (self.mean, self.variance, self.num)

    __repr__ = __str__
