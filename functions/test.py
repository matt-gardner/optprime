import _general

class Test(_general._Base):
    def setup(self, *args, **kargs):
        super(Test,self).setup( *args, **kargs)
        self._set_constraints(((3,200), (4,100), (0,302), (0,200),)*3)

    def __call__(self, vec):
        retval = 0
        for i in range(len(vec)):
            if abs(vec[i] - 5) < .01 : retval -= 5
        return retval
        #return sum(vec)


# vim: et sw=4 sts=4
