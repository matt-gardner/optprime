from __future__ import division
import _general

class ExternalFunction(_general._Base):
    """A function that takes an executable file and uses that as the function
    to optimize.  So, if you already have a C or C++ executable, you can just
    give that to this function, and you don't have to translate the C code to
    python.  It might run a lot slower than if you translate it, though, 
    depending on the function - Popen is pretty slow.  

    The only requirements for the executable are that it takes as command line
    arguments the function parameters, and spits out to stdout the function
    evaluation at that point.  Nothing else should go to stdout.
    """

    from mrs.param import Param

    _params = dict(
            externfunc=Param(doc='External function (must be an executable file)',
                default='')
            )

    def setup(self):
        if self.externfunc == '':
            raise Exception('Must supply an external function!')
        super(ExternalFunction, self).setup()
        self._set_constraints(((-50,50),) * self.dims)
        self.ERROR = float('inf')

    def __call__( self, vec ):
        import subprocess
        command = [self.externfunc]
        for x in vec:
            command.append(str(x))
        func_proc = subprocess.Popen(tuple(command), stdout=subprocess.PIPE)
        retcode = func_proc.wait()
        if retcode < 0:
            return self.ERROR
        retval = func_proc.stdout.readline()
        return float(retval)
