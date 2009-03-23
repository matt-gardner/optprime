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
    evaluation at that point.  Nothing else should go to stdout.  If stdin is
    set to true, the function must take arguments from stdin instead.  In that
    case, the process is never closed, so it should run faster.  The executable
    should be a loop that waits on stdin and spits out to stdout.
    """

    from mrs.param import Param

    _params = dict(
            externfunc=Param(doc='External function (must be an executable file)',
                default=''),
            stdin=Param(doc='Get parameters from stdin instead of as commandline '+
                'arguments',
                type='int',
                default=0),
            open_every_time=Param(doc='Open the executable every time, instead of '+
                'assuming that it will stay open (only if using stdin)',
                type='int',
                default=0),
            constraintsfile=Param(doc='File to get a set of constraints from - be sure'+
                ' you have the right format! (one line for each dimension: low,high)',
                default=''),
            )

    def setup(self):
        import subprocess
        if self.externfunc == '':
            raise Exception('Must supply an external function!')
        super(ExternalFunction, self).setup()
        if self.constraintsfile == '':
            self._set_constraints(((-50,50),) * self.dims)
        else:
            f = open(self.constraintsfile)
            lines = f.readlines()
            constraints = [map(float,line.split(',')) for line in lines]
            constraints = map(tuple, constraints)
            self._set_constraints(tuple(constraints))
        self.ERROR = float('inf')
        if self.stdin and not self.open_every_time:
            self.func_proc = subprocess.Popen((self.externfunc), stdout=subprocess.PIPE,
                stdin=subprocess.PIPE)

    def __call__(self, vec):
        import subprocess
        if self.stdin:
            if self.open_every_time:
                self.func_proc = subprocess.Popen((self.externfunc), 
                        stdout=subprocess.PIPE, stdin=subprocess.PIPE)
            self.func_proc.stdin.write(' '.join(str(x) for x in vec)+' ')
            retval = self.func_proc.stdout.readline()
        else:
            command = [self.externfunc]
            for x in vec:
                command.append(str(x))
            func_proc = subprocess.Popen(tuple(command), stdout=subprocess.PIPE)
            retcode = func_proc.wait()
            if retcode < 0:
                return self.ERROR
            retval = func_proc.stdout.readline()
        return float(retval)
