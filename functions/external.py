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
    set to true, the function must take arguments from stdin instead of as
    command line arguments.  In that case, the process is never closed, so it 
    should run faster.  The executable should be a loop that waits on stdin and 
    spits out to stdout.  That behavior can be overridden with dont_persist.
    """

    from mrs.param import Param

    _params = dict(
            exe=Param(default='',
                doc='External function (must be an executable file)'),
            stdin=Param(type='bool',
                doc='Get parameters from stdin instead of as commandline '
                    'arguments'),
            dont_persist=Param(type='bool',
                doc='Re-open the executable for every call, instead of keeping'
                    ' it open (only if using stdin).  If the function'
                    'evaluation takes a significant amount of time, do this.'),
            constraints=Param(default='',
                doc='Constraints file, formatted as one "low,high" line for '
                    'each dimension'),
            quiet=Param(type='bool',
                doc="Don't print messages to and from the external program"),
            )

    def setup(self):
        import subprocess
        if self.exe == '':
            raise Exception('Must supply an external function!')
        super(ExternalFunction, self).setup()
        if self.constraints == '':
            self._set_constraints(((-50,50),) * self.dims)
        else:
            f = open(self.constraints)
            lines = f.readlines()
            constraints = [map(float,line.split(',')) for line in lines]
            constraints = map(tuple, constraints)
            self._set_constraints(tuple(constraints))
        self.ERROR = float('inf')
        if self.stdin and not self.dont_persist:
            self.func_proc = subprocess.Popen((self.exe), stdout=subprocess.PIPE,
                stdin=subprocess.PIPE)

    def __call__(self, vec):
        import sys
        import subprocess
        if self.stdin:
            if self.dont_persist:
                self.func_proc = subprocess.Popen((self.exe), 
                        stdout=subprocess.PIPE, stdin=subprocess.PIPE)
            self.func_proc.stdin.write(' '.join(repr(x) for x in vec)+' ')
            retval = self.func_proc.stdout.readline()
        else:
            command = [self.exe]
            for x in vec:
                command.append(repr(x))
            func_proc = subprocess.Popen(tuple(command), stdout=subprocess.PIPE)
            retcode = func_proc.wait()
            if retcode != 0:
                raise ValueError('External program returned a nonzero exit code')
            retval = func_proc.stdout.readline()
        if not self.quiet:
            print >> sys.stderr, 'Sent to program:',' '.join(repr(x) for x in vec)+' '
            print >> sys.stderr, 'Received from program:',retval
        return float(retval)


