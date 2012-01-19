#!/usr/bin/env python
from __future__ import division

import math

def logrange(start, step, count):
    """Returns count integers from start using a step-based log scale.

    >>> logrange(10, 10, 4)
    [10, 100, 1000, 10000]
    >>> logrange(10, math.e, 4)
    [10, 27, 74, 201]
    >>> logrange(10, 10 ** .5, 4)
    [10, 32, 100, 316]
    >>>
    """
    return [int(round(start * (step ** x))) for x in xrange(0, count)]

def try_makedirs(path):
    try:
        os.makedirs(path)
    except OSError, e:
        import errno
        if e.errno != errno.EEXIST:
            raise

def run_batch(command, pwd, stdout_path, stderr_path):
    """Runs the command (given as a string) within batch.

    The output paths are appended with a mktemp-style suffix.  The given
    pwd is used as the working directory of the command.
    """
    outdir = os.path.dirname(stdout_path)
    outfile = os.path.basename(stdout_path)
    try_makedirs(outdir)

    errdir = os.path.dirname(stderr_path)
    errfile = os.path.basename(stderr_path)
    try_makedirs(errdir)

    command += ' >"$OUT_FILENAME" 2>"$ERR_FILENAME"\n'

    proc = subprocess.Popen(('batch',), stdin=subprocess.PIPE)
    proc.stdin.write('OUT_FILENAME="$(mktemp --tmpdir=%s %s.XXX)"\n'
            % (outdir, outfile))
    proc.stdin.write('ERR_FILENAME="$(mktemp --tmpdir=%s %s.XXX)"\n'
            % (errdir, errfile))
    proc.stdin.write('cd %s\n' % pwd)
    proc.stdin.write(command)
    proc.stdin.close()
    proc.wait()


# vim: et sw=4 sts=4
