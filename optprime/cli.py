"""Helper functions for AMLPSO command-line utilities."""


class GitStatus(object):
    """Finds Git status for a Python module.

    Looks up the file that the module is defined in and performs status
    checks on its Git repository.

    Attributes:
        commit: short form of the commit id (str)
        dirty: whether there were differences since the last commit (bool)
    """
    def __init__(self, module):
        from subprocess import Popen, PIPE
        import os, subprocess
        self.directory, _ = os.path.split(module.__file__)
        if not self.directory:
            self.directory = '.'
        # Start both processes at once so they can run concurrently.
        rev_parse = Popen(['git', 'rev-parse', '--short', 'HEAD'],
                cwd=self.directory, stdout=PIPE)
        diff = Popen(['git', 'diff', '--quiet', 'HEAD'], cwd=self.directory)

        # git rev-parse prints the commit name on stdout.
        stdout, _ = rev_parse.communicate()
        self.commit = stdout.strip()
        # git diff returns 1 if differences and 0 if no differerces.
        self.dirty = bool(diff.wait())

    def __str__(self):
        if self.dirty:
            dirty = 'dirty'
        else:
            dirty = 'clean'
        return 'commit %s (%s)' % (self.commit, dirty)


# vim: et sw=4 sts=4
