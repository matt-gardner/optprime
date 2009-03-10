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
        directory, _ = os.path.split(module.__file__)
        # Start both processes at once so they can run concurrently.
        rev_parse = Popen(['git', 'rev-parse', '--short', 'HEAD'],
                cwd=directory, stdout=PIPE)
        diff = Popen(['git', 'diff', '--quiet', 'HEAD'], cwd=directory)

        # git rev-parse prints the commit name on stdout.
        stdout, _ = rev_parse.communicate()
        self.commit = stdout.strip()
        # git diff returns 1 if differences and 0 if no differerces.
        self.dirty = bool(diff.wait())


# vim: et sw=4 sts=4
