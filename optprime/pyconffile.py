"""Config

Loads a config file written in Python.  This allows you to perform
configuration by using a Turing-complete language.  Just make sure you
understand that such a config file can do _anything_, so the file must be
completely trusted.
"""

import sys
from UserDict import DictMixin

# Historical note: we used to make Config inherit from dict.  That's no good,
# because '__builtins__' gets in the way.  We used to delete '__builtins__'
# from the dict, but then you can't call functions from the config file.

class ConfFile(DictMixin):
    """Import configuration from a file.

    After reading, the dictionary is read-only.  You can use this either as a
    dict or as an object with properties.

    Load this file into a ConfFile:
    >>> c = ConfFile(__file__)
    >>> c.the_answer
    42
    >>> c['the_answer']
    42
    >>> 'the_answer' in c
    True
    >>>

    Builtins is completely hidden from view, even it actually exists in the
    underlying dictionary.
    >>> '__builtins__' in c
    False
    >>> try:
    ...   c['__builtins__']
    ... except AttributeError:
    ...   print "__builtins__ is hidden"
    __builtins__ is hidden
    >>>
    """
    def __init__(self, fname=None):
        self.data = {}
        if fname:
            self.readpython(fname)

    def readpython(self, fname):
        execfile(fname, self.data)

    def __getitem__(self, name):
        if name is '__builtins__':
            raise AttributeError
        else:
            return self.data[name]

    def keys(self):
        keylist = self.data.keys()
        keylist.remove('__builtins__')
        return keylist

    def __contains__(self, name):
        if name is '__builtins__':
            return False
        else:
            return (name in self.data)

    def __getattr__(self, name):
        if name is '__builtins__':
            raise AttributeError
        else:
            try:
                return self.data[name]
            except KeyError:
                raise AttributeError("'ConfFile' object has no attribute '%s'"
                        % name)

    def write_java_properties(self, f=sys.stdout):
        """Write out a Java properties file

        This is an ugly XML file that can be imported within Java.
        """
        from xml.sax.saxutils import escape
        isfile = True
        try:
            if isinstance(f, (str,unicode)):
                isfile = False
                f = open(f, 'w')

            print >> f, \
                    '<?xml version="1.0" encoding="UTF-8"?>' \
                    '<!DOCTYPE properties SYSTEM ' \
                    '"http://java.sun.com/dtd/properties.dtd">'
            print >> f, '<properties>'
            for k, v in self.iteritems():
                try:
                    print >> f, '\t<entry key="%s">%s</entry>' % (
                            k,escape(str(v)))
                except TypeError, e:
                    pass
            print >> f, '</properties>'

        finally:
            if not isfile:
                f.close()

the_answer = 42

if __name__ == '__main__':
    import doctest
    doctest.testmod()
