#------------------------------------------------------------------------------
class _Arg(object):
    __slots__ = (
            'name',
            'default',
            'description',
            'converter',
            'cls',
            'value',
            )
    def __init__( self, name, default=None, desc=None, converter=None ):
        self.name = name
        self.default = default
        self.value = default
        self.description = desc
        if converter is None:
            if default is None:
                converter = lambda x:x # punt
            else:
                converter = type(default) # use the type of the default
        self.converter = converter

    def copy( self ):
        return _Arg(self.name, self.default, self.description, self.converter)

    def __str__( self ):
        return "%s -- %s (%s)" % (self.name, self.description, self.default)

#------------------------------------------------------------------------------
class _MVarArgs(type):
    def __init__( cls, name, bases, dct ):
        super(_MVarArgs,cls).__init__( name, bases, dct )
        args = {}
        argstrs = []
        for c in cls.__mro__:
            if not isinstance(c, _MVarArgs):
                # We've reached a place where we shouldn't proceed
                break
            # We could use hasattr here, but it searches the hierarchy, and we
            # don't want to do that.  So, we test the class's __dict__ directly
            if '_args' in c.__dict__:
                for argtuple in c._args:
                    arg = _Arg(*argtuple)
                    arg.cls = c
                    if arg.name in args:
                        raise NameError("'%s' is a duplicate name" % arg.name)
                    argstrs.append(str(arg))
                    args[arg.name] = arg
        cls.args = args
        if args:
            argstr = "Optional keyword arguments for __init__:\n%s" % (
                    "\n".join(argstrs),
                    )
            if cls.__doc__ is None:
                cls.__doc__ = argstr
            else:
                cls.__doc__ += "\n" + argstr

#------------------------------------------------------------------------------
class VarArgs(object):
    __metaclass__ = _MVarArgs

    def __init__( self, *args, **kargs ):
        super(VarArgs,self).__init__( *args, **kargs )
        self.__parse_args( *args, **kargs )

    def __parse_args( self, *args, **kargs ):
        for a in self.args.itervalues():
            setattr(self, a.name, a.converter(kargs.get(a.name,a.default)))

#------------------------------------------------------------------------------
