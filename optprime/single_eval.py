#!/usr/bin/env python

from mrs import param
from mrs.cli import option_parser

parser = option_parser()

parser.add_option('-f','--func', metavar='FUNCTION',
        dest='func', action='extend', search=['functions'],
        help='Function to optimize',
        default='sphere.Sphere',
        )
parser.add_option('-v','--vec', metavar='VEC',
        dest='vec',
        help='Position to evaluate (example: -v 2.3,4.6)',
        )
opts, args = parser.parse_args()

function = param.instantiate(opts, 'func')
function.setup()
vec = [float(v) for v in opts.vec.split(',')]
value = function(vec)
function.master_log()
print value

# vim: et sw=4 sts=4
