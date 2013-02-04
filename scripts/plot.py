#!/usr/bin/env python

from __future__ import division
import math
import optparse
from itertools import islice

from psodata import PSOData
from evilplot import Plot, Points, RawData

MAX_BAR_SAMPLES = 40
MAX_SAMPLES = 500

parser = optparse.OptionParser()
parser.add_option('--print', dest='print_page', action='store_true',
        help='Send to the printer')
opts, args = parser.parse_args()

if not args:
    parser.error('Log file not specified.')

plot = Plot()
plot.ylogscale = 10
plot.xlabel = 'Function Evaluations'
plot.ylabel = 'Best Function Value'

for filename in args:
    data = PSOData(open(filename))
    trim = int(len(data) / 10)
    points = []
    bars = []
    iterations = len(data[0])
    samples_step = int(math.ceil(iterations / MAX_SAMPLES))
    bar_samples_step = int(math.ceil(iterations / MAX_BAR_SAMPLES))
    for iteration in islice(data[0], 0, None, samples_step):
        points.append((iteration, data.average(iteration)))
    for iteration in islice(data[0], 0, None, bar_samples_step):
        low, med, high = data.statistics(iteration, trim)
        bars.append((iteration, med, low, high))
    plot.append(Points(points, title=filename, style='lines'))
    plot.append(RawData(bars, style='errorbars'))

plot.show()

if opts.print_page:
    plot.print_page()

# vim: et sw=4 sts=4
