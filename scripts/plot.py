#!/usr/bin/env python

from __future__ import division
import optparse

from amlpso.psodata import PSOData
from evilplot import Plot, Points, RawData

parser = optparse.OptionParser()
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
    for iteration in data[0]:
        points.append((iteration, data.average(iteration)))
        low, med, high = data.statistics(iteration, trim)
        bars.append((iteration, med, low, high))
    plot.append(Points(points, title=filename, style='lines'))
    plot.append(RawData(bars, style='errorbars'))

plot.show()

# vim: et sw=4 sts=4
