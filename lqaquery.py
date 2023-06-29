#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import liteqa

########################################################################

def print_usage():
	print(f"""
{sys.argv[0]} <DATA_LQA_PATH> ['<QUERY>'] [values] [points]

QUERY OPERATORS
---------------
<QUERY> = 'x_0 < VARIABLE'
<QUERY> = 'x_0 <= VARIABLE'
<QUERY> = 'VARIABLE <= x_1'
<QUERY> = 'x_0 < VARIABLE < x_1'
<QUERY> = 'x_0 <= VARIABLE < x_1'
<QUERY> = 'x_0 <= VARIABLE <= x_1'
<QUERY> = 'x_0 < VARIABLE <= x_1'

QUERY FUNCTIONS ON SAME VARIABLE
--------------------------------
<QUERY> = 'VARIABLE > MIN()'
<QUERY> = 'VARIABLE > MAX()'
<QUERY> = 'VARIABLE > MEAN()'
<QUERY> = 'VARIABLE > MEDIAN()'
<QUERY> = 'VARIABLE > QUANTILE(0.99)'

QUERY FUNCTION ON OTHER VARIABLE
--------------------------------
<QUERY> = 'VARIABLE1 > MIN(DATA_LQA_PATH/VARIABLE2)'
<QUERY> = 'VARIABLE1 > MAX(DATA_LQA_PATH/VARIABLE2)'
<QUERY> = 'VARIABLE1 > MEAN(DATA_LQA_PATH/VARIABLE2)'
<QUERY> = 'VARIABLE1 > MEDIAN(DATA_LQA_PATH/VARIABLE2)'
<QUERY> = 'VARIABLE1 > QUANTILE(DATA_LQA_PATH/VARIABLE2, 0.99)'
""")

########################################################################

if len(sys.argv) < 3:
	print_usage()
	sys.exit(1)

archive = sys.argv[1]
query = sys.argv[2]

points = "points" in sys.argv[3:]
values = "values" in sys.argv[3:]

show = lambda what, flag: ("with" if flag else "without") + " " + what
print("# QUERY:", query, show("values", values), show("points", points))

########################################################################

lqa = liteqa.liteqa()
qpars = liteqa.parse_query(archive, query, index=True, liteqa=lqa)
fnames = []
xrngs = []
xopts = []
invts = []
for alias, varn, xrng, xopt, nots in qpars:
	fnames.append(varn)
	xrngs.append(xrng)
	xopts.append(xopt)
	invts.append(nots)

########################################################################

if None in fnames:
	print("# one or more arrays for query not found in", archive)
else:
	if values and points:
		size, qrngs, X, P = lqa.get_index_ranges(fnames, xrngs, xopts, invts, values=True, points=True)
		count = len(X)
	elif values:
		size, qrngs, X = lqa.get_index_ranges(fnames, xrngs, xopts, invts, values=True, points=False)
		count = len(X)
	elif points:
		size, qrngs, P = lqa.get_index_ranges(fnames, xrngs, xopts, invts, values=False, points=True)
		count = len(P)
	else:
		size, qrngs, count = lqa.get_index_ranges(fnames, xrngs, xopts, invts, values=False, points=False)

	for fname, qrng in zip(fnames, qrngs):
		print("#", "bin_range", fname[0], "=", qrng)

	if values and points:
		for x, p in zip(X, P):
			print(x, p)
	elif values: print(X)
	elif points: print(P)

	print("# count", count, "of", "%d**3" % size, "=", size**3)
	print("# density", "%.05f" % (count/size**3*100), "%")

print("")


