import os
import glob
import sys
import math
import struct
import numpy as np
from collections import namedtuple

import zstd
import pyfastpfor
from hilbertcurve.hilbertcurve import HilbertCurve

########################################################################

liteqa_file_info = namedtuple('liteqa_file_header', ["ftype", "cube", "size"])

FNAME_GRID = "data.lqa/array"
FNAME_INDEX = "data.lqa/array.i"

########################################################################

HILBMAP = {}	#< hilbert curve cache

########################################################################

def get_distfile(name):
	import importlib
	spec = importlib.util.find_spec("liteqa")
	return "/".join(spec.origin.split("/")[0:-1]) + "/" + name

def get_array_name(query):
	query = query.split("/")[-1]
	if query[-2:] in [".i", ".g", ".*", ".3"]: return query[:-2]
	return query

def get_archive_data(query, index=False):
	if len(query) == 0: return None
	if ":" in query: query = query.split(":")[0]
	if query[-2:] in [".i", ".g"]:
		if os.path.exists(query):
			return [query]
		print("WARNING:", query, "does not exist")
		return [None]
	elif query[-2:] in [".*", ".3"]:
		query = query[:-2]
		files = []
		for i in range(3):
			fname = query + str(i) + ".g"
			if not os.path.exists(fname):
				print("WARNING:", fname, "does not exist")
				return [None]
			files.append(fname)
		return files
	fname = query + (".i" if index else ".g")
	if os.path.exists(fname):
		return [fname]
	print("WARNING:", fname, "does not exist")
	return [None]

def parse_fun_args(args, defdata):
	return defdata, args

def calc_quantile(data, freq, q):
	order = np.argsort(data)
	cdf = np.cumsum(freq[order])
	val = int(np.sum(freq)*q + 0.5)
	return data[order][np.searchsorted(cdf, val)]

def parse_fun(text, dataset, index, liteqa):
	if "(" in text and text.endswith(")"):
		fun, args = tuple([i.strip() for i in text.split(")")[0].split("(")[0:2]])
		if not "," in args: param = [args]
		else:
			dataset, *param = tuple([i.strip() for i in args.split(",")])
			dataset = get_archive_data(dataset, index=index)
		res = liteqa.get_index_table(dataset)
		if res is None: return text
		size, xrng, data, binn, freq = res
		fun = fun.upper()
		if fun == "QUANTILE": return calc_quantile(data, freq, float(param[0]))
		elif fun == "MEAN": return np.sum(data*freq)/np.sum(freq)
		elif fun == "MEDIAN": return calc_quantile(data, freq, 0.5)
		elif fun == "MIN": return np.min(data)
		elif fun == "MAX": return np.max(data)
		else: raise ValueError("unknown function, got", text)
		return "1"
	return text

def parse_and_as(text, sep=","):
	text = text.replace("\n", " ")
	text = text.replace("\t", " ")
	text = text.replace(" AND ", sep)
	text = text.replace(" And ", sep)
	text = text.replace(" aNd ", sep)
	text = text.replace(" anD ", sep)
	text = text.replace(" ANd ", sep)
	text = text.replace(" AnD ", sep)
	text = text.replace(" aND ", sep)
	text = text.replace(" and ", sep)
	text = text.replace(" AS ", ":")
	text = text.replace(" As ", ":")
	text = text.replace(" aS ", ":")
	text = text.replace(" as ", ":")
	text = [tuple((i + ":").split(":")[0:2]) for i in text.split(sep)]
	return [(i.strip(), j.strip()) for i, j in text]

def split_condexp(text, splitc, i):
	varn = None
	xrng = [None, None]
	xopt = [1, 1]
	toks = text.split(splitc)
	if len(toks) == 1:
		varn = toks[0]
	elif len(toks) == 2:
		varn = toks[0]
		xrng[i] = toks[1].replace("=", "")
		xopt[i] = 1 if "=" in toks[1] else 0
	elif len(toks) == 3:
		i, j = (0 if i else 1, 1 if i else 0)
		xrng[i] = toks[0]
		varn = toks[1].replace("=", "")
		xopt[i] = 1 if "=" in toks[1] else 0
		xrng[j] = toks[2].replace("=", "")
		xopt[j] = 1 if "=" in toks[2] else 0
	else: ValueError("wrong condexp", text)
	return varn, xrng, xopt

def parse_condexp(text):
	text = (" " + text + " ").replace(" NOT ", "!")
	nots = ("!" in text)
	text = text.replace("!", "")
	toks = text.replace(" ", "")
	if "<" in toks: varn, xrng, xopt = split_condexp(toks, "<", 1)
	elif ">" in toks: varn, xrng, xopt = split_condexp(toks, ">", 0)
	elif "=" in toks:
		toks = toks.replace("==", "=").split("=")
		varn, xrng, xopt = (toks[0], (toks[1], toks[1]), (1, 1))
	else: varn, xrng, xopt = (toks, [None, None], [None, None])
	return varn, xrng, xopt, nots

def parse_query(data, query, resolv=True, index=False, dtype=float, liteqa=None):
	# DATA: path/dir : A , path/dir : B AND path/dir AS C
	# QUERY: $A.var : vara, $B.var : varb AND $C.var AS varc
	# QUERY: NOT $A.var =  0 : vara , $B.var != 0 AND 0 < $C.var < 0 AS varc
	# QUERY: !   $A.var =  0 : vara , $B.var >  0 AND 0<= $C.var < 0 AS varc
	# QUERY:     $A.var =  0 : vara , $B.var <  0 AND !0< $C.var <=0 AS varc
	# QUERY:     $A.var == 0 : vara , $B.var >= 0 AND NOT 0<$C.var<0 AS varc
	if query is None or len(query) == 0: return []
	data = [] if data is None else parse_and_as(data)
	parsed = []
	for expr, alias in parse_and_as(query, sep=";" if index else ","):
		varn, xrng, xopt, nots = parse_condexp(expr)
		for path, name in data:
			if varn.startswith("$"): varn = varn.replace("$" + name + ".", path + "/")
			elif len(name) == 0:
				varn = path + "/" + varn
				break
			xrng = [i if i is None else i.replace("$" + name + ".", path + "/") for i in xrng]
		if resolv:
			if len(alias) == 0: alias = get_array_name(varn)
			varn = get_archive_data(varn, index=index)
		if not liteqa is None: xrng = [i if i is None else parse_fun(i, varn, index, liteqa) for i in xrng]
		xrng = [i if i is None else dtype(i) for i in xrng]
		parsed.append((alias, varn, xrng, xopt, nots))
	return parsed

def parse_asset(data, query):
	if query is None or len(query) == 0: return []
	data = [] if data is None else parse_and_as(data)
	data = [(i if "/" in i else i.replace(".", "/"), j) for i, j in data if len(j) > 0]
	for path, name in data:
		query = query.replace("$" + name + ".", path + ".")
		query = query.replace("$" + name + "/", path + "/")
	return query

########################################################################

def liteqa_hilbmap(cube):
	"""hilbert linearization map"""
	global HILBMAP
	if cube in HILBMAP: return HILBMAP[cube]
	else:
		hiter = int(np.log2(cube))
		cube2 = cube**2
		cube3 = cube**3
		fname = "liteqa-hilbmap." + str(cube)
		mapdir = os.getenv("LITEQA_HILBMAP")
		fname = fname if mapdir is None else f"{mapdir}/{fname}"
		if not os.path.exists(fname):
			if not mapdir is None:
				print("# make_hilbert_ERROR", "LITEQA_MAPDIR is set and map does not exist:", fname)
				return np.zeros((cube3,), dtype=np.uint32)
			else:
				print("# make_hilbert", fname)
				hc = HilbertCurve(hiter, 3) #> p iters, n dims
				hilbmap = np.zeros((cube3,), dtype=np.uint32)
				for n, (i, j, k) in enumerate(hc.points_from_distances(range(cube3))):
					hilbmap[n] = i + j*cube + k*cube2
				fstream = open(fname, "wb")
				fstream.write(hilbmap.tobytes())
				fstream.close()
				print("# make_hilbert", fname, "done")
		else:
			# ~ print("# load_hilbert", fname)
			fstream = open(fname, "rb")
			hilbmap = np.frombuffer(fstream.read(cube3*4), dtype=np.uint32)
			fstream.close()
	HILBMAP[cube] = hilbmap
	return hilbmap

########################################################################

class liteqa:
	def __init__(self, E0=20, M=35):
		# fastpfor bit packing library
		self.pfor = pyfastpfor.getCodec('simdbinarypacking')
		# hilbert linearization cache
		self.hilbmap = []	#< hilbert curve indices
		self.hilbert = None	#< Hilbert curve translator
		# initialize liteqa_c
		self.liteqa_cinit()
		# initialize lisq
		self.lisq_init(E0, M)

########################################################################

	def liteqa_cinit(self):
		import ctypes
		lqapath = os.getenv("LITEQA_C")
		if lqapath is not None:
			if os.path.exists(lqapath):
				liteqa_c = ctypes.cdll.LoadLibrary(lqapath)
			else:
				print("# ERROR: liteqa_c.so not found as in envvar LITEQA_C:", lqapath)
				return False
		else:
			import importlib
			spec = importlib.util.find_spec("liteqa_c")
			liteqa_c = ctypes.cdll.LoadLibrary(spec.origin)

		from ctypes import c_size_t, c_uint8, c_int32, c_uint32, c_float, c_double
		from numpy.ctypeslib import ndpointer
		ndp = lambda ctype: ndpointer(ctype, flags="C_CONTIGUOUS")
		def liteqa_cdef(cobj, cret, cargs):
			cobj.restype = cret
			cobj.argtypes = cargs
			return cobj

		# LISQ - logarithmically increasing steps quantization
		self.c_lisq_float_max = liteqa_cdef(liteqa_c.lisq_float_max, c_float, [])

		# EBATE - error-bounded binning and tabular encoding
		self.c_ebate_encode = liteqa_cdef(
			liteqa_c.ebate_encode, c_size_t,
			[c_size_t, c_double, c_double, ndp(c_float), ndp(c_int32),
				ndp(c_uint32), ndp(c_uint32), ndp(c_uint32), ndp(c_uint8)]
		)
		self.c_ebate_decode = liteqa_cdef(
			liteqa_c.ebate_decode, None,
			[c_double, c_double, ndp(c_float), c_int32, ndp(c_int32), ndp(c_uint32), ndp(c_uint32), ndp(c_uint32)]
		)
		self.c_ebate_count = liteqa_cdef(
			liteqa_c.ebate_count, c_size_t,
			[c_size_t, ndp(c_int32), ndp(c_uint32), ndp(c_int32), c_int32, c_int32]
		)
		self.c_ebate_select = liteqa_cdef(
			liteqa_c.ebate_select, c_size_t,
			[c_size_t, ndp(c_int32), ndp(c_uint32), c_int32, c_int32]
		)
		self.c_ebate_merge = liteqa_cdef(
			liteqa_c.ebate_merge, c_size_t,
			[c_size_t, ndp(c_int32), ndp(c_uint32), c_size_t, ndp(c_int32), ndp(c_uint32)]
		)
		self.c_ebate_filter = liteqa_cdef(
			liteqa_c.ebate_filter, c_size_t,
			[c_size_t, ndp(c_int32), ndp(c_uint32), c_size_t, ndp(c_int32), ndp(c_uint32), c_int32, c_int32]
		)
		self.c_ebate_points = liteqa_cdef(
			liteqa_c.ebate_points, None,
			[c_double, c_double, c_uint32, ndp(c_uint32), c_size_t, ndp(c_uint32), ndp(c_int32), ndp(c_uint32), ndp(c_uint32), c_size_t, ndp(c_uint32), ndp(c_float), ndp(c_uint32)]
		)
		self.c_ebate_intersect = liteqa_cdef(
			liteqa_c.ebate_intersect, c_size_t,
			[c_size_t, ndp(c_uint32), c_size_t, c_size_t, ndp(c_int32), c_size_t, c_size_t, ndp(c_uint32), ndp(c_int32), ndp(c_uint32), ndp(c_uint32)]
		)
		self.c_ebate_intpoints = liteqa_cdef(
			liteqa_c.ebate_intpoints, None,
			[c_uint32, ndp(c_uint32), c_size_t, ndp(c_uint32), c_size_t, ndp(c_uint32), ndp(c_uint32)]
		)
		self.c_ebate_intdata = liteqa_cdef(
			liteqa_c.ebate_intdata, None,
			[ndp(c_double), ndp(c_double), c_size_t, c_size_t, ndp(c_int32), c_size_t, ndp(c_float)]
		)

		# GLATE - grid linearization and truncation encoding
		self.c_glate_encode = liteqa_cdef(
			liteqa_c.glate_encode, None,
			[c_size_t, c_size_t, c_double, c_double, ndp(c_float),
				ndp(c_uint32), ndp(c_uint32), ndp(c_uint32)]
		)
		self.c_glate_decode = liteqa_cdef(
			liteqa_c.glate_decode, None,
			[c_size_t, c_size_t, c_double, c_double, ndp(c_float),
				ndp(c_uint32), ndp(c_uint32), ndp(c_uint32)]
		)

		return True

########################################################################

	def hilbert_init(self, hiter):
		#> p iters, n dims -> m = 2**(3*p)
		self.hilbert = HilbertCurve(max(1, hiter), 3)
		return 2**(3*hiter)	#< num_cubes

	def hilbert_cube_origin(self, cube, num):
		q = self.hilbert.point_from_distance(num)
		return np.array((q[2], q[1], q[0]), dtype=np.uint32)*cube

	def hilbert_cube_get(self, cube, num, X):
		q = self.hilbert.point_from_distance(num)
		return X[
			q[2]*cube:(q[2]+1)*cube,
			q[1]*cube:(q[1]+1)*cube,
			q[0]*cube:(q[0]+1)*cube
		].flatten(order="F")

	def hilbert_cube_set(self, hiter, cube, num, X, Y, origin=(0, 0, 0)):
		q = self.hilbert.point_from_distance(num)
		q = (q[0] - origin[2], q[1] - origin[1], q[2] - origin[0])
		edge = 2**hiter*cube
		X[
			(q[2]*cube) % edge:((q[2] + 1)*cube - 1) % edge + 1,
			(q[1]*cube) % edge:((q[1] + 1)*cube - 1) % edge + 1,
			(q[0]*cube) % edge:((q[0] + 1)*cube - 1) % edge + 1
		] = Y.reshape((cube, cube, cube), order="F")

	def hilbert_cube_list(self, hiter, cube, extent):
		edge = 2**hiter
		cidx = []
		for k in         range(extent[0]//cube, extent[1]//cube + 1):
			k = k % edge
			for j in     range(extent[2]//cube, extent[3]//cube + 1):
				j = j % edge
				for i in range(extent[4]//cube, extent[5]//cube + 1):
					i = i % edge
					cidx.append(self.hilbert.distance_from_point((i, j, k)))
		return cidx

	def hilbert_cube_points(self, hiter, points):
		edge = 2**hiter
		cidx = []
		for k, j, i in points:
			k = k % edge
			j = j % edge
			i = i % edge
			cidx.append(self.hilbert.distance_from_point((i, j, k)))
		return cidx

########################################################################

	def lisq_init(self, E0, M):
		"""LISQ quantization setup"""
		self.E0 = E0
		self.M = M
		self.omega = math.pow(2., 1./self.M)
		self.omega = (self.omega - 1.)/(self.omega + 1.)
		self.delta = math.pow(2, -self.E0)
		self.log_delta = np.log(self.delta)
		self.log_omega = math.log(1. + self.omega) - math.log(1. - self.omega)
		self.div_omega = (1. + self.omega)/(1. - self.omega)
		self.lisq_nmax = self.lisq_stepnum(self.c_lisq_float_max()) - 1
		self.lisq_nmin = -self.lisq_nmax
		return self.delta, self.omega

	def lisq_stepnum(self, x):
		"""LISQ value quantization"""
		if np.fabs(x) < self.delta: return int(0)
		if x > 0: return int((np.log(np.fabs(x)) - self.log_delta)/self.log_omega + 1.5);
		return -int((np.log(np.fabs(x)) - self.log_delta)/self.log_omega + 1.5);

	def lisq_stepfun(self, n):
		"""LISQ value reconstruction"""
		if n == 0: return 0.
		if n > 0: return self.delta*self.div_omega**(n - 1)
		return -self.delta*self.div_omega**(-n - 1)

	def lisq_range(self, nmin, nmax):
		return (self.lisq_stepfun(nmin + 1), self.lisq_stepfun(nmax - 1))

	def lisq_translate(self, xrng=None, xopt=(1, 1)):
		if xrng is None:
			nmin = self.lisq_nmin
			nmax = self.lisq_nmax
		else:
			if not xrng[0] is None and xrng[0] == xrng[1]:
				if xopt[0] == 0 and xopt[1] == 0:
					nmin = self.lisq_stepnum(xrng[0]) - 2
					nmax = self.lisq_stepnum(xrng[1]) + 2
				else:
					nmin = self.lisq_stepnum(xrng[0]) - xopt[0] - 1 + xopt[1]
					nmax = self.lisq_stepnum(xrng[1]) + xopt[1] + 1 - xopt[0]
				# ~ nmin, nmax = min([nmin, nmax]), max([nmin, nmax])
			else:
				nmin = self.lisq_nmin if xrng[0] is None else (self.lisq_stepnum(xrng[0]) - xopt[0])
				nmax = self.lisq_nmax if xrng[1] is None else (self.lisq_stepnum(xrng[1]) + xopt[1])
		return nmin, nmax

########################################################################

	def get_dataset_info(self, path):
		ifile = glob.glob(path + "/*/*.i")
		gfile = glob.glob(path + "/*/*.g")
		archive = list(set([i.split("/")[-2] for i in gfile]))
		archive.extend(list(set([i.split("/")[-2] for i in ifile])))
		grid = list(set([i.split("/")[-1].split(".")[0] for i in gfile]))
		index = list(set([i.split("/")[-1].split(".")[0] for i in ifile]))
		return archive, index, grid

	def get_file_info(self, fnames):
		fstream = open(fnames[0], "rb")
		ftype, hiter, cube, E0, M = self.read_header(fstream)
		return liteqa_file_info(ftype, cube, 2**hiter*cube)

	def get_grid_extent(self, fnames, extent=None, center=None):
		fstream = open(fnames[0], "rb")
		ftype, hiter, cube, E0, M = self.read_header(fstream)
		if extent is None:
			grid_size = 2**hiter*cube
			return (0, grid_size - 1, 0, grid_size - 1, 0, grid_size - 1)
		edge = 2**hiter
		if center is None:
			return (
				0, min(edge, extent[1] - extent[0] + 1)*cube - 1,
				0, min(edge, extent[3] - extent[2] + 1)*cube - 1,
				0, min(edge, extent[5] - extent[4] + 1)*cube - 1
			)
		center = (int(center[0]/cube), int(center[1]/cube), int(center[2]/cube))
		return (
			(center[0] + extent[0])*cube, (center[0] + min(extent[0] + edge, extent[1] + 1))*cube - 1,
			(center[1] + extent[2])*cube, (center[1] + min(extent[2] + edge, extent[3] + 1))*cube - 1,
			(center[2] + extent[4])*cube, (center[2] + min(extent[4] + edge, extent[5] + 1))*cube - 1
		)

	def get_grid_cubes(self, fnames, points, extent=(0, 0, 0)):
		fstream = open(fnames[0], "rb")
		ftype, hiter, cube, E0, M = self.read_header(fstream)
		points = np.unique(np.array(points/cube, dtype=np.int32), axis=0)
		cubes = np.copy(points)
		for a in         range(extent[0], extent[1] + 1):
			for b in     range(extent[2], extent[3] + 1):
				for c in range(extent[4], extent[5] + 1):
					if a == 0 and b == 0 and c == 0: continue
					newpts = points + (a, b, c)
					cubes = np.unique(np.vstack([cubes, newpts]), axis=0)
		self.hilbert_init(hiter)
		return self.hilbert_cube_points(hiter, cubes)

	def get_grid_data(self, fnames, extent=None, cube_list=None):
		if len(fnames) == 1:
			return self.decompress(fname=fnames[0], extent=extent, cube_list=cube_list)
		data = None
		for i, f in enumerate(fnames):
			X = self.decompress(fname=f, extent=extent, cube_list=cube_list)
			if data is None: data = np.zeros(X.shape + (3,), dtype=np.float32, order="F")
			data[:, :, :, i] = X
		return data

	def get_index_count(self, fnames, xrng=None, xopt=(1, 1), inv=False):
		size, xrng, C, X = self.query_count(xrng=xrng, xopt=xopt, inv=inv, fname=fnames[0])
		return size, xrng, C, X

	def get_index_table(self, fnames, xrng=None, xopt=(1, 1), inv=False):
		size, xrng, X, B, N = self.query_table(xrng=xrng, xopt=xopt, inv=inv, fname=fnames[0])
		return size, xrng, X, B, N

	def get_index_stats(self, fnames, fun="quantile", param=[], xrng=None, xopt=(1, 1), inv=False):
		size, xrng, data, binn, freq = self.query_table(xrng=xrng, xopt=xopt, inv=inv, fname=fnames[0])
		fun = fun.lower()
		if fun == "quantile": return calc_quantile(data, freq, float(param[0]))
		elif fun == "mean": return np.sum(data*freq)/np.sum(freq)
		elif fun == "avg": return np.sum(data*freq)/np.sum(freq)
		elif fun == "median": return calc_quantile(data, freq, 0.5)
		elif fun == "min": return np.min(data)
		elif fun == "max": return np.min(data)
		elif fun == "count": return len(data)
		else: raise ValueError("unknown function, got", text)

	def get_index_points(self, fnames, xrng, xopt=(1, 1), inv=False):
		size, xrng, X, P = self.query_points(xrng=xrng, xopt=xopt, inv=inv, fname=fnames[0])
		return size, xrng, X, P

	def get_index_ranges(self, fnames, xrngs, xopts, invs, values=False, points=False):
		# ~ return size, xrng, C
		# ~ return size, xrng, X
		# ~ return size, xrng, P
		# ~ return size, xrng, X, P
		return self.query_ranges(xrngs=xrngs, xopts=xopts, invs=invs, fnames=[i[0] for i in fnames], values=values, points=points)

########################################################################

	def write_header(self, fstream, ftype, hiter, cube):
		fstream.write(struct.pack('B', ord(ftype)))
		fstream.write(struct.pack('i', hiter))
		fstream.write(struct.pack('i', cube))
		fstream.write(struct.pack('i', self.E0))
		fstream.write(struct.pack('i', self.M))
		return 1 + 4*4 # header_len in int32

	def read_header(self, fstream):
		ftype = chr(struct.unpack('B', fstream.read(1))[0])
		hiter = struct.unpack('i', fstream.read(4))[0]
		cube =  struct.unpack('i', fstream.read(4))[0]
		E0 =    struct.unpack('i', fstream.read(4))[0]
		M =     struct.unpack('i', fstream.read(4))[0]
		return ftype, hiter, cube, E0, M

########################################################################

	def decompress(self, fname=FNAME_GRID, extent=None, cube_list=None):
		"""decompress GLATE, EBATE, zstd"""
		fstream = open(fname, "rb")
		ftype, hiter, cube, E0, M = self.read_header(fstream)
		self.lisq_init(E0, M)
		if ftype == "g": return self.decompress_grid(fstream, hiter, cube, extent=extent, cube_list=cube_list)
		if ftype == "z": return self.decompress_zstd(fstream, hiter, cube, extent=extent, cube_list=cube_list)
		if ftype == "i": return self.decompress_index(fstream, hiter, cube)
		raise ValueError("unknown file type")

########################################################################

	def compress_grid(self, X, cube=16, fname=FNAME_GRID, zlevel=2, zthreads=1, callb=None):
		"""GLATE compress"""
		assert X.shape[0] == X.shape[1]
		assert X.shape[0] == X.shape[2]
		hiter = int(np.log2(X.shape[0]//cube))
		cube3 = cube**3
		if len(self.hilbmap) != cube3: self.hilbmap = liteqa_hilbmap(cube)

		num_cubes = self.hilbert_init(hiter)
		fstream = open(fname, "wb")
		offs_pos = self.write_header(fstream, "g", hiter, cube)
		offs = np.zeros((num_cubes,), dtype=np.uint32)
		fstream.write(offs.tobytes())
		mod_cubes = max(1, num_cubes//8)
		for i in range(num_cubes):
			Y = self.hilbert_cube_get(cube, i, X)
			expo = np.zeros((cube3,), dtype=np.uint32)
			mant = np.zeros((cube3,), dtype=np.uint32)
			self.c_glate_encode(cube3, self.M, self.delta, self.omega, Y, expo, mant, self.hilbmap)

			zstdbuf = zstd.ZSTD_compress(expo.tobytes(), zlevel, zthreads)
			pforbuf = np.zeros(len(mant) + 1024, dtype=np.uint32)
			pforlen = self.pfor.encodeArray(mant, len(mant), pforbuf, len(pforbuf))

			outbuf = bytearray()
			outbuf.extend(struct.pack('i', len(zstdbuf)))
			outbuf.extend(zstdbuf)
			outbuf.extend(struct.pack('i', pforlen*4))
			outbuf.extend(pforbuf[0:pforlen].tobytes())

			offs[i] = fstream.tell()
			fstream.write(outbuf)

			if callb and not i % mod_cubes: callb(prog=int((i + 1)/num_cubes*100))

		size_last = fstream.tell()
		fstream.seek(offs_pos, 0)
		fstream.write(offs.tobytes())
		return size_last/num_cubes/cube**3/4

	def decompress_grid(self, fstream, hiter, cube, extent=None, cube_list=None):
		"""GLATE decompress"""
		cube3 = cube**3
		num_cubes = self.hilbert_init(hiter)
		if len(self.hilbmap) != cube3: self.hilbmap = liteqa_hilbmap(cube)

		if extent is None:
			grid_size = 2**hiter*cube
			X = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32, order="F")
			if cube_list is None: cube_list = range(num_cubes)
			origin = (0, 0, 0)
		else:
			X = np.zeros((extent[1] - extent[0] + 1, extent[3] - extent[2] + 1, extent[5] - extent[4] + 1), dtype=np.float32, order="F")
			if cube_list is None: cube_list = self.hilbert_cube_list(hiter, cube, extent)
			origin = (extent[0]//cube, extent[2]//cube, extent[4]//cube)

		offs = np.frombuffer(fstream.read(num_cubes*4), dtype=np.uint32)
		for i in cube_list:
			fstream.seek(offs[i], 0)
			lenzstd = struct.unpack('i', fstream.read(4))[0]
			expo = np.frombuffer(zstd.ZSTD_uncompress(fstream.read(lenzstd)), dtype=np.uint32)
			lenpfor = struct.unpack('i', fstream.read(4))[0]
			pforbuf = np.frombuffer(fstream.read(lenpfor), dtype=np.uint32)
			mant = np.zeros((cube3,), dtype=np.uint32)
			self.pfor.decodeArray(pforbuf, lenpfor//4, mant, cube3)

			Y = np.zeros((cube3,), dtype=np.float32, order="F")
			self.c_glate_decode(cube3, self.M, self.delta, self.omega, Y, expo, mant, self.hilbmap)
			self.hilbert_cube_set(hiter, cube, i, X, Y, origin=origin)
		return X

########################################################################

	def compress_zstd(self, X, cube=16, fname=FNAME_GRID, zlevel=2, zthreads=1, callb=None):
		"""zstd compress block-wise"""
		assert X.shape[0] == X.shape[1]
		assert X.shape[0] == X.shape[2]
		hiter = int(np.log2(X.shape[0]//cube))
		cube3 = cube**3
		if len(self.hilbmap) != cube3: self.hilbmap = liteqa_hilbmap(cube)

		num_cubes = self.hilbert_init(hiter)
		fstream = open(fname, "wb")
		self.write_header(fstream, "z", hiter, cube)
		dtype = (X.dtype.name + " "*(10 - len(X.dtype.name)))
		fstream.write(struct.pack('10B', *map(ord, dtype)))
		offs_pos = fstream.tell()
		offs = np.zeros((num_cubes,), dtype=np.uint32)
		fstream.write(offs.tobytes())
		mod_cubes = max(1, num_cubes//8)
		for i in range(num_cubes):
			Y = self.hilbert_cube_get(cube, i, X)
			zstdbuf = zstd.ZSTD_compress(Y.tobytes(), zlevel, zthreads)

			outbuf = bytearray()
			outbuf.extend(struct.pack('i', len(zstdbuf)))
			outbuf.extend(zstdbuf)

			offs[i] = fstream.tell()
			fstream.write(outbuf)

			if callb and not i % mod_cubes: callb(prog=int((i + 1)/num_cubes*100))

		size_last = fstream.tell()
		fstream.seek(offs_pos, 0)
		fstream.write(offs.tobytes())
		return size_last/num_cubes/cube**3/4

	def decompress_zstd(self, fstream, hiter, cube, extent=None, cube_list=None):
		"""zstd decompress block-wise"""
		dtype = struct.unpack('10B', fstream.read(10))
		dtype = "".join(map(chr, dtype)).strip()
		dtype = np.dtype(dtype)

		cube3 = cube**3
		num_cubes = self.hilbert_init(hiter)
		if len(self.hilbmap) != cube3: self.hilbmap = liteqa_hilbmap(cube)

		if extent is None:
			grid_size = 2**hiter*cube
			X = np.zeros((grid_size, grid_size, grid_size), dtype=dtype, order="F")
			if cube_list is None: cube_list = range(num_cubes)
			origin = (0, 0, 0)
		else:
			X = np.zeros((extent[1] - extent[0] + 1, extent[3] - extent[2] + 1, extent[5] - extent[4] + 1), dtype=dtype, order="F")
			if cube_list is None: cube_list = self.hilbert_cube_list(hiter, cube, extent)
			origin = (extent[0]//cube, extent[2]//cube, extent[4]//cube)

		offs = np.frombuffer(fstream.read(num_cubes*4), dtype=np.uint32)
		for i in cube_list:
			fstream.seek(offs[i], 0)
			lenzstd = struct.unpack('i', fstream.read(4))[0]
			Y = np.frombuffer(zstd.ZSTD_uncompress(fstream.read(lenzstd)), dtype=dtype)
			self.hilbert_cube_set(hiter, cube, i, X, Y, origin=origin)
		return X

########################################################################

	def compress_index(self, X, mask, cube, fname=FNAME_INDEX, callb=None):
		"""EBATE compress"""
		assert X.shape[0] == X.shape[1]
		assert X.shape[0] == X.shape[2]
		hiter = int(np.log2(X.shape[0]//cube))
		cube3 = cube**3
		if len(self.hilbmap) != cube3: self.hilbmap = liteqa_hilbmap(cube)

		num_cubes = self.hilbert_init(hiter)
		fstream = open(fname, "wb")
		offs_pos = self.write_header(fstream, "i", hiter, cube)
		offs = np.zeros((num_cubes,), dtype=np.uint32)
		fstream.write(offs.tobytes())
		mod_cubes = max(1, num_cubes//8)
		for i in range(num_cubes):
			Y = self.hilbert_cube_get(cube, i, X)
			Z = self.hilbert_cube_get(cube, i, mask)

			binn = np.zeros((cube3,), dtype=np.int32)
			bins = np.zeros((cube3,), dtype=np.uint32)
			bini = np.zeros((cube3,), dtype=np.uint32)
			nbin = self.c_ebate_encode(cube3, self.delta, self.omega, Y, binn, bins, bini, self.hilbmap, Z)
			binn = np.array(binn, dtype=np.uint32)

			pforbuf = np.zeros(cube3 + 1024, dtype=np.uint32)
			outbuf = bytearray()
			outbuf.extend(struct.pack('i', nbin))
			pforlen = self.pfor.encodeArray(binn, nbin, pforbuf, len(pforbuf))
			outbuf.extend(struct.pack('i', pforlen*4))
			outbuf.extend(pforbuf[0:pforlen].tobytes())
			outbuf.extend(bins[0:nbin].tobytes())

			bin_pos = 0
			outbin = bytearray()
			boff = np.zeros((nbin,), dtype=np.uint32)
			for j in range(nbin):
				boff[j] = len(outbuf) + nbin*4 + len(outbin)
				pforlen = self.pfor.encodeArray(np.copy(bini[bin_pos:bin_pos+bins[j]]), bins[j], pforbuf, len(pforbuf))
				outbin.extend(struct.pack('i', pforlen*4))
				outbin.extend(pforbuf[0:pforlen].tobytes())
				bin_pos += bins[j]

			outbuf.extend(boff.tobytes())
			outbuf.extend(outbin)
			offs[i] = fstream.tell()
			fstream.write(outbuf)

			if callb and not i % mod_cubes: callb(prog=int((i + 1)/num_cubes*100))

		size_last = fstream.tell()
		fstream.seek(offs_pos, 0)
		fstream.write(offs.tobytes())
		return size_last/num_cubes/cube**3/4

	def decompress_index(self, fstream, hiter, cube):
		"""EBATE decompress"""
		cube3 = cube**3
		num_cubes = self.hilbert_init(hiter)
		if len(self.hilbmap) != cube3: self.hilbmap = liteqa_hilbmap(cube)

		offs = np.frombuffer(fstream.read(num_cubes*4), dtype=np.uint32)
		grid_size = 2**hiter*cube
		X = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32, order="F")
		for i in range(num_cubes):
			fstream.seek(offs[i], 0)
			nbin = struct.unpack('i', fstream.read(4))[0]
			binn = np.zeros((nbin,), dtype=np.uint32)
			lenpfor = struct.unpack('i', fstream.read(4))[0]
			pforbuf = np.frombuffer(fstream.read(lenpfor), dtype=np.uint32)
			self.pfor.decodeArray(pforbuf, lenpfor//4, binn, nbin)
			binn = np.array(binn, dtype=np.int32)
			bins = np.frombuffer(fstream.read(nbin*4), dtype=np.uint32)
			boff = np.frombuffer(fstream.read(nbin*4), dtype=np.uint32)

			bini = np.zeros((cube3,), dtype=np.uint32)
			ibuf = np.zeros((cube3,), dtype=np.uint32)
			bin_pos = 0
			for j in range(nbin):
				lenpfor = struct.unpack('i', fstream.read(4))[0]
				pforbuf = np.frombuffer(fstream.read(lenpfor), dtype=np.uint32)
				self.pfor.decodeArray(pforbuf, lenpfor//4, ibuf, bins[j])
				bini[bin_pos:bin_pos+bins[j]] = ibuf[0:bins[j]]
				bin_pos += bins[j]

			Y = np.zeros((cube3,), dtype=np.float32, order="F")
			self.c_ebate_decode(self.delta, self.omega, Y, nbin, binn, bins, bini, self.hilbmap)
			self.hilbert_cube_set(hiter, cube, i, X, Y)
		return X

	def query_count(self, xrng=None, xopt=(1, 1), inv=False, fname=FNAME_INDEX):
		"""EBATE count points for range, return only count"""
		fstream = open(fname, "rb")
		ftype, hiter, cube, E0, M = self.read_header(fstream)
		nmin, nmax = self.lisq_translate(xrng, xopt)
		xrng = self.lisq_range(nmin, nmax)
		grid_size = 2**hiter*cube
		if nmin + 1 >= nmax: return grid_size, xrng, 0, []
		self.lisq_init(E0, M)
		num_cubes = 2**(3*hiter)

		num_vals = 0
		qrng = np.array((self.lisq_nmax, self.lisq_nmin), dtype=np.int32)
		offs = np.frombuffer(fstream.read(num_cubes*4), dtype=np.uint32)
		for i in range(num_cubes):
			fstream.seek(offs[i], 0)
			nbin = struct.unpack('i', fstream.read(4))[0]
			binn = np.zeros((nbin,), dtype=np.uint32)
			lenpfor = struct.unpack('i', fstream.read(4))[0]
			pforbuf = np.frombuffer(fstream.read(lenpfor), dtype=np.uint32)
			self.pfor.decodeArray(pforbuf, lenpfor//4, binn, nbin)
			binn = np.array(binn, dtype=np.int32)
			bins = np.frombuffer(fstream.read(nbin*4), dtype=np.uint32)
			if inv: num_vals += self.c_ebate_count_inv(nbin, binn, bins, qrng, nmin, nmax)
			else: num_vals += self.c_ebate_count(nbin, binn, bins, qrng, nmin, nmax)
		qrng = (self.lisq_stepfun(qrng[0]), self.lisq_stepfun(qrng[1]))
		return grid_size, xrng, num_vals, qrng

	def query_table(self, xrng=None, xopt=(1, 1), inv=False, fname=FNAME_INDEX):
		"""EBATE count query, return count and values"""
		fstream = open(fname, "rb")
		ftype, hiter, cube, E0, M = self.read_header(fstream)
		nmin, nmax = self.lisq_translate(xrng, xopt)
		xrng = self.lisq_range(nmin, nmax)
		grid_size = 2**hiter*cube
		if nmin + 1 >= nmax: return grid_size, xrng, [], [], []
		self.lisq_init(E0, M)
		num_cubes = 2**(3*hiter)

		tab_nbin = 0
		tab_binn = np.zeros((grid_size**3,), dtype=np.int32)
		tab_bins = np.zeros((grid_size**3,), dtype=np.uint32)
		offs = np.frombuffer(fstream.read(num_cubes*4), dtype=np.uint32)
		for i in range(num_cubes):
			fstream.seek(offs[i], 0)
			nbin = struct.unpack('i', fstream.read(4))[0]
			binn = np.zeros((nbin,), dtype=np.uint32)
			lenpfor = struct.unpack('i', fstream.read(4))[0]
			pforbuf = np.frombuffer(fstream.read(lenpfor), dtype=np.uint32)
			self.pfor.decodeArray(pforbuf, lenpfor//4, binn, nbin)
			binn = np.array(binn, dtype=np.int32)
			bins = np.frombuffer(fstream.read(nbin*4), dtype=np.uint32)
			if xrng is None: tab_nbin = self.c_ebate_merge(tab_nbin, tab_binn, tab_bins, nbin, binn, bins)
			elif inv: tab_nbin = self.c_ebate_filter_inv(tab_nbin, tab_binn, tab_bins, nbin, binn, bins, nmin, nmax)
			else: tab_nbin = self.c_ebate_filter(tab_nbin, tab_binn, tab_bins, nbin, binn, bins, nmin, nmax)
		X = np.fromiter((self.lisq_stepfun(n) for n in tab_binn[0:tab_nbin]), dtype=np.float32)
		return grid_size, xrng, X, tab_binn[0:tab_nbin], tab_bins[0:tab_nbin]

	def query_points(self, xrng, xopt=(1, 1), inv=False, fname=FNAME_INDEX):
		"""EBATE index query, return matching points"""
		fstream = open(fname, "rb")
		ftype, hiter, cube, E0, M = self.read_header(fstream)
		nmin, nmax = self.lisq_translate(xrng, xopt)
		xrng = self.lisq_range(nmin, nmax)
		grid_size = 2**hiter*cube
		if nmin + 1 >= nmax: return grid_size, xrng, [], []
		cube3 = cube**3
		self.lisq_init(E0, M)
		num_cubes = self.hilbert_init(hiter)
		if len(self.hilbmap) != cube3: self.hilbmap = liteqa_hilbmap(cube)

		offs = np.frombuffer(fstream.read(num_cubes*4), dtype=np.uint32)
		xbuf = np.zeros((0,), dtype=np.float32)
		pbuf = np.zeros((0,), dtype=np.uint32)
		ibuf = np.zeros((cube3,), dtype=np.uint32)
		ifor = np.zeros((cube3,), dtype=np.uint32)
		for i in range(num_cubes):
			origin = self.hilbert_cube_origin(cube, i)
			fstream.seek(offs[i], 0)
			nbin = struct.unpack('i', fstream.read(4))[0]
			binn = np.zeros((nbin,), dtype=np.uint32)
			lenpfor = struct.unpack('i', fstream.read(4))[0]
			pforbuf = np.frombuffer(fstream.read(lenpfor), dtype=np.uint32)
			self.pfor.decodeArray(pforbuf, lenpfor//4, binn, nbin)
			binn = np.array(binn, dtype=np.int32)
			bidx = np.zeros((nbin,), dtype=np.uint32)
			if inv: bin_sel = self.c_ebate_select_inv(nbin, binn, bidx, nmin, nmax)
			else: bin_sel = self.c_ebate_select(nbin, binn, bidx, nmin, nmax)
			if bin_sel == 0: continue
			bins = np.frombuffer(fstream.read(nbin*4), dtype=np.uint32)
			boff = np.frombuffer(fstream.read(nbin*4), dtype=np.uint32)

			bin_pos = 0
			for j in bidx[0:bin_sel]:
				# ~ print(bidx[0:bin_sel], j, origin, self.lisq_stepfun(binn[j]), binn[j], bins[j])
				fstream.seek(offs[i] + boff[j])
				lenpfor = struct.unpack('i', fstream.read(4))[0]
				pforbuf = np.frombuffer(fstream.read(lenpfor), dtype=np.uint32)
				self.pfor.decodeArray(pforbuf, lenpfor//4, ifor, bins[j])
				ibuf[bin_pos:bin_pos+bins[j]] = ifor[0:bins[j]]
				bin_pos += bins[j]
				# ~ print(ibuf[0:bins[j]])
			skip_len = xbuf.shape[0]
			xbuf = np.resize(xbuf, (skip_len + bin_pos,))
			pbuf = np.resize(pbuf, ((skip_len + bin_pos)*3,))
			self.c_ebate_points(self.delta, self.omega, cube, origin, bin_sel, bidx, binn, bins, ibuf, skip_len, pbuf, xbuf, self.hilbmap)

		pbuf = pbuf.reshape((pbuf.shape[0]//3, 3), order="C")
		return grid_size, xrng, xbuf, pbuf

	def query_ranges(self, xrngs, xopts, invs, fnames, values=True, points=True):
		"""EBATE index query, AND intersection of multiple ranges, return count or matching data/points"""
		fstreams = []
		delta = np.zeros((len(xrngs),), dtype=np.float64)
		omega = np.zeros((len(xrngs),), dtype=np.float64)
		nrngs = []
		qrngs = []
		run = True
		for i, (xrng, xopt, fname) in enumerate(zip(xrngs, xopts, fnames)):
			fstream = open(fname, "rb")
			ftype, hiter, cube, E0, M = self.read_header(fstream)
			nmin, nmax = self.lisq_translate(xrng, xopt=(1, 1))
			delta[i], omega[i] = self.lisq_init(E0, M)
			fstreams.append(fstream)
			nrngs.append((nmin, nmax))
			qrngs.append(self.lisq_range(nmin, nmax))
			if nmin + 1 >= nmax: run = False
		grid_size = 2**hiter*cube
		if not run:
			if values and points: return grid_size, qrngs, [], []
			elif values or points: return grid_size, qrngs, []
			return grid_size, qrngs, []
		cube3 = cube**3
		num_cubes = self.hilbert_init(hiter)
		if len(self.hilbmap) != cube3: self.hilbmap = liteqa_hilbmap(cube)

		offs = [np.frombuffer(fstream.read(num_cubes*4), dtype=np.uint32) for fstream in fstreams]
		count = 0
		if values: xbuf = np.zeros((0,), dtype=np.float32)
		if points: pbuf = np.zeros((0,), dtype=np.uint32)
		ibuf = np.zeros((cube3,), dtype=np.uint32)
		ifor = np.zeros((cube3,), dtype=np.uint32)
		ptsi = np.zeros((cube3,), dtype=np.uint32)
		ptsn = np.zeros((cube3*len(xrngs),), dtype=np.int32)
		for i in range(num_cubes):
			origin = self.hilbert_cube_origin(cube, i)
			npts = 0
			for f, (fstream, (nmin, nmax), inv) in enumerate(zip(fstreams, nrngs, invs)):
				fstream.seek(offs[f][i], 0)
				nbin = struct.unpack('i', fstream.read(4))[0]
				binn = np.zeros((nbin,), dtype=np.uint32)
				lenpfor = struct.unpack('i', fstream.read(4))[0]
				pforbuf = np.frombuffer(fstream.read(lenpfor), dtype=np.uint32)
				self.pfor.decodeArray(pforbuf, lenpfor//4, binn, nbin)
				binn = np.array(binn, dtype=np.int32)
				bidx = np.zeros((nbin,), dtype=np.uint32)
				if inv: bin_sel = self.c_ebate_select_inv(nbin, binn, bidx, nmin, nmax)
				else: bin_sel = self.c_ebate_select(nbin, binn, bidx, nmin, nmax)
				if bin_sel == 0:
					pts_binn = 0
					break
				bins = np.frombuffer(fstream.read(nbin*4), dtype=np.uint32)
				boff = np.frombuffer(fstream.read(nbin*4), dtype=np.uint32)
				bin_pos = 0
				for j in bidx[0:bin_sel]:
					fstream.seek(offs[f][i] + boff[j])
					lenpfor = struct.unpack('i', fstream.read(4))[0]
					pforbuf = np.frombuffer(fstream.read(lenpfor), dtype=np.uint32)
					self.pfor.decodeArray(pforbuf, lenpfor//4, ifor, bins[j])
					ibuf[bin_pos:bin_pos+bins[j]] = ifor[0:bins[j]]
					bin_pos += bins[j]
				npts = self.c_ebate_intersect(npts, ptsi, f, len(xrngs), ptsn, bin_sel, bin_pos, bidx, binn, bins, ibuf)
			if values:
				skip_len = xbuf.shape[0]//3
				xbuf = np.resize(xbuf, ((skip_len + npts)*len(xrngs),))
				self.c_ebate_intdata(delta, omega, npts, len(xrngs), ptsn, skip_len, xbuf)
			if points:
				skip_len = pbuf.shape[0]//3
				pbuf = np.resize(pbuf, ((skip_len + npts)*3,))
				self.c_ebate_intpoints(cube, origin, npts, ptsi, skip_len, pbuf, self.hilbmap)
			count += npts

		if values: xbuf = xbuf.reshape((xbuf.shape[0]//len(xrngs), len(xrngs)), order="C")
		if points: pbuf = pbuf.reshape((pbuf.shape[0]//3, 3), order="C")

		if values and points: return grid_size, qrngs, xbuf, pbuf
		elif values: return grid_size, qrngs, xbuf
		elif points: return grid_size, qrngs, pbuf
		return grid_size, qrngs, count
