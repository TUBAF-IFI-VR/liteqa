#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import liteqa
import argparse
import numpy as np
import vtk
import vtk.util.numpy_support as vtknp

########################################################################

def read_vti(fname, extent=None):
	import vtk
	fname, array = tuple((fname + ":").split(":")[0:2])
	r = vtk.vtkXMLImageDataReader()
	if not r.CanReadFile(fname): raise ValueError("file not found:", fname)
	r.SetFileName(fname)
	if len(array) >  0:
		r.GetPointDataArraySelection().DisableAllArrays()
		r.SetPointArrayStatus(array, 1)
	if extent is None:
		r.Update()
		img = r.GetOutputDataObject(0)
	else:
		e = vtk.vtkExtractVOI()
		e.SetInputConnection(r.GetOutputPort())
		e.SetVOI(*extent)
		e.Update()
		img = e.GetOutputDataObject(0)
	data = vtknp.vtk_to_numpy(img.GetPointData().GetArray(0 if len(array) == 0 else array))
	ni, nj, nk = img.GetDimensions()
	if len(data.shape) == 1:	# scalar
		data = data.reshape(img.GetDimensions(), order='F')	# 'F' FORTRAN order in vtk
	elif len(data.shape) == 2:	# vector
		data = data.reshape((*img.GetDimensions(), 3) , order='F')	# 'F' FORTRAN order in vtk
	else: raise ValueError("components not supported", array.shape)
	return data

def wrap_fwd(img, WrapPad):
	extent = img.GetExtent()
	wrap = vtk.vtkImageWrapPad()
	wrap.SetInputDataObject(0, img)
	wrap.SetOutputWholeExtent(
		extent[0] - WrapPad, extent[1] + WrapPad,
		extent[2] - WrapPad, extent[3] + WrapPad,
		extent[4] - WrapPad, extent[5] + WrapPad
	)
	wrap.Update()
	img.ShallowCopy(wrap.GetOutputDataObject(0))
	img.SetOrigin(0, 0, 0)
	img.SetSpacing(1, 1, 1)
	img.SetExtent(
		0, extent[1] - extent[0] + 2*WrapPad,
		0, extent[3] - extent[2] + 2*WrapPad,
		0, extent[5] - extent[4] + 2*WrapPad
	)
	del wrap

def wrap_bwd(img, WrapPad):
	extent = img.GetExtent()
	wrap = vtk.vtkImageWrapPad()
	wrap.SetInputDataObject(0, img)
	wrap.SetOutputWholeExtent(
		extent[0] + WrapPad, extent[1] - WrapPad,
		extent[2] + WrapPad, extent[3] - WrapPad,
		extent[4] + WrapPad, extent[5] - WrapPad
	)
	wrap.Update()
	img.ShallowCopy(wrap.GetOutputDataObject(0))
	img.SetOrigin(0, 0, 0)
	img.SetSpacing(1, 1, 1)
	img.SetExtent(
		0, extent[1] - extent[0] - 2*WrapPad,
		0, extent[3] - extent[2] - 2*WrapPad,
		0, extent[5] - extent[4] - 2*WrapPad
	)
	del wrap

def wrap_trafo(inpu, outp, array, WrapPad=0):
	shape = array.shape
	img = vtk.vtkImageData()
	img.SetDimensions(shape[0], shape[1], shape[2])
	if len(array.shape) == 3:	# scalar
		array = array.flatten(order="F")
	elif len(array.shape) == 4:	# vector
		array = array.reshape((shape[0]*shape[1]*shape[2], 3), order="F")
	array = vtknp.numpy_to_vtk(array, deep=True)
	array.SetName("ImageScalars")
	img.GetPointData().AddArray(array)
	img.GetPointData().SetActiveScalars("ImageScalars")
	if WrapPad < 1: WrapPad = shape[0]*WrapPad
	if WrapPad >= 1: wrap_fwd(img, int(WrapPad))
	inpu.SetInputDataObject(0, img)
	outp.Update()
	img.ShallowCopy(outp.GetOutputDataObject(0))
	if WrapPad >= 1: wrap_bwd(img, int(WrapPad))
	if not img.GetPointData().HasArray("ImageScalars"): array = vtknp.vtk_to_numpy(img.GetPointData().GetArray(0))
	else: array = vtknp.vtk_to_numpy(img.GetPointData().GetArray("ImageScalars"))
	if len(array.shape) == 1:	# scalar
		array = array.reshape((shape[0], shape[1], shape[2]) , order="F")
	elif len(array.shape) == 2:	# vector
		array = array.reshape((shape[0], shape[1], shape[2], 3), order="F")
	del img
	return array

########################################################################

def trafo_grad(array, wrappad=0):
	print("# compute_grad")
	g = vtk.vtkGradientFilter()
	g.ComputeDivergenceOff()
	g.ComputeGradientOn()
	g.ComputeQCriterionOff()
	g.SetResultArrayName("ImageScalars")
	return wrap_trafo(g, g, array, wrappad)

def trafo_qcrit(array, wrappad=0):
	print("# compute_qcrit")
	g = vtk.vtkGradientFilter()
	g.ComputeDivergenceOff()
	g.ComputeGradientOff()
	g.ComputeQCriterionOn()
	g.SetQCriterionArrayName("ImageScalars")
	return wrap_trafo(g, g, array, wrappad)

def trafo_vort(array, wrappad=0):
	print("# compute_vort")
	g = vtk.vtkGradientFilter()
	g.ComputeDivergenceOff()
	g.ComputeGradientOff()
	g.ComputeVorticityOn()
	g.SetVorticityArrayName("ImageScalars")
	return wrap_trafo(g, g, array, wrappad)

def trafo_dist(array, wrappad=0):
	print("# compute_dist")
	MaxDistance = 999999
	d = vtk.vtkImageEuclideanDistance()
	d.InitializeOn()
	d.SetMaximumDistance(MaxDistance)
	d.SetAlgorithmToSaito()
	#d.SetAlgorithmToSaitoCached()
	#d.SetConsiderAnisotropy(FlagAnisotropy)
	dist1 = wrap_trafo(d, d, 1 - array, wrappad)
	dist2 = (wrap_trafo(d, d, array, wrappad) - 1)*array
	return np.sqrt(dist1) - np.sqrt(dist2)

########################################################################

def check_error(X, Y, delta, omega, mask=None):
	if not mask is None:
		print("# skip_val_mask", np.sum(mask))
		X = X[mask == 0]
		Y = Y[mask == 0]
	print("# num_val_lt_delta", np.sum(np.fabs(X) < delta))
	idx = np.fabs(X) >= lqa.delta
	print("# num_val_geq_delta", np.sum(idx))
	X = X[idx]
	Y = Y[idx]
	RE = np.nan_to_num(np.fabs(X - Y)/np.fabs(X))
	idx = RE >= lqa.omega*1.05
	# ~ for x, y, re in zip(X[idx], Y[idx], RE[idx]):
		# ~ print(x, y, re, lqa.omega*1.05)

	print("# num_error_vals", np.sum(idx))

def status(prog=None):
	print("#  ", "%.02f" % prog, "%")

########################################################################

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--check", action="store_true", help="decompression error check")
parser.add_argument("--E0", type=int, default=20, help="lisq_delta = 2**-E0")
parser.add_argument("--M", type=int, default=35, help="lisq_omega = (2**(1/M)-1)/(2**(1/M)+1)")
parser.add_argument("--extent", type=int, default=[0,0,0,0,0,0], nargs=6, help="compression extent")
parser.add_argument("--gcube", type=int, default=32, help="grid cube size")
parser.add_argument("--icube", type=int, default=256, help="index cube size")
parser.add_argument("--maskfile", type=str, default="", help="vti mask data set")
parser.add_argument("--vtifile", type=str, default="data.vti", help="vti data set")
parser.add_argument("--component", type=int, default=-1, help="vti data array component")
parser.add_argument("--lqadir", type=str, default="", help="liteqa data set")
parser.add_argument("--lqaarray", type=str, default="", help="lqa data array name")
parser.add_argument("--zstd", action="store_true", help="compress data as mask with zstd")
parser.add_argument("--index", action="store_true", help="compress data as index with ebate")
parser.add_argument("--grid", action="store_true", help="compress data as grid with glate")
parser.add_argument("--trafo", type=str, default="", help="data trafo: mag qcrit vort dist")
parser.add_argument("--wrappad", type=float, default=0, help="wrap pad N voxels of data before trafo")
parser.add_argument("--zlevel", type=int, default=2, help="zstd level")
args = parser.parse_args()

########################################################################

if not args.index and not args.grid and not args.zstd:
	raise ValueError("no --index/--grid/--zstd compression enabled")

if all(x == args.extent[0] for x in args.extent): args.extent = None
print("# read_vti", args.vtifile)
data = read_vti(args.vtifile, extent=args.extent)

print("# read_mask", args.maskfile)
mask = None
if len(args.maskfile) > 0:
	mask = read_vti(args.maskfile, extent=args.extent)
	mask = np.array(mask != 0, dtype=np.uint8)
	masksum = np.sum(mask)
	print("# mask_blank", "%.02f" % (masksum/np.prod(mask.shape)*100), "%", masksum)

########################################################################

for i in args.trafo.replace("+", " ").split():
	print("# trafo", i)
	if   i == "mag": data = np.sqrt(data[:,:,:,0]**2 + data[:,:,:,1]**2 + data[:,:,:,2]**2)
	elif i == "vort": data = trafo_vort(data, args.wrappad)
	elif i == "qcrit": data = trafo_qcrit(data, args.wrappad)
	elif i == "grad": data = trafo_grad(data, args.wrappad)
	elif i == "dist": data = trafo_dist(data, args.wrappad)
	else:
		raise ValueError("wrong trafo", args.trafo)

########################################################################

grid_size = data[0].shape[0]
print("# grid_size", grid_size, grid_size**3)
if args.icube == -1: args.icube = grid_size//abs(args.icube)
if args.gcube == -1: args.gcube = grid_size//abs(args.gcube)
print("# grid_cube", args.gcube, args.gcube**3)
print("# index_cube", args.icube, args.icube**3)

if len(args.lqadir) == 0: args.lqadir = args.vtifile.split(":")[0][0:-4] + ".lqa"
if len(args.lqaarray) == 0: args.lqaarray = args.vtifile.split(":")[1]
if not os.path.exists(args.lqadir): os.mkdir(args.lqadir)

lqa = liteqa.liteqa(args.E0, args.M)
print("# lisq_E0", args.E0)
print("# lisq_M", args.M)
print("# lisq_delta", lqa.delta)
print("# lisq_omega", "%02f" % (lqa.omega*100), "%")

########################################################################

if len(data.shape) == 3: data = [data]
elif len(data.shape) == 4:
	if args.component > -1: data = [data[:,:,:,args.component]]
	else: data = [data[:,:,:,0], data[:,:,:,1], data[:,:,:,2]]
else: raise ValueError("wrong shape")

if mask is None:
	print("# read_mask", "zero")
	mask = np.zeros_like(data[0], dtype=np.uint8)

########################################################################

if args.zstd:
	for i, X in enumerate(data):
		lqaarray = args.lqaarray + ("" if len(data) == 1 else str(i))
		fname = args.lqadir + "/" + lqaarray + ".g"
		print("# compress_zstd", lqaarray, "->", fname)
		CR = lqa.compress_zstd(X, cube=args.gcube, fname=fname, callb=status, zlevel=args.zlevel)
		print("# compress_zstd", lqaarray, "->", fname, "%.02f" % (CR*100), "%")
		if args.check:
			print("# decompress_grid", fname)
			Y = lqa.decompress(fname=fname)
			check_error(X, Y, lqa.delta, lqa.omega)

########################################################################

if args.index or args.grid:
	data = [np.array(x, dtype=np.float32) for x in data]

	if args.index:
		for i, X in enumerate(data):
			lqaarray = args.lqaarray + ("" if len(data) == 1 else str(i))
			fname = args.lqadir + "/" + lqaarray + (""  if args.component == -1 else str(args.component)) + ".i"
			print("# compress_index", lqaarray, "->", fname)
			CR = lqa.compress_index(X, mask, cube=args.icube, fname=fname, callb=status)
			print("# compress_index", lqaarray, "->", fname, "%.02f" % (CR*100), "%")
			if args.check:
				print("# decompress_index", fname)
				Y = lqa.decompress(fname=fname)
				check_error(X, Y, lqa.delta, lqa.omega, mask)

########################################################################

	if args.grid:
		for i, X in enumerate(data):
			lqaarray = args.lqaarray + ("" if len(data) == 1 else str(i))
			fname = args.lqadir + "/" + lqaarray + (""  if args.component == -1 else str(args.component)) + ".g"
			print("# compress_grid", lqaarray, "->", fname)
			CR = lqa.compress_grid(X, cube=args.gcube, fname=fname, callb=status, zlevel=args.zlevel)
			print("# compress_grid", lqaarray, "->", fname, "%.02f" % (CR*100), "%")
			if args.check:
				print("# decompress_grid", fname)
				Y = lqa.decompress(fname=fname)
				check_error(X, Y, lqa.delta, lqa.omega)

########################################################################

print("# done")
