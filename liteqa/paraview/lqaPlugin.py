from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain, smhint
import vtk
import vtkmodules.vtkImagingCore
import vtk.util.numpy_support as vtknp
import numpy as np

import os
import liteqa

########################################################################

class lqaBase(VTKPythonAlgorithmBase):
	def __init__(self, nInputPorts, nOutputPorts, outputType):
		super().__init__(
			nInputPorts=nInputPorts,
			nOutputPorts=nOutputPorts,
			outputType=outputType
		)
		self.LiteQA = liteqa.liteqa()
		self.Query = None
		self.Data = None

	def SetQuery(self, value):
		self.Query = value
		self.Modified()

	def SetData(self, value):
		self.Data = value
		self.Modified()

	def RequestInformation(self, request, inInfo, outInfo):
		return 1	#< success

	def RequestData(self, request, inInfo, outInfo):
		return 1	#< success

########################################################################

class lqaBaseIndex(lqaBase):
	def __init__(self, nInputPorts, nOutputPorts, outputType):
		super().__init__(nInputPorts=nInputPorts, nOutputPorts=nOutputPorts, outputType=outputType)
		self.Array = None
		self.Files = None
		self.ApplyRange = None
		self.Range = None
		self.Invert = None
		self.IncludeLo = None
		self.IncludeHi = None

	def GetQuery(self, time=0):
		return liteqa.parse_query(self.Data, self.Query, index=True, liteqa=self.LiteQA)

	def RequestInformation(self, request, inInfo, outInfo):
		query = self.GetQuery()
		if len(query) > 0:
			alias, fnames, xrng, xopt, nots = query[0]
			self.Array = alias
			self.Files = fnames
			self.Range = xrng
			self.Invert = 1 if nots else 0
			self.IncludeLo = 1 if xopt[0] else 0
			self.IncludeHi = 1 if xopt[1] else 0
			if None in self.Files: self.Files = None
		else: self.Files = None
		return super().RequestInformation(request, inInfo, outInfo)

	def RequestData(self, request, inInfo, outInfo):
		return super().RequestData(request, inInfo, outInfo)

########################################################################

@smproxy.source(label="lqaTable")	#< name of filter
@smhint.xml("<Visibility replace_input='0' />")	#< do not effect visibility of inputs
@smproperty.stringvector(command="SetQuery", name="Query", default_values="", number_of_elements=1)
@smproperty.stringvector(command="SetData", name="Data", default_values="", number_of_elements=1)
class lqaTable(lqaBaseIndex):
	def __init__(self):
		super().__init__(nInputPorts=0, nOutputPorts=1, outputType='vtkTable')

	def RequestData(self, request, inInfo, outInfo):
		if self.Files is None: return 1	#< quiet_failure
		outData = self.GetOutputData(outInfo, 0)	#< preallocated data in output port
		if not self.ApplyRange: res = self.LiteQA.get_index_table(self.Files)
		else: res = self.LiteQA.get_index_table(self.Files, xrng=self.Range, xopt=(self.IncludeLo, self.IncludeHi), inv=self.Invert)
		if res is None: return 1	#< failure
		size, xrng, data, binn, freq = res
		data = vtknp.numpy_to_vtk(data, deep=True)
		binn = vtknp.numpy_to_vtk(binn, deep=True)
		freq = vtknp.numpy_to_vtk(freq, deep=True)
		data.SetName("bin")
		binn.SetName("id")
		freq.SetName("count")
		outData.AddColumn(data)
		outData.AddColumn(binn)
		outData.AddColumn(freq)
		return super().RequestData(request, inInfo, outInfo)

########################################################################

@smproxy.source(label="lqaHistogram")	#< name of filter
@smhint.xml("<Visibility replace_input='0' />")	#< do not effect visibility of inputs
@smproperty.intvector(command="SetMaxBins", name="MaxBins", default_values=[256], number_of_elements=1)
@smproperty.stringvector(command="SetQuery", name="Query", default_values="", number_of_elements=1)
@smproperty.stringvector(command="SetData", name="Data", default_values="", number_of_elements=1)
class lqaHistogram(lqaBaseIndex):
	def __init__(self):
		super().__init__(nInputPorts=0, nOutputPorts=1, outputType='vtkTable')
		self.MaxBins = None

	def SetMaxBins(self, value):
		self.MaxBins = value
		self.Modified()

	def RequestData(self, request, inInfo, outInfo):
		if self.Files is None: return 1	#< quiet_failure
		outData = self.GetOutputData(outInfo, 0)	#< preallocated data in output port
		if not self.ApplyRange: res = self.LiteQA.get_index_table(self.Files)
		else: res = self.LiteQA.get_index_table(self.Files, xrng=self.Range, xopt=(self.IncludeLo, self.IncludeHi), inv=self.Invert)
		if res is None: return 1	#< failure
		size, xrng, data, binn, freq = res
		num_bins = int((data[-1] - data[0])/(2.125*np.max(np.fabs(data))*self.LiteQA.omega))
		freq, data = np.histogram(data, weights=freq, bins=min(num_bins, self.MaxBins))
		data = ((data + np.roll(data, shift=1))*0.5)[1:]
		data = vtknp.numpy_to_vtk(data, deep=True)
		freq = vtknp.numpy_to_vtk(freq, deep=True)
		data.SetName("bin")
		freq.SetName("count")
		outData.AddColumn(data)
		outData.AddColumn(freq)
		return super().RequestData(request, inInfo, outInfo)

########################################################################

@smproxy.source(label="lqaPoints")	#< name of filter
@smhint.xml("<Visibility replace_input='0' />")	#< do not effect visibility of inputs
@smproperty.stringvector(command="SetQuery", name="Query", default_values="", number_of_elements=1)
@smproperty.stringvector(command="SetData", name="Data", default_values="", number_of_elements=1)
class lqaPoints(lqaBaseIndex):
	def __init__(self):
		super().__init__(nInputPorts=0, nOutputPorts=1, outputType='vtkPolyData')

	def RequestData(self, request, inInfo, outInfo):
		if self.Files is None: return 1	#< quiet_failure
		outData = self.GetOutputData(outInfo, 0)	#< preallocated data in output port
		res = self.LiteQA.get_index_points(self.Files, xrng=self.Range, xopt=(self.IncludeLo, self.IncludeHi), inv=self.Invert)
		if res is None: return 1	#< failure
		size, xrng, data, points = res
		data = vtknp.numpy_to_vtk(data, deep=True)
		points = vtknp.numpy_to_vtk(points, deep=True)
		data.SetName(self.Array)
		points.SetName("Points")
		vpts = vtk.vtkPoints()
		vpts.SetData(points)
		outData.SetPoints(vpts)
		outData.GetPointData().AddArray(data)
		outData.GetPointData().AddArray(points)

		mask = vtk.vtkMaskPoints()
		mask.SetInputDataObject(0, outData)
		mask.SetOnRatio(1)
		mask.SetMaximumNumberOfPoints(outData.GetNumberOfPoints())
		mask.SingleVertexPerCellOn()
		mask.GenerateVerticesOn()
		mask.SetOutputPointsPrecision(vtk.vtkAlgorithm.SINGLE_PRECISION)
		mask.RandomModeOff()
		mask.Update()

		outData.ShallowCopy(mask.GetOutputDataObject(0))
		del mask
		return super().RequestData(request, inInfo, outInfo)

########################################################################

@smproxy.source(label="lqaQuery")	#< name of filter
@smhint.xml("<Visibility replace_input='0' />")	#< do not effect visibility of inputs
@smproperty.stringvector(command="SetQuery", name="Query", default_values="", number_of_elements=1)
@smproperty.stringvector(command="SetData", name="Data", default_values="", number_of_elements=1)
class lqaQuery(lqaBase):
	def __init__(self):
		super().__init__(nInputPorts=0, nOutputPorts=1, outputType='vtkPolyData')
		self.Array = None
		self.Files = None
		self.Range = None
		self.Optis = None
		self.Invts = None

	def GetQuery(self, time=0):
		return liteqa.parse_query(self.Data, self.Query, index=True, liteqa=self.LiteQA)

	def RequestInformation(self, request, inInfo, outInfo):
		self.Array = []
		self.Files = []
		self.Range = []
		self.Optis = []
		self.Invts = []
		for alias, fnames, xrng, xopt, nots in self.GetQuery():
			self.Array.append(alias)
			self.Files.append(fnames)
			self.Range.append(xrng)
			self.Optis.append(xopt)
			self.Invts.append(nots)
		if None in self.Files: self.Files = None
		return super().RequestInformation(request, inInfo, outInfo)

	def RequestData(self, request, inInfo, outInfo):
		if self.Files is None: return 1	#< quiet_failure
		outData = self.GetOutputData(outInfo, 0)	#< preallocated data in output port
		res = self.LiteQA.get_index_ranges(self.Files, xrngs=self.Range, xopts=self.Optis, invs=self.Invts)
		if res is None: return 1	#< failure
		size, xrngs, data, points = res
		if len(points) > 0:
			points = vtknp.numpy_to_vtk(points, deep=True)
			points.SetName("Points")
			vpts = vtk.vtkPoints()
			vpts.SetData(points)
			outData.SetPoints(vpts)
			outData.GetPointData().AddArray(points)
			# ~ data = vtknp.numpy_to_vtk(data, deep=True)
			# ~ data.SetName(self.Array)
			# ~ outData.GetPointData().AddArray(data)

			mask = vtk.vtkMaskPoints()
			mask.SetInputDataObject(0, outData)
			mask.SetOnRatio(1)
			mask.SetMaximumNumberOfPoints(outData.GetNumberOfPoints())
			mask.SingleVertexPerCellOn()
			mask.GenerateVerticesOn()
			mask.SetOutputPointsPrecision(vtk.vtkAlgorithm.SINGLE_PRECISION)
			mask.RandomModeOff()
			mask.Update()

			outData.ShallowCopy(mask.GetOutputDataObject(0))
			del mask
		return super().RequestData(request, inInfo, outInfo)

########################################################################

class lqaBaseGrid(lqaBase):
	def __init__(self, nInputPorts, nOutputPorts, outputType):
		super().__init__(nInputPorts=nInputPorts, nOutputPorts=nOutputPorts, outputType=outputType)
		self.Array = None
		self.Files = None
		self.Extent = None

	def SetExtent(self, a, b, c, d, e, f):
		self.Extent = [a, b, c, d, e, f]
		self.Modified()

	def GetQuery(self, time=0):
		return liteqa.parse_query(self.Data, self.Query, index=False, liteqa=self.LiteQA)

	def UpdateFiles(self):
		query = self.GetQuery()
		if len(query) > 0:
			self.Array = []
			self.Files = []
			for alias, fnames, xrng, xopt, nots in self.GetQuery():
				if None in fnames: self.Files = None; break
				self.Array.append(alias)
				self.Files.append(fnames)
		else: self.Files = None

	def RequestInformation(self, request, inInfo, outInfo):
		self.UpdateFiles()
		if self.Files is None: extent = [0, -1, 0, -1, 0, -1]
		else: extent = self.LiteQA.get_grid_extent(self.Files[0])
		outInfo.GetInformationObject(0).Set(self.GetExecutive().WHOLE_EXTENT(), *extent)
		return super().RequestInformation(request, inInfo, outInfo)

	def RequestUpdateExtent(self, request, inInfo, outInfo):
		extent = outInfo.GetInformationObject(0).Get(self.GetExecutive().WHOLE_EXTENT())
		if extent is None: return 0	#< failure
		outInfo.GetInformationObject(0).Set(self.GetExecutive().UPDATE_EXTENT(), *extent)
		return 1	#< success

	def RequestData(self, request, inInfo, outInfo):
		return super().RequestData(request, inInfo, outInfo)

########################################################################

@smproxy.source(label="lqaGrid")	#< name of filter
@smhint.xml("<Visibility replace_input='0' />")	#< do not effect visibility of inputs
@smproperty.stringvector(command="SetQuery", name="Query", default_values="", number_of_elements=1)
@smproperty.stringvector(command="SetData", name="Data", default_values="", number_of_elements=1)
class lqaGrid(lqaBaseGrid):
	def __init__(self):
		super().__init__(nInputPorts=0, nOutputPorts=1, outputType='vtkImageData')

	def RequestData(self, request, inInfo, outInfo):
		if self.Files is None: return 1	#< quiet_failure
		outData = self.GetOutputData(outInfo, 0)	#< preallocated data in output port
		extent = outInfo.GetInformationObject(0).Get(self.GetExecutive().WHOLE_EXTENT())
		if extent is None: return 0	#< failure
		outData.SetExtent(extent)
		for array, fnames in zip(self.Array, self.Files):
			data = self.LiteQA.get_grid_data(fnames)
			if not data is None:
				if len(data.shape) == 3: data = data.flatten(order="F")
				elif len(data.shape) == 4: data = data.reshape((-1, 3), order="F")
				else: return 0	#< failure
				data = vtknp.numpy_to_vtk(data, deep=True)
				data.SetName(array)
				outData.GetPointData().AddArray(data)
		return super().RequestData(request, inInfo, outInfo)

########################################################################

@smproxy.filter(label="lqaRegions")
@smhint.xml("<Visibility replace_input='0' />")	#< do not effect visibility of inputs
@smproperty.input(name="Input", port_index=0)	#< pipeline input 0
@smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=False)	#< input data type
@smproperty.intvector(command="SetExtent", name="Extent", default_values=[0, 0, 0, 0, 0, 0], number_of_elements=6)
@smproperty.stringvector(command="SetQuery", name="Query", default_values="", number_of_elements=1)
@smproperty.stringvector(command="SetData", name="Data", default_values="", number_of_elements=1)
class lqaRegions(lqaBaseGrid):
	def __init__(self):
		super().__init__(nInputPorts=1, nOutputPorts=1, outputType='vtkImageData')

	def RequestData(self, request, inInfo, outInfo):
		if self.Files is None: return 1	#< quiet_failure
		inData = self.GetInputData(inInfo, 0, 0)	#< data from input port 0
		outData = self.GetOutputData(outInfo, 0)	#< preallocated data in output port
		extent = outInfo.GetInformationObject(0).Get(self.GetExecutive().WHOLE_EXTENT())
		outData.SetExtent(extent)
		if inData.GetNumberOfPoints() > 0:
			points = np.array(vtknp.vtk_to_numpy(inData.GetPoints().GetData()), dtype=np.int32)
			cube_list = self.LiteQA.get_grid_cubes(self.Files[0], points, extent=self.Extent)
		else: cube_list = []
		for array, fnames in zip(self.Array, self.Files):
			data = self.LiteQA.get_grid_data(fnames, cube_list=cube_list)
			if not data is None:
				if len(data.shape) == 3: data = data.flatten(order="F")
				elif len(data.shape) == 4: data = data.reshape((outData.GetNumberOfPoints(), 3), order="F")
				else: return 0	#< failure
				data = vtknp.numpy_to_vtk(data, deep=True)
				data.SetName(array)
				outData.GetPointData().AddArray(data)
		return super().RequestData(request, inInfo, outInfo)

########################################################################

@smproxy.filter(label="lqaBlock")
@smhint.xml("<Visibility replace_input='0' />")	#< do not effect visibility of inputs
@smproperty.input(name="Input")	#< pipeline input 0
@smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=False)	#< input data type
@smproperty.intvector(command="SetExtent", name="Extent", default_values=[-1, 1, -1, 1, -1, 1], number_of_elements=6)
@smproperty.stringvector(command="SetQuery", name="Query", default_values="", number_of_elements=1)
@smproperty.stringvector(command="SetData", name="Data", default_values="", number_of_elements=1)
class lqaBlock(lqaBaseGrid):
	def __init__(self):
		super().__init__(nInputPorts=1, nOutputPorts=1, outputType='vtkImageData')

	def FillInputPortInformation(self, port, info):
		if port == 0: info.Set(self.INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet")
		return 1	#< success

	def RequestInformation(self, request, inInfo, outInfo):
		self.UpdateFiles()
		if self.Files is None: extent = [0, -1, 0, -1, 0, -1]
		else: extent = self.LiteQA.get_grid_extent(self.Files[0], extent=self.Extent)
		outInfo.GetInformationObject(0).Set(self.GetExecutive().WHOLE_EXTENT(), *extent)
		return lqaBase.RequestInformation(self, request, inInfo, outInfo)

	def RequestData(self, request, inInfo, outInfo):
		if self.Files is None: return 1	#< quiet_failure
		inData = vtk.vtkDataSet.GetData(inInfo[0], 0)	#< data from input port 0
		outData = vtk.vtkImageData.GetData(outInfo, 0)	#< preallocated data in output port
		extent = outInfo.GetInformationObject(0).Get(self.GetExecutive().UPDATE_EXTENT())
		outData.SetExtent(extent)
		if inData.GetNumberOfPoints() > 0:
			bounds = inData.GetBounds()
			center = ((bounds[0] + bounds[1])*0.5, (bounds[2] + bounds[3])*0.5, (bounds[4] + bounds[5])*0.5)
			blockext = self.LiteQA.get_grid_extent(self.Files[0], extent=self.Extent, center=center)
			if not blockext is None: extent = blockext
		outData.SetOrigin(extent[0], extent[2], extent[4])
		for array, fnames in zip(self.Array, self.Files):
			data = self.LiteQA.get_grid_data(fnames, extent=extent)
			if not data is None:
				if len(data.shape) == 3: data = data.flatten(order="F")
				elif len(data.shape) == 4: data = data.reshape((-1, 3), order="F")
				else: return 0	#< failure
				data = vtknp.numpy_to_vtk(data, deep=True)
				data.SetName(array)
				outData.GetPointData().AddArray(data)
		return super().RequestData(request, inInfo, outInfo)

########################################################################

@smproxy.filter(label="lqaEuclideanClustering")	#< name of filter
@smhint.xml("<Visibility replace_input='0' />")	#< do not effect visibility of inputs
@smproperty.input(name="Input", port_index=0)	#< pipeline input 0
@smdomain.datatype(dataTypes=["vtkPolyData"], composite_data_supported=False)	#< input data type
class lqaEuclideanClustering(VTKPythonAlgorithmBase):
	def __init__(self):
		super().__init__(nInputPorts=1, nOutputPorts=1, outputType='vtkPolyData')
		self.Radius = None
		self.MinSize = None
		self.ClusterSize = None

	@smproperty.doublevector(name="Radius", default_values=2)
	def SetRadius(self, value):
		self.Radius = value
		self.Modified()

	@smproperty.intvector(name="MinSize", default_values=0)
	def SetMinSize(self, value):
		self.MinSize = value
		self.Modified()

	@smproperty.intvector(name="NumClusters", default_values=0, number_of_elements=1, information_only=1)
	def GetNumClusters(self):
		if self.ClusterSize is None: return int(-1)
		if self.MinSize == 0: return int(self.ClusterSize.shape[0])
		else: return int(np.sum(self.ClusterSize >= self.MinSize))

	def RequestData(self, request, inInfo, outInfo):
		import vtkmodules.vtkFiltersPoints
		inData = self.GetInputData(inInfo, 0, 0)	#< data from input port 0
		outData = vtk.vtkPolyData.GetData(outInfo, 0)	#< preallocated data in output port

		clust = vtkmodules.vtkFiltersPoints.vtkEuclideanClusterExtraction()
		clust.SetInputDataObject(0, inData)
		clust.ColorClustersOn()
		clust.SetExtractionModeToAllClusters()
		clust.SetRadius(self.Radius)
		clust.Update()

		clout = clust.GetOutputDataObject(0)
		if clout.GetNumberOfPoints() > 0:
			clpts = clout.GetPoints().GetData()
			clids = clout.GetPointData().GetArray("ClusterId")
			clpts = vtknp.vtk_to_numpy(clpts)
			clids = vtknp.vtk_to_numpy(clids)
			val, idx, cnt = np.unique(clids, return_inverse=True, return_counts=True)
			cnt = np.array(cnt, dtype=np.uint32)
			rank = np.zeros((len(cnt),), dtype=np.uint32)
			order = [(c, i) for i, c in enumerate(cnt)]
			self.ClusterSize = np.zeros(cnt.shape, dtype=np.uint32)
			for i, (c, j) in enumerate(sorted(order)):
				rank[j] = len(cnt) - i - 1
				self.ClusterSize[i] = c

			mask = vtk.vtkMaskPoints()
			mask.SetOnRatio(1)
			mask.SingleVertexPerCellOn()
			mask.GenerateVerticesOn()
			mask.SetOutputPointsPrecision(vtk.vtkAlgorithm.SINGLE_PRECISION)
			mask.RandomModeOff()

			vtk_rank = vtknp.numpy_to_vtk(rank[idx], deep=True)
			vtk_size = vtknp.numpy_to_vtk(cnt[idx], deep=True)
			vtk_rank.SetName("ClusterRank")
			vtk_size.SetName("ClusterSize")
			poly = clust.GetOutputDataObject(0)
			poly.GetPointData().AddArray(vtk_rank)
			poly.GetPointData().AddArray(vtk_size)
			poly.GetPointData().RemoveArray("ClusterId")
			mask.SetInputDataObject(0, poly)
			mask.SetMaximumNumberOfPoints(poly.GetNumberOfPoints())
			mask.Update()

			outData.ShallowCopy(mask.GetOutputDataObject(0))
			del mask
			del clust
		return 1	#< success

########################################################################

@smproxy.filter(label="lqaClusterCenter")	#< name of filter
@smhint.xml("<Visibility replace_input='0' />")	#< do not effect visibility of inputs
@smproperty.input(name="Input", port_index=0)	#< pipeline input 0
@smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=False)	#< input data type
class lqaClusterCenter(VTKPythonAlgorithmBase):
	def __init__(self):
		super().__init__(nInputPorts=1, nOutputPorts=1, outputType='vtkPolyData')

	def RequestData(self, request, inInfo, outInfo):
		import vtkmodules.vtkFiltersPoints
		inData = self.GetInputData(inInfo, 0, 0)	#< data from input port 0
		outData = vtk.vtkPolyData.GetData(outInfo, 0)	#< preallocated data in output port

		if inData.GetNumberOfPoints() > 0:
			clpts = inData.GetPoints().GetData()
			clids = inData.GetPointData().GetArray("ClusterRank")
			clpts = vtknp.vtk_to_numpy(clpts)
			clids = vtknp.vtk_to_numpy(clids)
			rank = np.unique(clids)

			C = np.zeros((len(rank),3), dtype=np.float32)
			S = np.zeros((len(rank),), dtype=np.uint32)
			for i, r in enumerate(rank):
				P = clpts[clids == r]
				C[i, :] = np.sum(P, axis=0)/len(P)
				S[i] = P.shape[0]
			vtk_rank = vtknp.numpy_to_vtk(np.arange(len(rank), dtype=np.uint32), deep=True)
			vtk_size = vtknp.numpy_to_vtk(S, deep=True)
			vtk_pts = vtknp.numpy_to_vtk(C, deep=True)
			vtk_rank.SetName("ClusterRank")
			vtk_size.SetName("ClusterSize")
			vtk_pts.SetName("Points")
			points = vtk.vtkPoints()
			points.SetData(vtk_pts)
			outData.SetPoints(points)
			outData.GetPointData().AddArray(vtk_rank)
			outData.GetPointData().AddArray(vtk_size)

			mask = vtk.vtkMaskPoints()
			mask.SetOnRatio(1)
			mask.SingleVertexPerCellOn()
			mask.GenerateVerticesOn()
			mask.SetOutputPointsPrecision(vtk.vtkAlgorithm.SINGLE_PRECISION)
			mask.RandomModeOff()
			mask.SetInputDataObject(0, outData)
			mask.SetMaximumNumberOfPoints(outData.GetNumberOfPoints())
			mask.Update()
			outData.ShallowCopy(mask.GetOutputDataObject(0))
			del mask
		return 1	#< success

########################################################################

@smproxy.filter(label="lqaClusterSelect")	#< name of filter
@smhint.xml("<Visibility replace_input='1' />")	#< do not effect visibility of inputs
@smproperty.input(name="Input", port_index=0)	#< pipeline input 0
@smdomain.datatype(dataTypes=["vtkPolyData"], composite_data_supported=False)	#< input data type
class lqaClusterSelect(VTKPythonAlgorithmBase):
	def __init__(self):
		super().__init__(nInputPorts=1, nOutputPorts=1, outputType='vtkPolyData')
		self.Rank = None

	@smproperty.intvector(name="Rank", default_values=-1)
	def SetRank(self, value):
		self.Rank = value
		self.Modified()

	def RequestData(self, request, inInfo, outInfo):
		import vtkmodules.vtkFiltersPoints
		inData = self.GetInputData(inInfo, 0, 0)	#< data from input port 0
		outData = vtk.vtkPolyData.GetData(outInfo, 0)	#< preallocated data in output port

		thres = vtk.vtkThresholdPoints()
		thres.SetInputDataObject(0, inData)
		thres.SetInputArrayToProcess(0, 0, 0, "vtkDataObject::FIELD_ASSOCIATION_POINTS", "ClusterRank")
		thres.ThresholdBetween(self.Rank, self.Rank)
		thres.Update()
		poly = thres.GetOutputDataObject(0)
		outData.ShallowCopy(poly)
		del thres
		return 1	#< success
