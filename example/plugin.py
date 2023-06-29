import liteqa

# trace generated using paraview version 5.11.0-RC1
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *

LoadPlugin(liteqa.get_distfile("lqaPlugin.py"), ns=globals())

########################################################################

# create new layout object
layout3 = CreateLayout(name='lqaTable')
# create a new 'lqaTable'
lqaTable1 = lqaTable(registrationName='lqaTable1')
# Properties modified on lqaTable1
lqaTable1.Data = 'data1.lqa'
lqaTable1.Query = 'vmag'
# Create a new 'SpreadSheet View'
spreadSheetView2 = CreateView('SpreadSheetView')
spreadSheetView2.ColumnToSort = ''
spreadSheetView2.BlockSize = 1024
# show data in view
lqaTable1Display = Show(lqaTable1, spreadSheetView2, 'SpreadSheetRepresentation')
# add view to a layout so it's visible in UI
AssignViewToLayout(view=spreadSheetView2, layout=layout3, hint=0)
# Properties modified on lqaTable1Display
lqaTable1Display.Assembly = ''
# update the view to ensure updated data information
spreadSheetView2.Update()
# split cell
layout3.SplitHorizontal(0, 0.5)
# Create a new 'Bar Chart View'
barChartView2 = CreateView('XYBarChartView')
# assign view to a particular cell in the layout
AssignViewToLayout(view=barChartView2, layout=layout3, hint=2)
# show data in view
lqaTable1Display_1 = Show(lqaTable1, barChartView2, 'XYBarChartRepresentation')
# trace defaults for the display properties.
lqaTable1Display_1.AttributeType = 'Row Data'
lqaTable1Display_1.XArrayName = 'bin'
lqaTable1Display_1.SeriesVisibility = ['bin', 'count', 'id']
lqaTable1Display_1.SeriesLabel = ['bin', 'bin', 'count', 'count', 'id', 'id']
lqaTable1Display_1.SeriesColor = ['bin', '0', '0', '0', 'count', '0.8899977111467154', '0.10000762951094835', '0.1100022888532845', 'id', '0.220004577706569', '0.4899977111467155', '0.7199969481956207']
lqaTable1Display_1.SeriesOpacity = ['bin', '1.0', 'count', '1.0', 'id', '1.0']
lqaTable1Display_1.SeriesPlotCorner = ['bin', '0', 'count', '0', 'id', '0']
lqaTable1Display_1.SeriesLabelPrefix = ''

########################################################################

# create new layout object
layout4 = CreateLayout(name='lqaHistogram')
# create a new 'lqaHistogram'
lqaHistogram1 = lqaHistogram(registrationName='lqaHistogram1')
# Properties modified on lqaHistogram1
lqaHistogram1.Data = 'data1.lqa'
lqaHistogram1.Query = 'vmag'
# Create a new 'SpreadSheet View'
spreadSheetView3 = CreateView('SpreadSheetView')
spreadSheetView3.ColumnToSort = ''
spreadSheetView3.BlockSize = 1024
# show data in view
lqaHistogram1Display = Show(lqaHistogram1, spreadSheetView3, 'SpreadSheetRepresentation')
# add view to a layout so it's visible in UI
AssignViewToLayout(view=spreadSheetView3, layout=layout4, hint=0)
# Properties modified on lqaHistogram1Display
lqaHistogram1Display.Assembly = ''
# update the view to ensure updated data information
spreadSheetView3.Update()
# split cell
layout4.SplitHorizontal(0, 0.5)
# Create a new 'Bar Chart View'
barChartView3 = CreateView('XYBarChartView')
# assign view to a particular cell in the layout
AssignViewToLayout(view=barChartView3, layout=layout4, hint=2)
# show data in view
lqaHistogram1Display_1 = Show(lqaHistogram1, barChartView3, 'XYBarChartRepresentation')
# trace defaults for the display properties.
lqaHistogram1Display_1.AttributeType = 'Row Data'
lqaHistogram1Display_1.XArrayName = 'bin'
lqaHistogram1Display_1.SeriesVisibility = ['bin', 'count']
lqaHistogram1Display_1.SeriesLabel = ['bin', 'bin', 'count', 'count']
lqaHistogram1Display_1.SeriesColor = ['bin', '0', '0', '0', 'count', '0.8899977111467154', '0.10000762951094835', '0.1100022888532845']
lqaHistogram1Display_1.SeriesOpacity = ['bin', '1.0', 'count', '1.0']
lqaHistogram1Display_1.SeriesPlotCorner = ['bin', '0', 'count', '0']
lqaHistogram1Display_1.SeriesLabelPrefix = ''

########################################################################

# create new layout object
layout5 = CreateLayout(name='lqaGrid and lqaPoints')
# create a new 'lqaGrid'
lqaGrid1 = lqaGrid(registrationName='lqaGrid1')
# Properties modified on lqaGrid1
lqaGrid1.Data = 'data1.lqa'
lqaGrid1.Query = 'obs'
# Create a new 'Render View'
renderView2 = CreateView('RenderView')
renderView2.AxesGrid = 'GridAxes3DActor'
renderView2.StereoType = 'Crystal Eyes'
renderView2.CameraFocalDisk = 1.0
# show data in view
lqaGrid1Display = Show(lqaGrid1, renderView2, 'UniformGridRepresentation')
# trace defaults for the display properties.
lqaGrid1Display.Representation = 'Outline'
lqaGrid1Display.ColorArrayName = [None, '']
# add view to a layout so it's visible in UI
AssignViewToLayout(view=renderView2, layout=layout5, hint=0)
# update the view to ensure updated data information
renderView2.Update()
# create a new 'Contour'
contour1 = Contour(registrationName='Contour1', Input=lqaGrid1)
contour1.ContourBy = ['POINTS', 'obs']
contour1.Isosurfaces = [0.5]
contour1.PointMergeMethod = 'Uniform Binning'
# show data in view
contour1Display = Show(contour1, renderView2, 'GeometryRepresentation')
# trace defaults for the display properties.
contour1Display.Representation = 'Surface'
contour1Display.ColorArrayName =  [None, '']
# update the view to ensure updated data information
renderView2.Update()
# create a new 'lqaPoints'
lqaPoints1 = lqaPoints(registrationName='lqaPoints1')
# Properties modified on lqaPoints1
lqaPoints1.Data = 'data1.lqa'
lqaPoints1.Query = 'vmag > QUANTILE(data2.lqa/vmag,0.9)'
# show data in view
lqaPoints1Display = Show(lqaPoints1, renderView2, 'GeometryRepresentation')
# trace defaults for the display properties.
lqaPoints1Display.Representation = 'Surface'
lqaPoints1Display.ColorArrayName = [None, '']
# update the view to ensure updated data information
renderView2.Update()

########################################################################

# create new layout object
layout6 = CreateLayout(name='lqaQuery, lqaRegions and lqaBlock')
# create a new 'lqaQuery'
lqaQuery1 = lqaQuery(registrationName='lqaQuery1')
# Properties modified on lqaQuery1
lqaQuery1.Data = 'data2.lqa'
lqaQuery1.Query = 'vel0 < 0 AND dist > 4'
# Create a new 'Render View'
renderView3 = CreateView('RenderView')
renderView3.AxesGrid = 'GridAxes3DActor'
renderView3.StereoType = 'Crystal Eyes'
renderView3.CameraFocalDisk = 1.0
# show data in view
lqaQuery1Display = Show(lqaQuery1, renderView3, 'GeometryRepresentation')
# trace defaults for the display properties.
lqaQuery1Display.Representation = 'Surface'
lqaQuery1Display.ColorArrayName = [None, '']
# add view to a layout so it's visible in UI
AssignViewToLayout(view=renderView3, layout=layout6, hint=0)
# reset view to fit data
renderView3.ResetCamera(False)
# update the view to ensure updated data information
renderView3.Update()
# create a new 'lqaRegions'
lqaRegions1 = lqaRegions(registrationName='lqaRegions1', Input=lqaQuery1)
# Properties modified on lqaRegions1
lqaRegions1.Data = 'data2.lqa'
lqaRegions1.Query = 'obs, vel.3'
# show data in view
lqaRegions1Display = Show(lqaRegions1, renderView3, 'UniformGridRepresentation')
# trace defaults for the display properties.
lqaRegions1Display.Representation = 'Outline'
lqaRegions1Display.ColorArrayName = [None, '']
# update the view to ensure updated data information
renderView3.Update()
# create a new 'Contour'
contour2 = Contour(registrationName='Contour2', Input=lqaRegions1)
contour2.ContourBy = ['POINTS', 'obs']
contour2.Isosurfaces = [0.5]
contour2.PointMergeMethod = 'Uniform Binning'
# show data in view
contour2Display = Show(contour2, renderView3, 'GeometryRepresentation')
# trace defaults for the display properties.
contour2Display.Representation = 'Surface'
contour2Display.ColorArrayName = [None, '']
# show color bar/color legend
contour2Display.SetScalarBarVisibility(renderView3, True)
# update the view to ensure updated data information
renderView3.Update()
# set active source
SetActiveSource(lqaRegions1)
# toggle interactive widget visibility (only when running from the GUI)
ShowInteractiveWidgets(proxy=lqaRegions1Display.SliceFunction)
# toggle interactive widget visibility (only when running from the GUI)
ShowInteractiveWidgets(proxy=lqaRegions1Display)
# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=lqaRegions1Display.SliceFunction)
# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=lqaRegions1Display)
# create a new 'Stream Tracer With Custom Source'
streamTracerWithCustomSource1 = StreamTracerWithCustomSource(registrationName='StreamTracerWithCustomSource1', Input=lqaRegions1,
    SeedSource=lqaQuery1)
streamTracerWithCustomSource1.Vectors = ['POINTS', 'vel']
streamTracerWithCustomSource1.MaximumStreamlineLength = 127.0
# Properties modified on streamTracerWithCustomSource1
streamTracerWithCustomSource1.MaximumStreamlineLength = 32.0
# show data in view
streamTracerWithCustomSource1Display = Show(streamTracerWithCustomSource1, renderView3, 'GeometryRepresentation')
# trace defaults for the display properties.
streamTracerWithCustomSource1Display.Representation = 'Surface'
streamTracerWithCustomSource1Display.ColorArrayName = [None, '']
# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
streamTracerWithCustomSource1Display.ScaleTransferFunction.Points = [-0.006811988330317279, 0.0, 0.5, 0.0, 0.005550919159610221, 1.0, 0.5, 0.0]
# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
streamTracerWithCustomSource1Display.OpacityTransferFunction.Points = [-0.006811988330317279, 0.0, 0.5, 0.0, 0.005550919159610221, 1.0, 0.5, 0.0]
# hide data in view
Hide(lqaQuery1, renderView3)
# update the view to ensure updated data information
renderView3.Update()

# create a new 'Tube'
tube1 = Tube(registrationName='Tube1', Input=streamTracerWithCustomSource1)
tube1.Scalars = ['POINTS', 'AngularVelocity']
tube1.Vectors = ['POINTS', 'Normals']
tube1.Radius = 1.2704171523265542

# Properties modified on tube1
tube1.Radius = 0.5

# show data in view
tube1Display = Show(tube1, renderView3, 'GeometryRepresentation')

# trace defaults for the display properties.
tube1Display.Representation = 'Surface'
tube1Display.ColorArrayName = [None, '']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
tube1Display.ScaleTransferFunction.Points = [-0.006811988330317279, 0.0, 0.5, 0.0, 0.005550919159610221, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
tube1Display.OpacityTransferFunction.Points = [-0.006811988330317279, 0.0, 0.5, 0.0, 0.005550919159610221, 1.0, 0.5, 0.0]

# hide data in view
Hide(streamTracerWithCustomSource1, renderView3)

# update the view to ensure updated data information
renderView3.Update()

# set scalar coloring
ColorBy(tube1Display, ('POINTS', 'vel', 'Magnitude'))

# rescale color and/or opacity maps used to include current data range
tube1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
tube1Display.SetScalarBarVisibility(renderView3, True)

# get color transfer function/color map for 'vel'
velLUT = GetColorTransferFunction('vel')

# get opacity transfer function/opacity map for 'vel'
velPWF = GetOpacityTransferFunction('vel')

# get 2D transfer function for 'vel'
velTF2D = GetTransferFunction2D('vel')

# rescale color and/or opacity maps used to exactly fit the current data range
tube1Display.RescaleTransferFunctionToDataRange(False, True)

# split cell
layout6.SplitHorizontal(0, 0.5)

# set active view
SetActiveView(None)

# set active source
SetActiveSource(lqaQuery1)

# create a new 'lqaEuclideanClustering'
lqaEuclideanClustering1 = lqaEuclideanClustering(registrationName='lqaEuclideanClustering1', Input=lqaQuery1)

# Create a new 'Render View'
renderView4 = CreateView('RenderView')
renderView4.AxesGrid = 'GridAxes3DActor'
renderView4.StereoType = 'Crystal Eyes'
renderView4.CameraFocalDisk = 1.0

# show data in view
lqaEuclideanClustering1Display = Show(lqaEuclideanClustering1, renderView4, 'GeometryRepresentation')

# trace defaults for the display properties.
lqaEuclideanClustering1Display.Representation = 'Surface'
lqaEuclideanClustering1Display.ColorArrayName = [None, '']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
lqaEuclideanClustering1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 60.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
lqaEuclideanClustering1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 60.0, 1.0, 0.5, 0.0]

# add view to a layout so it's visible in UI
AssignViewToLayout(view=renderView4, layout=layout6, hint=2)

# reset view to fit data
renderView4.ResetCamera(False)

# update the view to ensure updated data information
renderView4.Update()

# create a new 'lqaClusterSelect'
lqaClusterSelect1 = lqaClusterSelect(registrationName='lqaClusterSelect1', Input=lqaEuclideanClustering1)

# Properties modified on lqaClusterSelect1
lqaClusterSelect1.Rank = 0

# show data in view
lqaClusterSelect1Display = Show(lqaClusterSelect1, renderView4, 'GeometryRepresentation')

# trace defaults for the display properties.
lqaClusterSelect1Display.Representation = 'Surface'
lqaClusterSelect1Display.ColorArrayName = [None, '']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
lqaClusterSelect1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
lqaClusterSelect1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# hide data in view
Hide(lqaEuclideanClustering1, renderView4)

# update the view to ensure updated data information
renderView4.Update()

# create a new 'lqaBlock'
lqaBlock1 = lqaBlock(registrationName='lqaBlock1', Input=lqaClusterSelect1)

# Properties modified on lqaBlock1
lqaBlock1.Data = 'data2.lqa'
lqaBlock1.Query = 'obs, vel.3'

# show data in view
lqaBlock1Display = Show(lqaBlock1, renderView4, 'UniformGridRepresentation')

# trace defaults for the display properties.
lqaBlock1Display.Representation = 'Outline'
lqaBlock1Display.ColorArrayName = [None, '']

# init the 'Plane' selected for 'SliceFunction'
lqaBlock1Display.SliceFunction.Origin = [7.5, 103.5, 119.5]

# update the view to ensure updated data information
renderView4.Update()

# create a new 'Stream Tracer With Custom Source'
streamTracerWithCustomSource2 = StreamTracerWithCustomSource(registrationName='StreamTracerWithCustomSource2', Input=lqaBlock1,
    SeedSource=lqaClusterSelect1)
streamTracerWithCustomSource2.Vectors = ['POINTS', 'vel']
streamTracerWithCustomSource2.MaximumStreamlineLength = 47.0

# show data in view
streamTracerWithCustomSource2Display = Show(streamTracerWithCustomSource2, renderView4, 'GeometryRepresentation')

# trace defaults for the display properties.
streamTracerWithCustomSource2Display.Representation = 'Surface'
streamTracerWithCustomSource2Display.ColorArrayName = [None, '']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
streamTracerWithCustomSource2Display.ScaleTransferFunction.Points = [-0.008691775663042448, 0.0, 0.5, 0.0, 0.02283624724879691, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
streamTracerWithCustomSource2Display.OpacityTransferFunction.Points = [-0.008691775663042448, 0.0, 0.5, 0.0, 0.02283624724879691, 1.0, 0.5, 0.0]

# hide data in view
Hide(lqaClusterSelect1, renderView4)

# update the view to ensure updated data information
renderView4.Update()

# set active source
SetActiveSource(lqaBlock1)

# toggle interactive widget visibility (only when running from the GUI)
ShowInteractiveWidgets(proxy=lqaBlock1Display.SliceFunction)

# toggle interactive widget visibility (only when running from the GUI)
ShowInteractiveWidgets(proxy=lqaBlock1Display)

# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=lqaBlock1Display.SliceFunction)

# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=lqaBlock1Display)

# create a new 'Contour'
contour3 = Contour(registrationName='Contour3', Input=lqaBlock1)
contour3.ContourBy = ['POINTS', 'obs']
contour3.Isosurfaces = [0.5]
contour3.PointMergeMethod = 'Uniform Binning'

# show data in view
contour3Display = Show(contour3, renderView4, 'GeometryRepresentation')

# trace defaults for the display properties.
contour3Display.Representation = 'Surface'
contour3Display.ColorArrayName = [None, '']

# update the view to ensure updated data information
renderView4.Update()

# reset view to fit data
renderView4.ResetCamera(False)

# set active view
SetActiveView(renderView3)

# reset view to fit data
renderView3.ResetCamera(False)

# set active view
SetActiveView(renderView4)

# set active source
SetActiveSource(streamTracerWithCustomSource2)

# create a new 'Tube'
tube2 = Tube(registrationName='Tube2', Input=streamTracerWithCustomSource2)
tube2.Scalars = ['POINTS', 'AngularVelocity']
tube2.Vectors = ['POINTS', 'Normals']
tube2.Radius = 0.5

# show data in view
tube2Display = Show(tube2, renderView4, 'GeometryRepresentation')

# trace defaults for the display properties.
tube2Display.Representation = 'Surface'
tube2Display.ColorArrayName = [None, '']

# hide data in view
Hide(streamTracerWithCustomSource2, renderView4)

# set scalar coloring
ColorBy(tube2Display, ('POINTS', 'vel', 'Magnitude'))

# update the view to ensure updated data information
renderView4.Update()

