# `liteqa` -- Lossy In-Situ Tabular Encoding for Query-Driven Analytics

This is the reference implementation of `liteqa`, as applied in the Virtual Prototyping Scenario described in [Ref1]().

`liteqa` is supposed to run on Linux. MacOS and Windows are not tested yet.
`liteqa` was applied in a virtual prototyping scenario for aluminum metal melt filters with unconventional designs, which is described in [AEM Journal, Volume 24, Issue 2](https://onlinelibrary.wiley.com/doi/10.1002/adem.202100878) and [Springer Book Series in Materials Science by Springer Nature]().

> *Computer-Aided Design of Metal Melt Filters: Geometric Modifications of Open-Cell Foams, Effective Hydraulic Properties and Filtration Performance*; Henry Lehmann, Eric Werzner, Alexander Malik, Martin Abendroth, Subhashis Ray, Bernhard Jung; AEM Journal, Volume 24, Issue 2, February 2022, Page 2100878

> * Multifunctional Ceramic Filter Systems for Metal Melt Filtration: Towards Zero-Defect Materials*; Edited by:
Christos G. Aneziris and Horst Biermann, released in 2023, ISBN: *to come*

`liteqa` is a data compression utility with integrated indexing for query-driven analytics on compressed data.
`liteqa` is used for the generation of compressed indices and compressed grids from cubic uniform grid data e.g. from LBM simulations or stored in vtkImageData files.
Visualizations are generated directly from compressed contents in a query-driven manner, where only the data parts required for the visulization are loaded and decompressed from disk.

The first version of `liteqa` using for compression and indexing of scientific data is described in [CSE 2018 Conference Proceedings](https://ieeexplore.ieee.org/document/8588228).

> *Efficient Visualization of Large-Scale Metal Melt Flow Simulations Using Lossy In-Situ Tabular Encoding for Query-Driven Analytics*; Henry Lehmann, Eric Werzner, Cornelius Demuth, Subhashis Ray, Bernhard Jung; 2018 IEEE International Conference on Computational Science and Engineering (CSE); Year: 2018

`liteqa` uses t-GLATE for error-bounded compression of numerical 32-bit `float` data, e.g. maximum error of 1%, as described in[CSCI 2018 Conference Proceedings](https://www.computer.org/csdl/proceedings-article/csci/2018/136000b386/1gjRqFMGJIk).

> *Temporal In-Situ Compression of Scientific Floating Point Data with t-GLATE*; Henry Lehmann, Bernhard Jung; 2018 International Conference on Computational Science and Computational Intelligence (CSCI); Year: 2018, Pages: 1386-1391

The latest implementation of the `liteqa` compression and the `liteqa` data format, as applied in the above menioned virtual prototyping scenario, is described in `docs/compress/*.md` and in `docs/compress.pdf`.

`liteqa` comes with tools for accessing the features on the command line:

1. example data `data1/2.vti` and scripts `compress.sh`, `query.sh` and `plugin.py`
2. compression of vtkImageData using `vti2liteqa.py` on the command line
3. perform queries in indices using `lqaquery.py` on the command line
4. ParaView Plugin for performing `liteqa` queries in an interactive GUI

---

# Example

The example demonstrates `liteqa` compression and `liteqa` queries using two example CFD datasets `example/data1.vti` and `data2.vti` composed of 128**3 voxels each containing a flow field `vec` and a blanking mask `obs`.
Several compressed indexes and compressed grids are generated from the example data.

`liteqa` reaches following compression rate on `data1.vti` with a maximum point-wise error of 1%:

File             |Rate  |Type |Info
-----------------|------|-----|----------------------------
data1.lqa/vel0.i |34.74%|Index|Velocity Field Component 0
data1.lqa/dist.i |33.57%|Index|Fluid Domain Distance Field
data1.lqa/qcrit.i|17.66%|Index|Flow Field Q-Criterion
data1.lqa/vmag.i |34.13%|Index|Velocity Magnitude
data1.lqa/vort.i |33.69%|Index|Flow Field Vorticity
data1.lqa/vel0.g |19.11%|Grid |Velocity Field Component 0
data1.lqa/vel1.g |28.81%|Grid |Velocity Field Component 1
data1.lqa/vel2.g |29.21%|Grid |Velocity Field Component 2
data1.lqa/obs.g  |3.05% |Grid |Fluid Domain Geometry

The example is run without installing `liteqa` in the system.
First, compile `liteqa` shared object.
```bash
cd liteqa/liteqa_c/
make
```

Second, generate compressed indexes and grids using the following commands:
```bash
cd liteqa/example/
./compress.sh
```

Third, perform queries on the example data using the following commands:
```bash
cd liteqa/example/
./query.sh
```

---

# Install `liteqa` in Local System

To install run the following commands in the `liteqa/` source directory:
```bash
cd liteqa
pip install .
```

`liteqa` requires the following dependencies.
Dependencies are installed by pip as required:

* `pyfastpfor`: for bitpacking on the compressor data
* `zstd`: for lossess compression of the compressor data
* `hilbertcurve`: for the generation of 3D hilbert linearization curves
* `numpy`: numerical computing for Python

`liteqa` requires a C compiler for the compilation of the C shared object of the compressor.
The compilation is done using python `distutils`.
The shared object is loaded and executed using python `ctypes`.

`liteqa` requires Paraview.
The stable binary version of ParaView from the [ParaView website](https://www.paraview.org/) can be used, **only if** it uses the system site-packages of the default `python3` interpreter.
Otherwise, ParaView must be installed using a package manager, or it must be compiled from source in order to find the `liteqa` python package installed in the local system.
When using the binary version from the website or manual compilation, make sure that the `pvpython` executable is accessible in the PATH.

---

# Tools

The following `liteqa` tools are availaible in the PATH after pip installation.

## `liteqa` Compressor -- `vti2liteqa.py`

`vti2liteqa.py` is used to create compressed indices and compressed grids from vtkImageData files.
The index and the data grid is compressed in a block-wise manner.
In order to compress in-situ e.g. during a numerical simulation the compression blocks for data grids and data indices must be generated according to the compressed file format as described in the `liteqa` documentation.

For further description of the command line compressor refer to `example/compress.sh`.

## `liteqa` Query Tool -- `lqaquery.py`

`lqaquery.py` is used to execute querys on commandline.

For further description of the command line query tool refer to `example/query.sh`.

## `liteqa` ParaView Plugin

The *count query*, *index query*, and *grid query* as described in [Ref1]() are implemented as data producers for the ParaView Pipeline in `lqaPlugin.py`.
The plugin is used directly inside the GUI of the ParaView application.

For further description of the plugin refer to `example/plugin.py`

