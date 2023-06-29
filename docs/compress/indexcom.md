# Index Variable Encoding

A challenge for in-situ indexing is its efficient integration with compression, in order to reduce storage requirements, while allowing for fast search of grid cells based on data values during post-processing.
Where bitmap-based indices on floating point data often require a large storage footprint of $\geq100\%$ additionally to the data\ [@Wu2009], recent indexing methods for scientific data like ALACRITY\ [@Jenkins2013] and DIRAQ\ [@Lakshminarasimhan2014] reduce the storage footprint to 50--100% for both index and data.

The procedure used for index creation in *Lossy In-Situ Tabular Encoding for Query-Driven Analytics* (LITE-QA) is called *Error-bounded Binning and Tabular Encoding* (EBATE), which generates an inverted index through binning, similar to the differential encoding used in the lossy floating point compressor SBD\ [@Iverson2012;@Lehmann2018].
EBATE uses bins with increasing width, according to the definition of the global step function $S(n)$, under consderation of the user-defined parameters $\omega$ and $\delta$ for value quantization.
The bins are tagged with the global step numbers $n$, and therefore, the bin values underlie the error bound $e_{\mathrm{R}}$, e.g.\ $0.01\ldots1\%$.
The index allows for the fast reconstruction of the location of grid cells based on their value within the error bound.

###### **Index Generation for Fluid Domain**

The procedure for index generation, as illustrated in Fig.\ @fig:indexgen, consists of three steps, which are applied to all index variables $u,M,Q$ on all simulation processes independently.
First, the local grid in each simulation process is linearized using the Hilbert-curve and the quantization scheme is applied as described in *LITE-QA Grid Linearization* and *LITE-QA Float Quantization* respectively.
The values $x_i$ inside the fluid domain are transformed into the step numbers $n(x_i)$ on the global step function $S(n)$ for the complete local grid $i=1\ldots l$ in each process.
During linearization for index creation, the voxels of the filter structure are skipped as they do not account to the fluid domain.
Hence, unneeded voxels are not in the index and therefore do not create cluttered results with grid cells ususally blanked to zero as placeholders.
The index $i$ of the linearization sequence corresponds to the original location of grid cells and is counted consistently while filter voxels are skipped.

![Index generation for fluid domain consists of four steps: (1)\ linearization of simulation grid and quantization of data values, (2)\ lexicographical sorting of data $(\bar{n},i)$ for determination of the bin look-up table $N=\{n_j\}$ and the bin sizes $C=\{c_j\}$, and (3)\ collecting monotonic sequences of grid cell indices $i$ for all bins $B_j$ from the sorted data. Monotonic sequences of grid cell indices $i$ in index bins $B_j$ are encoded using differences $\Delta^B$.](img/indexgen.png){#fig:indexgen}

Second, after quantization is finished, all tuples $(\bar{n},i)$ composed of integer step numbers $\bar{n}=n(x_i)$ and their corresponding locations $i$ are sorted lexicographically in ascending order with respect to the first and second component.
Due to the lexicographical sorting, large amounts of consecutive tuples in the sorted sequence, which share the same quantization step number $\bar{n}$, are being arranged in clusters $j=1\ldots L$ with size $c_j$, which reflects the amount of repetition of tuples in the sorted sequence with the same $\bar{n}$, hence $\sum c_j=l$.
The amount of clusters $L$ depends on the data and varies in every time step.
For each cluster $j$, the $L$ unique step numbers $\bar{n}_j$ and corresponding cluster sizes $c_j$ are collected in arrays $N=\{\bar{n}_j\}$ and $C=\{c_j\}$ accordingly.

Third,  for all values $k=1\ldots c_j$ in the clusters $j=1\ldots L$, the sorted locations $i^j_k$ form sequences of monotonic increasing grid cell indices $i^j_1>i^j_2>\ldots>i^j_{c_j}$.
Those indices are collected in index bins $B_j=\{i^j_1,i^j_2,\ldots\}$, which constitute a so-called inverted index.

###### **Compact Index Data Encoding**

The results of the index generation procedure are the bin look-up table $N=\{\bar{n}_j\}$, bin sizes $C=\{c_j\}$ and index bins $B_j$, for $j=1\ldots L$ clusters.
As scientific data usually concentrates within a smaller bounded value range, sorting the linearized data $X$ yields smooth monotonic sequences\ [@Iverson2012;@Lakshminarasimhan2014].
This allows for a compact representation of the bin look-up table $N=\{\bar{n}_1,\Delta^N_2,\Delta^N_3,\ldots\}$ and the grid cell indices in index bins $B_j=\{i^j_1,\Delta^B_2,\Delta^B_3,\ldots\}$ using difference encoding as shown in Fig.\ @fig:idxnoise.
The differences in $N$ are computed between step numbers, i.e.\ $\Delta^N_j=n_j-n_{j-1}$, whereas the differences in $B_j$ are computed between grid cell indices, i.e.\ $\Delta^B_k=i^j_k-i^j_{k-1}$.

![Compact encoding for the index. Absolute values in the bin look-up table $N$ and in the index bins $B_j$ are replaced with differences $\Delta^N_j$ and $\Delta^B_i$ respectively. (1)\ Absolute values in $N$ and $B_j$, before difference encoding. (2)\ Noise reduction after application of difference encoding $\Delta^N$ and $\Delta^B$. The first value $\bar{n}_1$ in the bin look-up table $N$ and $i^j_1$ in all index bins $B_j$ is stored as absolute value.](img/idxnoise.png){#fig:idxnoise}
