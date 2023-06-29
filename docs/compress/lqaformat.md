# Block-Wise Grid and Index Compression

*Lossy In-Situ Tabular Encoding for Query-Driven Analytics* (LITE-QA) applies a compact encoding for grid and index, and uses selected lossless compressors for efficient compression of the specific data parts $X_E$ and $X_M$ produced by GLATE, as explained in *LITE-QA Grid Compression*, and $N$ and $B_j$ produced by EBATE, as explained in *LITE-QA Index Compression*.
For lossless compression of high-entropy parts, i.e.\ $X_M$, $N$ and $B_j$, the codec `fastpfor`\ [@Lemire2014] is used, while low-entropy parts $X_E$ of the floating point data are compressed using `zstd`, a fast general purpose lossless compressor with a very fast decoder.

In order to ensure a high degree of data accessibility and allow for partial decompression of simulation grids, the streams $X_E$ and $X_M$, each of length $l=128^3$ values, are split into 512 equally-sized chunks $X^j_E$ and $X^j_M$ for $j=1,2,\ldots,512$.
Chunks are arranged in an interlaced sequence of exponents and mantissa parts $O_X,X^1_{EM},X^2_{EM},X^3_{EM},\ldots$, where each pair of chunks $X^j_{EM}$ represents a cubic subgrid of $16^3=4,096$ voxels according to the Hilbert-curve linearization as described in *LITE-QA Grid Linearization*.
$O_X$ represents the offset table, which stores the size of the compressed blocks $X^j_{EM}$ required for partial grid decompression, and is stored uncompressed.

The index data is arranged in the sequence $L,N,C,O_B,B_1,B_2,\ldots$, where $L,N,C,O_B$ constitute the index header with look-up table size $L$, bin look-up table $N$, bin sizes $C$ and offset table $O_B$.
Each bin $B_j$ is compressed independently, unless the size $c_j$ is smaller than 128 indices. In the latter case the bin is merged with the next bin, until the size exceeds 128\ [@Lehmann2018].
$O_B$ contains the sizes of the compressed index bins, which are required for partial index decompression.
$L$, $C$ and $O_B$ are stored uncompressed.

# Quality of Decompressed Data

Fig.\ @fig:comerror shows the error distribution for decompressed data using LITE-QA and ZFP.
For LITE-QA, different choices of $\omega$ resulting in a maximum error $e_{\mathrm{R}}$ between 0.01% and 1% are shown.
ZFP is being operated on precision level 12--18.
As can be seen, LITE-QA restricts the point-wise maximum relative error by using error-bounded quantization, while ZFP restricts the average precision of decompressed data, i.e.\ the average relative error.
The GLATE non-temporal compression using $\omega=35$ and $e_{\mathrm{R}}=1$% is slightly worse by 2.5% as compared to ZFP in terms of compression ratio, however GLATE guarantees the error bound for all values.

![Error distribution for decompressed data. (1)\ LITE-QA using maximum point wise error $e_{\mathrm{R}}$ between 0.01% and 1% corresponding to different choices for $\omega$ as describes in *LITE-QA Float Quantization*. (2)\ ZFP using precision level 12--18. While GLATE bounds the error for all values, ZFP restricts the average relative error.](img/comerror.png){#fig:comerror}

Fig.\ @fig:decomerror shows visualizations based on LITE-QA decompressed data with at most 1% point-wise error, and raw uncompressed data.
Streamlines are seeded in the inlet of the filter and traced along the complete filter depth.
As can be seen, streamlines start to diverge after longer distances of tracing at the outlet of the filter.
The visual difference for short paths in the inlet is merely perceivable, hence, the decompressed data is sufficient for the task of local visualization, e.g.\ for the visualizations local regions in the flow field.

![Error accumulation in visualizations based on decompressed data. (1)\ Streamlines are seeded in the inlet and traced along the filter depth. (2)\ Visual results for short distances of tracing is sufficiently accurate for local visualization of fluid flow. (3)\ decompression artifacts, i.e.\ diverging streamlines reaching the outlet of the filter after tracing along the whole filter depth. Visualization based on uncompressed data is shown in green, based on decompressed data with point-wise maximum error of 1% in red.](img/decomerr.png){#fig:decomerror}

