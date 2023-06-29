---
title: "LITE-QA Compression"
bibliography: references.bib
---

## Data Management

*Lossy In-Situ Tabular Encoding for Query-Driven Analytics* (LITE-QA) supports the data management by limiting the amount of data stored during the simulation phase and the amount of data loaded for the generation of visualizations.
LITE-QA provides compact storage of full-resolution data grids and data indices as well as improved data access methods for visualization based on compressed contents.

For a typical CFD data set containing index and data for metal melt flow simulation, LITE-QA achieves a data reduction of three to four fold for index and data combined.
For the aluminum simulations carried out with the LBM, the compression achieves a 5 fold average data reduction for stationary simulation grids, up to 12 fold data reduction for high-resolution temporal data grids and a average data reduction by factor of three for data indices while guaranteeing a maximum decompression error of 1%.

LITE-QA realizes compression and index generation based on a floating point encoding for absolute values and differences, which is optimized for a lossless compression backend and achieves high compression rates at high compression speeds\ [@Lehmann2016;@Lehmann2018a].
The floating point compression and decompression operate directly inside the simulation code and the visualization application without noteworthy decline of the run-time as compared to using uncompressed data, therefore improving the data management\ [@Lehmann2018d].

LITE-QA can store additional low-resolution data with 1/8 of the resolution.
Storing low-resolution data uncompressed would demand for additional 12.5% storage space.
LITE-QA provides access to low-resolution contents with only 3% of additional storage.

In combination with an in-situ generated index, LITE-QA improves the time-to-analysis by employing a query-driven approach for selective loading and decompression of only the data needed for the task at hand.
The query-driven approach to decompression allows for efficient localization and visualization of interesting phenomena leaving large parts of the data untouched, which are not needed.
The index-based localization and decompression accelerates data loading for typical visualization tasks by a factor of 16 as compared to loading uncompressed data and performing linear search\ [@Lehmann2018].

For the compression of simulation grids and data indices, two schemes are used, which consist of grid linearization and tabular encoding (GLATE) and error-bounded binning and tabular encoding (EBATE).
The compression performance competes with state-of-the art in-situ compression methods like ISABELA\ [@Lakshminarasimhan2011], ZFP\ [@Lindstrom2014] and SZ\ [@Di2016] as well as with in-situ indexing methods like ALACRITY\ [@Jenkins2013], ISABELA-QA\ [@Lakshminarasimhan2011a] and DIRAQ\ [@Lakshminarasimhan2014] with respect to data reduction while restricting the point-wise maximum error for decompressed contents.
The temporal compression schemes t-GLATE\ [@Lehmann2018d;@Lehmann2018a] as well as the indexing sheme EBATE operate at high speeds and provide a trade off between data accuracy and compression rate based on a error bound in percent.
