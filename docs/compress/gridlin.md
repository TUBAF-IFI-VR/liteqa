# Linearization of Simulation Grid

In order to prepare the data for compression and index generation, the first step of the *Lossy In-Situ Tabular Encoding for Query-Driven Analytics* (LITE-QA) in-situ processing pipeline consists of applying a linearization procedure to the local simulation grids in each parallel process.
The Hilbert-curve is used to construct a single polygonal chain always passing through one direct neighbor of each voxel cell without jumps as shown in Fig.\ @fig:linearization\ (1).
Therefore, the spatial adjacency of grid cells is preserved to a large extent in the sequential values of the linearized data stream, which has positive effects on the resulting compression rate.
The linearization, called $X$, is applied subsequently to all simulation variables contained in the local grid before the actual data compression is performed in parallel by each simulation process independently.

![Linearization of voxel grid based on Hilbert-curve. (1)\ Local grids of simulation processes are linearized into a sequence $X$. (2)\ The smooth linear data sequences $X$ are stored as a compressed grid for the variables $u,v,w$, and as compressed index for the variables $u,M,Q$ accordingly.](img/lincomp.png){#fig:linearization}

For the present visualization task in the context of the virtual prototyping scenario, the data set consists of the flow field $u,v,w$ and two additional flow field properties, i.e.\ velocity magnitude $M$ and local vortex characteristics $Q$[^qcrit].
As shown in Fig.\ @fig:linearization\ (2), after linearization, the variables $u,v,w$ are stored in a compressed grid, and $u,M,Q$ are stored in a compressed index.
In the experiments describes in @Lehmann2023, the simulations are carried out by $64$ parallel processes on a global grid with $512^3$ voxels.
Thus, for each grid corresponding to the variables $u,v,w$, and for each index corresponding to $u,M,Q$, each single process manages voxel linearizations $X$ of length $l=128^3$  for compression and index generation.
However, before the actual compression and index generation takes place, the linearized data streams $X$ are quantized for all variables as explained in the next section.
Based on the quantized data, the compression and index generation is performed as explained in *LITE-QA Grid Compression* and *LITE-QA Index Compression*.

[^qcrit]: The velocity magnitude $M=\sqrt{u^2+v^2+w^2}$ and the Q-criterion $Q$ are computed on-the-fly before compression. $Q={}^1/{}_2(||\Omega||^2-||S||^2)$, where $S$ and $\Omega$ are the symmetric and the anti-symmetric components of the velocity gradient tensor\ [@Dong2016]. They are also known as the strain rate tensor and the rotation tensor respectively. The unit for values of variable $Q$ is omitted.

