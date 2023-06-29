#  Floating Point Quantization

The second step of the in-situ processing pipeline concerns quantization of the linearized data sequences $X$ for each simulation variable $u,v,w,M,Q$.
The quantization of floating point values is performed on a global step function $S(n)$ based on the integer step numbers $n$ given as $\ldots,-2,-1,0,1,2,\ldots$.
The function $S(n)$ restricts the quantization error in percent as a configurable point-wise maximum error of e.g.\ $0.01\ldots1\%$ for each value in the grid and the index.
The step function uses an optimized data encoding based on a cyclic pattern of mantissa bits for compact representation of the step values $n$ and differences between two quantized values $S(n')-S(n)$.
The compact encoding allows for the application of standard lossless compression and bit-packing as a backend with high compression rates and high compression speeds for grid and index as well.

###### **Cyclic Pattern of Mantissa Bits**

The step function $S(n)$ is chosen such that the step size increases with respect to increasing $n$ without violation of the error policy, i.e.\ maximum point-wise error of $e_{\mathrm{R}}\times100$ in percent.
Interestingly, particular choices of $e_{\mathrm{R}}$ imply a cyclic pattern of mantissa bits for values $S(n),S(n+\omega),S(n+2\omega),\ldots$ with cycle length $\omega$ as illustrated in Fig.\ @fig:stepfun.
The condition holds for certain choices of the maximum error, e.g.\ 0.99%, 0.49% or 0.27% as shown in Fig.\ @fig:stepwidth\ (1).

![Illustration of the step function $S(n)$ for $n\geq1$ with cyclic mantissa bits using a short cycle legth of $\omega=4$ resulting in point-wise maximum error of $e_{\mathrm{R}}=8.64$%. Mantissa bits repeat for $n,n+4,n+8,\ldots$, and therefore allow for efficient compression. The log scale view illustrates the equal spacing of step function in logarithmic space.](img/stepfun.png){#fig:stepfun}

###### **Step Function Definition**

The function $S(n)$ is defined using the so-called positive step function $T(n)$ according to the definition
$$
\begin{array}{rcl}
T(n) & = & x_{\mathrm{Z}}\cdot\left(\dfrac{1+e_{\mathrm{R}}}{1-e_{\mathrm{R}}}\right)^n\mbox{,}\\
S(n) & = & \left\{\begin{array}{ll}
0          & \mbox{, for }n=0\\
-T(|n|-1|) & \mbox{, for }n<0\\
+T(n-1)    & \mbox{, for }n>0\mbox{.}\\
\end{array}\right.\\
\end{array}
$$
The positive step function starts at a small non-zero number $T(0)=x_{\mathrm{Z}}$ and uses the error bound $e_{\mathrm{R}}$, for definition of the step width.
The transition from step $T(n)$ to $T(n+1)$ occurs when the maximum quantization error of $e_{\mathrm{R}}$ is met according to $T(n)\cdot(1+e_{\mathrm{R}})=T(n+1)\cdot(1-e_{\mathrm{R}})$.
The function $S(n)$ extends the positive step function $T(n)$ for negative step numbers $n<0$ and introduces a zero value as $S(0)=0$ for all values $|x|<x_{\mathrm{Z}}$.
The error between a value $|x|\geq x_{\mathrm{Z}}$ and $S(n)$ is defined in percent according to
$$|x-S(n)|/|x| \leq e_{\mathrm{R}}\times100\%\mbox{.}$$

Given a floating point value $x$, the quantization is performed by mapping $x$ to $\tilde{x}=S(n(x))$ by calculating the step number $n$ according to
$$
\begin{array}{rcl}
s(x) & = & \left\{\begin{array}{ll}
0    & \mbox{, for }|x|<x_{\mathrm{Z}}\\
\pm1 & \mbox{, for }|x|\geq x_{\mathrm{Z}}\\
\end{array}\right.\\
n(x) & = & s\cdot\left\lfloor\dfrac{\log|x|-\log x_{\mathrm{Z}}}{\log(1+e_{\mathrm{R}})-\log(1-e_{\mathrm{R}})}+\dfrac{3}{2}\right\rfloor\mbox{,}
\end{array}
$$
where $s$ indicates the sign of $x$.

###### **Cyclic Encoding with Mantissa Look-Up Table**

The cyclic pattern of mantissa bits, as shown in Fig.\ @fig:stepfun, is arranged by choosing $e_{\mathrm{R}}$ and $x_{\mathrm{Z}}$ depending on $\omega$ and $\delta$ according to
$$
\begin{array}{rcl}
e_{\mathrm{R}} & = & (\sqrt[\omega]{2}-1)/(\sqrt[\omega]{2}+1) \\
x_{\mathrm{Z}} & = & 2^{-\delta}\mbox{.}
\end{array}
$$
As illustrated in Fig.\ @fig:stepwidth\ (2), for a low resolution step function with maximum error 8.64%, the mantissa parts $m=0,1,2,3$ repeat after the cycle length $\omega=4$ steps for $n,n+\omega,n+2\omega,\ldots$.
$\omega$ and $\delta$ are specified by the user and determine the quality of the decompressed data as well as of binning during index creation.
$\omega$ defines the length of one cycle of the mantissa bits and $\delta$ defines the actual values of $S(n)=\pm x_{\mathrm{Z}}$ for $n=\pm1$.

![Error bounded cyclic quantization of mantissa. (1)\ Cycle length $\omega$ of mantissa bits and corresponding resulting maximum error bound $e_{\mathrm{R}}$ in percent. (2)\ Illustration of cyclic mantissa look-up table $\Omega$ for a low resolution step function with cycle length $\omega=4$. The exponent part $e$ and mantissa part $m$ are shown for values of $S(n)$ with $n\geq1$, where $n$ is obtained by $n=s\cdot[(e-1)\cdot \omega+m-1]$ with $s=+1$.](img/stepwidth.png){#fig:stepwidth}

###### **Quantization and Reconstruction**

The cyclic values of the mantissa bits are stored into the so-called mantissa look-up table $\Omega_m$ for $m=0,1,\ldots,\omega-1$.
Due to the cyclic encoding using the mantissa look-up table $\Omega$, as shown in Fig.\ @fig:stepwidth\ (2), any step number $n\neq0$ can be translated into sign $s=\pm1$, exponent part $e>0$ and the mantissa part $0\leq m<\omega$ according to the following equations:
$$
\begin{array}{rcl}
s & = & \left\{\begin{array}{ll}
+1  & \mbox{, for }n>0\\
-1  & \mbox{, for }n<0\\
\end{array}\right.\\
e & = & \lceil |n|/\omega\rceil\\
m & = & (|n|-1)\mbox{ modulo }\omega\mbox{,}\\
n & = & s\cdot\left[(e-1)\cdot \omega+m-1\right]\mbox{.}
\end{array}
$$
Given a linearized sequence $X=\{x_1,x_2,\ldots,x_l\}$, as explained in *LITE-QA Grid Linearization*, the quantization of each value is performed by computing the step number $n(x)$ under consideration of the step function parameters $\omega$ and $\delta$.
Based on $n$, the sign $s$, the exponent part $e$ and the mantissa part $m$ are determined as described, whereas the combination $e=m=s=0$ encodes the zero $S(n)=0$ for values $|x|<x_{\mathrm{Z}}$ with $n=0$.

The fast implementation of the quantization procedure $n(x)$ directly determines $s$ and $e$ from the bits of the floating point value $x$ using standard operations, while $m$ is determined using fast search inside the cyclic mantissa look-up table $\Omega$\ [@Lehmann2018].
For fast value reconstruction, $\tilde{x}=S(n)$ is computed given a triplet $(e,m,s)$ using the following procedure:

* If $e=m=0$, then set $\tilde{x}$ to zero,
* else, if $e>0$, then initialize $\tilde{x}$ by
	1. setting the sign to $s$ and the exponent to $e-\delta-1$ and
	2. setting the mantissa bits to $\Omega_m$.
