Mathematical Background
=======================

This page summarizes the Hermes formulation behind PyHermes. The same
mathematical object appears throughout the package: a weighted point catalog is
turned into a continuous multiresolution field, window functions are applied as
convolutions, and statistics are read out as field averages or sampled products.

Point Catalogs And Window Counts
--------------------------------

A particle or halo catalog is treated as a weighted spatial point process,

.. math::

   n(\mathbf{x}) =
   \sum_{i=1}^{N} w_i\,
   \delta_{\rm D}^{(3)}(\mathbf{x}-\mathbf{x}_i).

Here :math:`w_i` can be a unit weight, a mass weight, or another mark carried by
the object. In PyHermes the stored example fields are normalized by the total
input weight, so the resulting field integrates to one and the uniform density
in a periodic box is :math:`1/V`.

Counting in any geometric volume is written as a convolution with a normalized
window function,

.. math::

   n_W(\mathbf{x})
   =
   (W \circ n)(\mathbf{x})
   =
   \int W(\mathbf{x}-\mathbf{x}') n(\mathbf{x}')\,d^3x'
   =
   \sum_i w_i W(\mathbf{x}-\mathbf{x}_i),
   \qquad
   \int W(\mathbf{x})\,d^3x = 1.

Changing the window changes the statistic: a sphere gives count-in-cell values,
a shell gives pair counts, and redshift-space windows describe cylindrical or
ring-like pair geometries.

Multiresolution Field Reconstruction
------------------------------------

``Convols`` replaces the singular Dirac-delta catalog by scaling-function
coefficients on a compact multiresolution basis,

.. math::

   n_j(\mathbf{x}) =
   \sum_{\ell} \epsilon_{j\ell}\,\phi_{j\ell}(\mathbf{x}),
   \qquad
   \epsilon_{j\ell}
   =
   \int n(\mathbf{x})\phi_{j\ell}(\mathbf{x})\,d^3x
   =
   \sum_i w_i \phi_{j\ell}(\mathbf{x}_i).

Applying a window to the field becomes a linear operation on those
coefficients,

.. math::

   \widetilde{\epsilon}_{j\ell}
   =
   \sum_m W^j_{\ell m}\epsilon_{jm}.

For homogeneous windows, :math:`W^j_{\ell m}` has convolution structure, so the
operation can be evaluated efficiently with FFTs. This is the reason downstream
tasks can reuse a saved ``ConvolsData`` object instead of returning to the raw
catalog.

Counting
--------

``Counting`` evaluates a smoothed field at random centers
:math:`\{\mathbf{y}_a\}`. For a count-in-cell window this is simply

.. math::

   c_a = n_W(\mathbf{y}_a),
   \qquad
   p(c)\simeq {1\over N_{\rm sample}}
   \sum_a \mathbf{1}\!\left[c_a \in {\rm bin}(c)\right].

For fluctuation fields, with
:math:`\delta=(n-\bar n)/\bar n`, the same operation probes
:math:`\delta_W = W\circ\delta`. Moments of these samples estimate quantities
such as

.. math::

   \sigma_W^2
   =
   \langle \delta_W, \delta_W\rangle.

Two-Point Correlations
----------------------

The in-situ view of the two-point correlation function writes a pair statistic
as a local product between the field and a windowed copy of itself,

.. math::

   \xi_P =
   \left\langle
   \delta(\mathbf{x})\,
   (W_P\circ\delta)(\mathbf{x})
   \right\rangle.

For the usual isotropic real-space 2PCF, :math:`P` is a shell radius. A thin
spherical shell has

.. math::

   W_{\rm shell}(r;R)
   =
   {1\over 4\pi R^2}\delta_{\rm D}(r-R),
   \qquad
   \widehat{W}_{\rm shell}(k;R)
   =
   {\sin(kR)\over kR}.

The finite-bin version is a normalized spherical shell between
:math:`R_{\rm in}` and :math:`R_{\rm out}`. In Fourier space it is the
corresponding difference of two spherical top-hat windows.

In redshift space the line of sight introduces axial symmetry. With transverse
separation :math:`r_\perp` and line-of-sight separation :math:`r_\parallel`,
the thin ring window is

.. math::

   W_{r_\perp,r_\parallel}(\rho,z)
   =
   {1\over 2\pi r_\perp}
   \delta_{\rm D}(\rho-r_\perp)
   \delta_{\rm D}(z-r_\parallel),

and its Fourier-space form contains the Bessel factor

.. math::

   \widehat{W}_{r_\perp,r_\parallel}(k_\perp,k_\parallel)
   =
   e^{i k_\parallel r_\parallel}
   J_0(k_\perp r_\perp).

In practice PyHermes can use shell, ring, disk, or cylinder pair windows, all
with the same coefficient-level machinery.

Random Fields And Estimators
----------------------------

Random fields encode the survey volume or selection function and provide the
normalization for correlation measurements. The familiar Landy-Szalay estimator
is

.. math::

   \widehat{\xi}_{\rm LS}
   =
   {DD - 2DR + RR\over RR}.

The same idea extends compactly to higher orders as the
Szapudi-Szalay form,

.. math::

   \widehat{\xi}_N
   =
   {\prod_{i=1}^N (D_i-R_i)\over \prod_{i=1}^N R_i}.

PyHermes implements this idea by forming data, random, and difference fields at
the coefficient level, then evaluating the required products for the requested
geometry.

Three-Point Correlations
------------------------

For a triangle with two sides measured from a chosen center, the monopole
triplet count can be viewed as

.. math::

   DDD(R_1,R_2,\theta)
   =
   \left\langle
   n(\mathbf{x})\,
   \widetilde n_{R_1}(\mathbf{x})\,
   \widetilde n_{R_2,\theta}(\mathbf{x})
   \right\rangle,

where each :math:`\widetilde n` is a windowed field evaluated at the same
center after averaging over the requested rotations. After random
normalization, PyHermes forms the connected three-point statistic
:math:`\zeta` and the reduced statistic

.. math::

   Q =
   {\zeta\over \zeta_H},
   \qquad
   \zeta_H
   =
   \xi_{12}\xi_{13}
   +
   \xi_{12}\xi_{23}
   +
   \xi_{13}\xi_{23}.

The low-level ``Q`` reconstruction in ``corr3pcf.ipynb`` follows exactly this
dependency chain: triplet products give :math:`\zeta`, pair products give
:math:`\zeta_H`, and their ratio gives :math:`Q`.

Multipoles
----------

The multipole extension projects the angular dependence of the triangle onto
spherical-harmonic basis functions. For one leg,

.. math::

   n_{\ell m}(\mathbf{x};r)
   =
   (W_{\ell m}(r)\circ n)(\mathbf{x}),

with a Fourier-space window of the schematic form

.. math::

   \widehat W_{\ell m}(\mathbf{k};r)
   \propto
   \widehat G(k)\widehat\Phi(k)\,
   j_\ell(kr)\,
   Y_{\ell m}(\widehat{\mathbf{k}}).

Products of these filtered legs are coupled into rotationally invariant
multipole components. Truncating at ``lmax`` keeps a finite angular basis; the
``corr3pcf.ipynb`` multipole section shows how changing ``lmax`` and field
resolution changes the recovered spectrum.
