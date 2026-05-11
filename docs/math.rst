Mathematical Background
=======================

This page summarizes the Hermes formulation behind PyHermes. The same
mathematical object appears throughout the package: a weighted point catalog is
turned into a continuous multiresolution field, window functions are applied as
convolutions, and statistics are read out as field averages or sampled products.
For a task-oriented reference to the built-in windows and their YAML/Python
definitions, see :doc:`windows`.

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
ring-like pair geometries. The practical window families are summarized in
:doc:`windows`.

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
triplet count combines one center leg with two displaced legs. PyHermes
supports two center choices, and the distinction matters because it determines
whether the first leg is a discrete catalog sample or a convolved field.

Particle Centers
~~~~~~~~~~~~~~~~

With ``center="particle"``, the first leg is the input catalog itself. A
schematic triplet product is

.. math::

   DDD_{\rm pcenter}(R_2,R_3,\theta)
   =
   {1\over W_1}
   \sum_{i\in D_1} w_i\,
   \widetilde n_{2,R_2}(\mathbf{x}_i)\,
   \widetilde n_{3,R_3,\theta}(\mathbf{x}_i),
   \qquad
   W_1=\sum_{i\in D_1}w_i.

The center positions :math:`\mathbf{x}_i` are real particles, halos, or
galaxies. The second and third legs are windowed fields evaluated at those
positions after averaging over the requested rotations. Since the first leg is
not evaluated as a continuous field, this mode cannot apply a window
convolution to leg 1. It is therefore best suited to sparse tracer catalogs,
for example halo or galaxy samples with up to roughly a million objects.

Box-Random Centers
~~~~~~~~~~~~~~~~~~

With ``center="box_random"``, PyHermes draws Monte Carlo centers
:math:`\{\mathbf{y}_a\}` uniformly in the periodic box and evaluates all three
legs as fields at those centers:

.. math::

   DDD_{\rm rcenter}(R_1,R_2,R_3,\theta)
   \simeq
   {1\over N_{\rm c}}
   \sum_{a=1}^{N_{\rm c}}
   \widetilde n_{1,R_1}(\mathbf{y}_a)\,
   \widetilde n_{2,R_2}(\mathbf{y}_a)\,
   \widetilde n_{3,R_3,\theta}(\mathbf{y}_a).

Equivalently, this is a Monte Carlo estimate of the volume average of the
product of three convolved fields. Because leg 1 is also a field in this mode,
``window1`` can be applied to the center leg. This is the natural mode for very
dense simulation catalogs, such as dark matter particle samples with
ten-million-scale particle counts, where using every particle as a center would
be unnecessarily expensive.

Reduced Statistic
~~~~~~~~~~~~~~~~~

After random normalization, both center modes feed the same connected
three-point statistic :math:`\zeta` and reduced statistic

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
