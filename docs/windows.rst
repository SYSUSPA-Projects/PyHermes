Window Functions
================

Window functions are one of the central objects in PyHermes. They define how a
point catalog is turned into a measured statistic: smoothing is a windowed
field, pair counting is a windowed field product, and multipoles are built from
angular window filters. The mathematical background page explains the shared
convolution viewpoint; this page focuses on the practical window vocabulary
used by the code and examples.

Why Windows Matter
------------------

PyHermes represents a catalog as a multiresolution field and then asks spatial
questions by convolving that field with windows:

.. math::

   n_W(\mathbf{x}) = (W \circ n)(\mathbf{x}).

Changing :math:`W` changes the statistic.

- A smoothing window such as ``sphere`` or ``gaussian`` suppresses small-scale
  structure before later measurements.
- A 2PCF ``pair_window`` selects the separation bin being counted, for example
  a real-space shell or a redshift-space ring.
- Standard 3PCF windows define the smoothed triangle legs.
- 3PCF multipoles use angular windows to build
  :math:`n_{\ell m}(\mathbf{x};r)` before coupling the filtered legs.

This is why window definitions appear repeatedly in the examples: they are not
just numerical settings, but part of the estimator definition.

Built-In Windows
----------------

PyHermes evaluates window functions in Fourier space, but each built-in
window also has a coordinate-space interpretation. The formulas below use the
Fourier convention

.. math::

   \widehat W(\mathbf{k})
   =
   \int d^3x\,W(\mathbf{x})\,
   e^{-2\pi i\mathbf{k}\cdot\mathbf{x}}.

For isotropic windows, :math:`r=|\mathbf{x}|`, :math:`k=|\mathbf{k}|`, and
:math:`q=2\pi kR`. The standard windows below are normalized so that
:math:`\int d^3x\,W(\mathbf{x})=1` and :math:`\widehat W(0)=1`.

Isotropic Windows
~~~~~~~~~~~~~~~~~

``sphere`` is a spherical top-hat of radius :math:`R`. It is the standard
choice for smoothing and count-in-cell style measurements:

.. math::

   W_{\rm sphere}(r;R)
   =
   {3\over 4\pi R^3}\,\Theta(R-r),
   \qquad
   \widehat W_{\rm sphere}(k;R)
   =
   3\,{\sin q-q\cos q\over q^3}.

``gaussian`` is a normalized 3D Gaussian with smoothing scale :math:`R`:

.. math::

   W_{\rm gaussian}(r;R)
   =
   {1\over (2\pi)^{3/2}R^3}
   \exp\left(-{r^2\over 2R^2}\right),
   \qquad
   \widehat W_{\rm gaussian}(k;R)
   =
   \exp\left(-{q^2\over 2}\right).

``shell`` is a thin spherical shell. It is the natural real-space pair window
for :math:`\xi(r)`:

.. math::

   W_{\rm shell}(r;R)
   =
   {1\over 4\pi R^2}\delta_{\rm D}(r-R),
   \qquad
   \widehat W_{\rm shell}(k;R)
   =
   {\sin q\over q}.

``Tshell`` is a finite-thickness spherical shell between
:math:`R_{\rm in}` and :math:`R_{\rm out}`. It is equivalently a
volume-normalized difference of two spherical top-hats:

.. math::

   W_{\rm Tshell}(r;R_{\rm in},R_{\rm out})
   =
   {R_{\rm out}^3 W_{\rm sphere}(r;R_{\rm out})
   -
   R_{\rm in}^3 W_{\rm sphere}(r;R_{\rm in})
   \over
   R_{\rm out}^3-R_{\rm in}^3}
   =
   {3\,[\Theta(R_{\rm out}-r)-\Theta(R_{\rm in}-r)]
   \over
   4\pi(R_{\rm out}^3-R_{\rm in}^3)}.

With :math:`q_{\rm in}=2\pi kR_{\rm in}` and
:math:`q_{\rm out}=2\pi kR_{\rm out}`,

.. math::

   \widehat W_{\rm Tshell}
   =
   {R_{\rm out}^3 \widehat W_{\rm sphere}(k;R_{\rm out})
   -
   R_{\rm in}^3 \widehat W_{\rm sphere}(k;R_{\rm in})
   \over
   R_{\rm out}^3-R_{\rm in}^3}
   =
   3\,{
   \sin q_{\rm out}
   -
   \sin q_{\rm in}
   -
   q_{\rm out}\cos q_{\rm out}
   +
   q_{\rm in}\cos q_{\rm in}
   \over
   q_{\rm out}^3-q_{\rm in}^3
   }.

``gaussian_shell`` is a Gaussian-smoothed shell-like filter. Let
:math:`a=R_{\rm shell}`, :math:`\sigma=R_{\rm smooth}`, and
:math:`G_\sigma` be the normalized Gaussian above. Define the
Gaussian-smoothed thin-shell term

.. math::

   S_{a,\sigma}(r)
   =
   {1\over (2\pi)^{3/2}\sigma^3}
   \exp\left[-{r^2+a^2\over 2\sigma^2}\right]
   {\sinh(ra/\sigma^2)\over ra/\sigma^2},

and the Gaussian-smoothed cosine-shell term

.. math::

   C_{a,\sigma}(r)
   =
   {1\over (2\pi)^{3/2}\sigma^3}
   \exp\left[-{r^2+a^2\over 2\sigma^2}\right]
   \left[
   \cosh\left({ra\over \sigma^2}\right)
   -
   {a\over r}
   \sinh\left({ra\over \sigma^2}\right)
   \right],

with the :math:`r\to0` limit used at the origin. Then

.. math::

   W_{\rm gaussian\_shell}
   =
   {\sigma^2 C_{a,\sigma}(r)+a^2 S_{a,\sigma}(r)
   \over
   a^2+\sigma^2}.

In Fourier space, with
:math:`q_{\rm shell}=2\pi kR_{\rm shell}` and
:math:`q_{\rm smooth}=2\pi kR_{\rm smooth}`,

.. math::

   \widehat W_{\rm gaussian\_shell}
   =
   {
   R_{\rm smooth}^2\cos q_{\rm shell}
   +
   R_{\rm shell}^2\,{\sin q_{\rm shell}\over q_{\rm shell}}
   \over
   R_{\rm shell}^2 + R_{\rm smooth}^2
   }
   \exp\left(-{q_{\rm smooth}^2\over 2}\right).

Line-Of-Sight Windows
~~~~~~~~~~~~~~~~~~~~~

The redshift-space windows use a line-of-sight vector
:math:`\hat{\mathbf{n}}`. PyHermes stores it as ``los_args`` with components
``nx``, ``ny``, and ``nz``. Let

.. math::

   z = \mathbf{x}\cdot\hat{\mathbf{n}},
   \qquad
   \rho = \sqrt{r^2-z^2},
   \qquad
   k_\parallel = \mathbf{k}\cdot\hat{\mathbf{n}},
   \qquad
   k_\perp = \sqrt{k^2-k_\parallel^2}.

The ``ring`` and ``disk`` kernels use a cosine factor in Fourier space, so
their coordinate-space definitions below are symmetrized over
:math:`z=\pm H`; the factor
:math:`\delta_{\rm D}(|z|-H)/2` keeps the line-of-sight part normalized.

``ring`` is a thin transverse ring at radius :math:`R` and line-of-sight
offset :math:`H`:

.. math::

   W_{\rm ring}(\rho,z;R,H)
   =
   {\delta_{\rm D}(\rho-R)\over 2\pi R}
   {\delta_{\rm D}(|z|-H)\over 2},
   \qquad
   \widehat W_{\rm ring}
   =
   J_0(2\pi k_\perp R)\,
   \cos(2\pi k_\parallel H).

``disk`` is a transverse disk of radius :math:`R` at line-of-sight offset
:math:`H`:

.. math::

   W_{\rm disk}(\rho,z;R,H)
   =
   {\Theta(R-\rho)\over \pi R^2}
   {\delta_{\rm D}(|z|-H)\over 2},
   \qquad
   \widehat W_{\rm disk}
   =
   {2J_1(2\pi k_\perp R)\over 2\pi k_\perp R}\,
   \cos(2\pi k_\parallel H).

``cylshell`` is a cylindrical side surface at radius :math:`R` and half-height
:math:`H`:

.. math::

   W_{\rm cylshell}(\rho,z;R,H)
   =
   {\delta_{\rm D}(\rho-R)\over 2\pi R}
   {\Theta(H-|z|)\over 2H},
   \qquad
   \widehat W_{\rm cylshell}
   =
   J_0(2\pi k_\perp R)\,
   {\sin(2\pi k_\parallel H)\over 2\pi k_\parallel H}.

``cylinder`` is a cylindrical top-hat of radius :math:`R` and half-height
:math:`H`:

.. math::

   W_{\rm cylinder}(\rho,z;R,H)
   =
   {\Theta(R-\rho)\over \pi R^2}
   {\Theta(H-|z|)\over 2H},
   \qquad
   \widehat W_{\rm cylinder}
   =
   {2J_1(2\pi k_\perp R)\over 2\pi k_\perp R}\,
   {\sin(2\pi k_\parallel H)\over 2\pi k_\parallel H}.

All sinc-like fractions in this section are evaluated with their limiting
value of one when the denominator is zero.

Defining Windows In PyHermes
----------------------------

A window definition is a small dictionary. In YAML:

.. code-block:: yaml

   window:
     type: "sphere"
     len_args:
       R: 5

In Python:

.. code-block:: python

   from pyhermes.io import WindowFunc

   window = WindowFunc(
       {"type": "sphere", "len_args": {"R": 5}},
       convols_data.convols_info,
       threads=8,
   )
   smoothed = convols_data @ window

Important details:

- ``type`` selects the built-in window function.
- ``len_args`` contains physical lengths in the same units as ``box_size``.
  PyHermes rescales them internally to the multiresolution grid.
- ``WindowFunc`` must be built with the ``convols_info`` of the field it will
  convolve. This keeps ``J``, ``box_size``, ``phi_resolution``, and the wavelet
  basis consistent.
- Line-of-sight windows use ``los_args``. The default LOS is the z axis:
  ``[0, 0, 1]``.
- ``kernel_mode`` controls kernel construction. ``octant`` is fastest but only
  valid when the Fourier-space window has the required sign-flip symmetries.
  ``full_rfft`` is more general. ``auto`` uses the fast path for axis-aligned
  LOS windows and the general path for oblique LOS windows.

There is no universal default ``WindowFunc`` object. Task-level parameters may
choose to apply no additional smoothing, but an explicit ``WindowFunc`` needs a
valid ``type`` and matching length arguments.

Custom Window Functions
-----------------------

Advanced users can pass a custom ``func`` instead of choosing a built-in
``type``. A custom window function should describe the Fourier-space kernel
:math:`\widehat W(\mathbf{k})`, not the real-space profile. PyHermes evaluates
that kernel on the wavelet/Fourier grid and folds it into the convolution
kernel used by ``ConvolsData @ WindowFunc``.

Custom windows should be written as Numba ``@njit`` functions. The required
signature pattern is:

.. code-block:: python

   from numba import njit

   @njit
   def my_window(ki, kj, kk, ...):
       ...

The first three arguments are the Fourier-grid coordinates. Any remaining
arguments are matched by name against the window dictionary:

- put physical length scales in ``len_args`` so PyHermes rescales them from
  box units to grid units;
- put line-of-sight components in ``los_args`` using ``nx``, ``ny``, and
  ``nz``;
- put dimensionless controls or other non-length parameters in ``other_args``.

For example:

.. code-block:: python

   import numpy as np
   from numba import njit

   from pyhermes.io import WindowFunc

   @njit
   def custom_gaussian(ki, kj, kk, R, amplitude=1.0):
       k = np.sqrt(ki * ki + kj * kj + kk * kk)
       q = 2.0 * np.pi * k * R
       return amplitude * np.exp(-0.5 * q * q)

   window = WindowFunc(
       {
           "type": "custom_gaussian",
           "func": custom_gaussian,
           "len_args": {"R": 5},
           "other_args": {"amplitude": 1.0},
       },
       convols_data.convols_info,
       threads=8,
   )

Practical cautions:

- Use the ``@njit`` style shown above. The kernel is evaluated inside Numba
  loops, so a plain Python function will either fail during compilation or make
  kernel construction impractically slow.
- Handle the zero mode explicitly when the formula contains divisions by
  :math:`k`, :math:`q`, or Bessel-like factors. Most normalized smoothing or
  counting windows should return ``1`` at ``k = 0``.
- Return a real scalar. ``WindowFunc`` is designed for real convolution
  kernels.
- Be careful with parameter placement. If a length scale is accidentally put in
  ``other_args``, PyHermes will not rescale it by ``J`` and ``box_size``.
- Custom windows default to the conservative ``full_rfft`` kernel mode. Only
  request ``octant`` if the Fourier kernel is invariant under independent sign
  flips of all grid axes:
  ``W(kx, ky, kz) = W(-kx, ky, kz) = W(kx, -ky, kz) = W(kx, ky, -kz)``.
- Custom functions are Python objects, so they are meant for Python-level task
  construction. YAML configs can describe built-in windows, but cannot store a
  live Python function.

Windows In 2PCF
---------------

``Corr_2PCF`` uses windows in two different places.

Field windows are optional preprocessing windows applied to the input fields
before pair counting:

.. code-block:: yaml

   Corr_2PCF:
     convols_data: "./output/quijote8000_snap004_rsd_sfc.pkl"
     random: "uniform"
     window:
       type: "sphere"
       len_args:
         R: 5

This turns the input density into a smoothed field before the 2PCF products are
formed. ``window1`` and ``window2`` can be used when the two legs need
different preprocessing.

``pair_window`` defines the separation bin itself. For a real-space
:math:`\xi(s)` measurement, the default is a shell:

.. code-block:: yaml

   pair_window: "shell"
   sampling:
     s:
       min: 0.0
       max: 180.0
       n: 46

For redshift-space coordinates, use ``ring``, ``disk``, ``cylinder``, or
``cylshell``. The ``mapping`` field tells PyHermes how to convert sampling
coordinates into window lengths:

.. code-block:: yaml

   pair_window:
     type: "ring"
     mapping: "smu_to_RH"
     los_args: [0, 0, 1]
   sampling:
     s:
       min: 0.0
       max: 180.0
       n: 46
     mu:
       min: 0.0
       max: 1.0
       n: 51

The built-in mappings are:

- ``s_to_R``: set ``R = s`` for shell-like real-space measurements.
- ``smu_to_RH``: set ``R = s sqrt(1 - mu^2)`` and ``H = s mu``.
- ``rppi_to_RH``: set ``R = rp`` and ``H = pi``.

For Python-level workflows, ``mapping`` can also be a custom callable. This is
useful when the sampling coordinates do not map onto ``R`` and ``H`` in one of
the built-in ways. The callable receives the current sampling point and the
template ``pair_window`` dictionary, then returns the concrete window
dictionary for that sample:

.. code-block:: python

   import copy
   import numpy as np

   def custom_mapping(sample, pair_window):
       params = copy.deepcopy(pair_window)
       params.setdefault("len_args", {})
       params["len_args"]["R"] = sample["s"] * np.sqrt(1.0 - sample["mu"] ** 2)
       params["len_args"]["H"] = sample["s"] * sample["mu"]
       params["len_args"]["R"] += 2.0  # Example offset for a custom bin rule.
       return params

   task.run(
       pair_window={
           "type": "ring",
           "mapping": custom_mapping,
           "los_args": [0, 0, 1],
       }
   )

The returned dictionary should contain a valid window ``type`` and concrete
``len_args`` for that sample. PyHermes removes ``mapping`` before constructing
the ``WindowFunc``. As with custom window functions, callable mappings are
Python objects, so they are intended for Python API usage rather than plain
YAML configs.

When the LOS is axis-aligned, ``auto`` can use the faster symmetry-folded
kernel. For diagonal or oblique LOS choices, PyHermes uses the more general
full real-FFT kernel, which is slower but mathematically safe.

Windows In 3PCF
---------------

Standard 3PCF uses ``window1``, ``window2``, and ``window3`` to define the
three triangle legs. If a shared ``window`` is supplied, it acts as the
fallback for legs without a leg-specific window.

The center mode determines whether ``window1`` can act on the first leg:

- ``center: "particle"`` uses catalog objects as the centers. ``window1`` has
  no effect because leg 1 is sampled as discrete particle centers.
- ``center: "box_random"`` samples Monte Carlo positions in the box. All three
  legs are evaluated as convolved fields, so ``window1`` can smooth the center
  leg as well.

This distinction is also discussed in :doc:`get_start/corr_3pcf/corr_3pcf` and
in the 3PCF section of :doc:`math`.

Windows In 3PCF Multipoles
--------------------------

``Corr_3PCF_Multipole`` has two window layers.

First, the user-provided ``window`` or ``window1``/``window2``/``window3``
smooths the input legs, just as in standard 3PCF:

.. code-block:: yaml

   Corr_3PCF_Multipole:
     convols_data: "./output/quijote8000_snap004_sfc.pkl"
     random: "uniform"
     window:
       type: "sphere"
       len_args:
         R: 5
     r12: 20.0
     r13: 40.0
     l_max: 10

Second, the multipole calculation constructs angular filters internally. For
one leg, the filtered field has the schematic form

.. math::

   n_{\ell m}(\mathbf{x};r)
   =
   (W_{\ell m}(r)\circ n)(\mathbf{x}),

where :math:`W_{\ell m}` contains the radial scale and spherical harmonic
angular dependence. The user controls this layer mainly through ``r12``,
``r13``, and ``l_max`` rather than by writing a separate YAML window for each
``(\ell,m)`` mode.

Practical Rules Of Thumb
------------------------

- Use ``sphere`` for smoothing before Counting, 2PCF, or 3PCF measurements.
- Use ``shell`` or ``Tshell`` for real-space pair separations.
- Use ``ring`` for thin redshift-space pair bins, ``cylshell`` for cylindrical
  side surfaces, and ``disk`` or ``cylinder`` when a finite transverse or
  line-of-sight average is desired.
- Keep the LOS axis-aligned when possible if runtime matters. Oblique LOS is
  supported, but it uses a more general kernel path.
- Use ``pair_window_cache`` for heavy 2PCF runs when many pair windows are
  rebuilt under a memory-saving strategy.
- Treat window choices as part of the science definition of the statistic, not
  only as performance parameters.
