Window Functions
================

Window functions are one of the central objects in PyHermes. They define how a
point catalog is turned into a measured statistic: smoothing is a windowed
field, pair counting is a windowed field product, and multipoles are built from
angular window filters. The mathematical background page explains the shared
convolution viewpoint; this page focuses on the practical window vocabulary
used by the code and examples.

For the notebook-style walkthrough of ``SFCField`` and ``WindowFunc``
arithmetic, see :doc:`get_start/window/window`.

Why Windows Matter
------------------

PyHermes represents a catalog as a multiresolution field and then asks spatial
questions by convolving that field with windows:

.. math::

   n_W(\mathbf{x}) = (W \circ n)(\mathbf{x}).

Changing :math:`W` changes the statistic.

- A smoothing window such as ``sphere`` or ``gaussian`` suppresses small-scale
  structure before later measurements.
- A 2PCF ``binning_window`` selects the separation bin being counted, for example
  a real-space shell or a redshift-space ring.
- Standard 3PCF windows define the smoothed triangle legs.
- 3PCF multipoles use angular windows to build
  :math:`n_{\ell m}(\mathbf{x};r)` before coupling the filtered legs.

This is why window definitions appear repeatedly in the examples: they are not
just numerical settings, but part of the estimator definition.

The same ``WindowFunc`` machinery appears in several roles:

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Role
     - Typical examples
     - Where it appears
   * - Field smoothing or filtering
     - ``sphere``, ``gaussian``, ``gaussian_shell``, ``cubic``
     - ``window``, ``window1``, ``window2``, ``window3`` in Counting, 2PCF,
       and 3PCF tasks
   * - High-pass or band-pass filtering
     - ``cw``, ``cws``, ``gdw``
     - scale-localized power and wavelet-like filtered fields
   * - Pair geometry
     - ``shell``, ``ring``, ``disk``, ``cylinder``, ``cylshell``
     - ``binning_window`` in 2PCF tasks
   * - Angular multipole filters
     - ``legendre_multipole`` and internal spherical-harmonic filters
     - 3PCF multipoles, with 2PCF multipoles planned as a future extension
   * - Field operators
     - ``directional_derivative``, ``laplacian``, ``inverse_laplacian``
     - gradients, divergence, curl, and Poisson potential fields

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

The standard windows below are normalized so that
:math:`\int d^3x\,W(\mathbf{x})=1` and :math:`\widehat W(0)=1`. For isotropic
windows, :math:`r=|\mathbf{x}|`, :math:`k=|\mathbf{k}|`, and
:math:`q=2\pi kR`.

The current built-in window vocabulary contains fifteen named windows:
``shell``, ``sphere``, ``gaussian``, ``gaussian_shell``, ``cubic``,
``cylinder``, ``cylshell``, ``disk``, ``ring``, ``cw``, ``cws``, ``gdw``,
``directional_derivative``, ``laplacian``, and ``inverse_laplacian``. The
first nine are ordinary low-pass or pair-geometry windows,
``cw``/``cws``/``gdw`` are high-pass or wavelet-like filters, and the last
three are field-operator windows.

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

``shell`` is a thin spherical shell. It is the natural real-space binning window
for :math:`\xi(r)`:

.. math::

   W_{\rm shell}(r;R)
   =
   {1\over 4\pi R^2}\delta_{\rm D}(r-R),
   \qquad
   \widehat W_{\rm shell}(k;R)
   =
   {\sin q\over q}.

``gaussian_shell`` is a Gaussian-smoothed shell-like filter. Let
:math:`R_s` be the shell scale, :math:`R_g` be the Gaussian smoothing scale, and
:math:`G_\sigma` be the normalized Gaussian above. Define the
Gaussian-smoothed thin-shell term

.. math::

   S_{R_s,R_g}(r)
   =
   {1\over (2\pi)^{3/2}R_g^3}
   \exp\left[-{r^2+R_s^2\over 2R_g^2}\right]
   {\sinh(rR_s/R_g^2)\over rR_s/R_g^2},

and the Gaussian-smoothed cosine-shell term

.. math::

   C_{R_s,R_g}(r)
   =
   {1\over (2\pi)^{3/2}R_g^3}
   \exp\left[-{r^2+R_s^2\over 2R_g^2}\right]
   \left[
   \cosh\left({rR_s\over R_g^2}\right)
   -
   {R_s\over r}
   \sinh\left({rR_s\over R_g^2}\right)
   \right],

with the :math:`r\to0` limit used at the origin. Then

.. math::

   W_{\rm gaussian\_shell}
   =
   {R_g^2 C_{R_s,R_g}(r)+R_s^2 S_{R_s,R_g}(r)
   \over
   R_s^2+R_g^2}.

In Fourier space, with
:math:`q_s=2\pi kR_s` and :math:`q_g=2\pi kR_g`,

.. math::

   \widehat W_{\rm gaussian\_shell}
   =
   {
   R_g^2\cos q_s
   +
   R_s^2\,{\sin q_s\over q_s}
   \over
   R_s^2 + R_g^2
   }
   \exp\left(-{q_g^2\over 2}\right).

In code, these two length arguments are named ``R_shell`` and ``R_smooth`` for
backward compatibility with existing configuration files; they correspond to
:math:`R_s` and :math:`R_g` in the formulas above.

Axis-Aligned Windows
~~~~~~~~~~~~~~~~~~~~

``cubic`` is an axis-aligned rectangular top-hat with side lengths
:math:`L_x`, :math:`L_y`, and :math:`L_z`; it is a cube when all three lengths
are equal:

.. math::

   \begin{aligned}
   W_{\rm cubic}(\mathbf{x};L_x,L_y,L_z)
   &=
   \prod_{i=x,y,z}{1\over L_i}
   \Theta\left({L_i\over2}-|x_i|\right),\\
   \widehat W_{\rm cubic}(\mathbf{k};L_x,L_y,L_z)
   &=
   \prod_{i=x,y,z}
   {\sin(\pi k_i L_i)\over \pi k_i L_i}.
   \end{aligned}

The factor of :math:`\pi` comes from PyHermes' Fourier convention
:math:`e^{-2\pi i\mathbf{k}\cdot\mathbf{x}}`; the expression often appears as
:math:`\sin(k_i L_i/2)/(k_i L_i/2)` when :math:`k_i` denotes angular
wavenumber. Each sinc factor uses its limiting value of one when the
denominator is zero.

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

All line-of-sight windows below are even in :math:`z`, so their Fourier
kernels are real. The ``ring`` and ``disk`` windows use two infinitesimally
thin offsets at :math:`z=\pm H`, written compactly as
:math:`\delta_{\rm D}(|z|-H)/2`; this gives the cosine factor in Fourier
space. The ``cylshell`` and ``cylinder`` windows use the symmetric finite
interval :math:`|z|\le H`, written as :math:`\Theta(H-|z|)/(2H)`, which gives
the sinc factor. These prefactors keep the line-of-sight part normalized.

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

High-Pass And Wavelet-Like Filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``cw``, ``cws``, and ``gdw`` windows suppress the zero mode and respond to
a finite band of wavenumbers. They are useful when a statistic should measure
scale-localized fluctuation power rather than low-pass smoothed density.

For the cosine-wavelet windows define

.. math::

   G_{\rm CW}(q)
   =
   q\,[q\cosh q-\sinh q]\,\exp(-q^2/2).

``cw`` implements the one-dimensional cosine-wavelet response as an isotropic
radial kernel in :math:`|\mathbf{k}|`:

.. math::

   \widehat W_{\rm cw}(k;R)
   =
   (2\pi)^{1/2}C_{\rm CW}R^{1/2}G_{\rm CW}(q),
   \qquad
   C_{\rm CW}
   =
   {2\sqrt{2}\over \sqrt{1+5e}\,\pi^{1/4}}.

``cws`` is the corresponding three-dimensional spherical cosine wavelet:

.. math::

   \widehat W_{\rm cws}(k;R)
   =
   (2\pi)^{3/2}C_{\rm CWS}R^{3/2}G_{\rm CW}(q),
   \qquad
   C_{\rm CWS}
   =
   {2\sqrt{2}\over \sqrt{9+55e}\,\pi^{3/4}}.

``gdw`` is the Gaussian-derivative wavelet:

.. math::

   \widehat W_{\rm gdw}(k;R)
   =
   {2^{5/2}\pi^{3/4}\over \sqrt{15}}\,R^{3/2}
   q^2\exp(-q^2/2).

Field-Derivative Windows
~~~~~~~~~~~~~~~~~~~~~~~~

The standard windows above mostly average or select parts of a field. PyHermes
also provides field-derivative operator windows, which apply differential
operators to the field represented by a ``SFCField`` object. This works
because derivatives have a simple Fourier-space form. In a periodic box, write

.. math::

   f(\mathbf{x})
   =
   \sum_{\mathbf{k}}\widetilde f(\mathbf{k})
   e^{2\pi i\mathbf{k}\cdot\mathbf{x}}.

Differentiating term by term gives

.. math::

   \widetilde{\partial_x f}(\mathbf{k})
   =
   2\pi i k_x\,\widetilde f(\mathbf{k}).

More generally, a directional derivative along
:math:`\widehat{\mathbf n}` is multiplication by
:math:`2\pi i(\mathbf{k}\cdot\widehat{\mathbf n})`. This means a derivative
can be represented as a Fourier-space window:

.. math::

   \widehat W_{\partial_{\widehat n}}(\mathbf{k})
   =
   2\pi i\,(\mathbf{k}\cdot\widehat{\mathbf n}).

In code this field-derivative window is ``directional_derivative``. The
derivative direction :math:`\widehat{\mathbf n}` is passed through
``los_args``; no ``len_args`` are needed. This window is not a normalized
smoothing window: its zero mode is zero, so it removes constant backgrounds.
It also uses the ``complex_rfft`` kernel mode because the Fourier-space
derivative multiplier is imaginary and odd.

Applied to a scalar field, the result is the directional derivative of the
PyHermes field, :math:`\partial_{\widehat n}f`, in grid-coordinate units. To
convert the derivative to physical box units, multiply the output by
:math:`L/L_{\rm box}`; this factor is available as
``SFCField.scale_factor``. To keep derivatives stable on discrete tracer
fields, combine this operator with an ordinary smoothing window such as
``gaussian``. In PyHermes this can be done with the same convolution syntax,
for example ``D @ W_G @ W_deriv`` for a ``SFCField`` object ``D``, a
Gaussian window ``W_G``, and a derivative window ``W_deriv``:

.. math::

   \widehat W_{\partial_{\widehat n}G}(\mathbf{k};R)
   =
   \widehat W_{\rm gaussian}(\mathbf{k};R)
   \widehat W_{\partial_{\widehat n}}(\mathbf{k})
   =
   2\pi i\,(\mathbf{k}\cdot\widehat{\mathbf n})
   \exp\left[-{(2\pi kR)^2\over 2}\right].

PyHermes also provides the field-derivative window ``laplacian`` for the
scalar operator
:math:`\nabla^2`:

.. math::

   \widehat W_{\nabla^2}(\mathbf{k})
   =
   -(2\pi)^2|\mathbf{k}|^2.

The companion ``inverse_laplacian`` window represents
:math:`\nabla^{-2}`:

.. math::

   \widehat W_{\nabla^{-2}}(\mathbf{k})
   =
   -{1\over (2\pi)^2|\mathbf{k}|^2},
   \qquad
   \widehat W_{\nabla^{-2}}(\mathbf{0})=0.

The zero mode is set to zero because the mean of a Poisson potential is an
arbitrary convention. This window is intentionally a pure mathematical
operator; multiply the result externally by the physical comoving Poisson
prefactor and the grid-to-box coordinate factor when constructing
:math:`\Phi/c^2`:

.. math::

   {\Phi\over c^2}
   =
   {3\Omega_m\over 2a}
   \left({H_0\over c}\right)^2
   \left({L_{\rm box}\over L}\right)^2
   \nabla_{\rm grid}^{-2}\delta_m.

When ``box_size`` is in :math:`{\rm Mpc}/h`, use ``H0=100`` in
``km/s/(Mpc/h)`` for the external prefactor.

The ``gdw`` window is a scale-normalized Gaussian-derivative wavelet. It is
related to combining a Gaussian with the negative Laplacian, but includes its
own scale and :math:`L^2` normalization factors.

This construction is natural for ``SFCField`` because PyHermes stores a
periodic scaling-function representation of the particle field and applies
``WindowFunc`` objects through Fourier-space convolution. The derivative is
therefore the derivative of the PyHermes-represented field, optionally after
whatever smoothing windows you compose with the derivative operator; it is not
an unsmoothed derivative of the original delta-function particle catalog.

For a worked example using these windows to build velocity and
momentum-density divergence and curl fields, see
:doc:`get_start/weighted_fields/weighted_fields`.

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
       sfc_field.sfc_info,
       threads=8,
   )
   smoothed = sfc_field @ window

``WindowFunc`` objects can also be combined after their kernels are built. The
operations act directly on ``w_kernel``:

.. code-block:: python

   W_shell = WindowFunc(shell_params, sfc_field.sfc_info, threads=8)
   W_disk = WindowFunc(disk_params, sfc_field.sfc_info, threads=8)

   W_mix = 0.7 * W_shell + 0.3 * W_disk
   W_chain = W_shell * W_disk
   smoothed = sfc_field @ W_mix

Supported operations are ``W1 + W2``, ``W1 - W2``, ``W1 * W2``, ``a * W``,
``W * a``, ``W / a``, and ``-W``. Both windows in a binary operation must share the same
``J``, ``box_size``, ``phi_resolution``, ``wavelet_mode``, ``wavelet_level``,
``bandwidth``, and kernel shape. Composite windows are materialized: they store
the resulting ``w_kernel`` and do not rebuild from a window function.
The product ``W1 * W2`` is a projected-kernel product,
``W1.as_array() * W2.as_array()``. It is useful for matching a chained
projected convolution, ``(field @ W1) @ W2``, up to FFT roundoff; it is not a
raw analytic transfer product of the form
``P_phi[W_hat_1 * W_hat_2]``.

Important details:

- ``type`` selects the built-in window function.
- ``len_args`` contains physical lengths in the same units as ``box_size``.
  PyHermes rescales them internally to the multiresolution grid.
- ``WindowFunc`` must be built with the ``sfc_info`` of the field it will
  convolve. This keeps ``J``, ``box_size``, ``phi_resolution``, and the wavelet
  basis consistent.
- Line-of-sight windows use ``los_args``. The default LOS is the z axis:
  ``[0, 0, 1]``.
- ``kernel_mode`` controls kernel construction. ``octant`` is fastest but only
  valid when the Fourier-space window has the required sign-flip symmetries.
  ``full_rfft`` is more general. ``auto`` uses the fast path for axis-aligned
  LOS windows and the general path for oblique LOS windows. ``complex_rfft``
  supports complex Hermitian-preserving kernels such as directional derivative
  filters. ``complex_full_fft`` is reserved for full complex kernels such as
  the built-in 3PCF multipole filters.

There is no universal default ``WindowFunc`` object. Task-level parameters may
choose to apply no additional smoothing, but an explicit ``WindowFunc`` needs a
valid ``type`` and matching length arguments.

Custom Window Functions
-----------------------

Advanced users can pass a custom ``func`` instead of choosing a built-in
``type``. A custom window function should describe the Fourier-space kernel
:math:`\widehat W(\mathbf{k})`, not the real-space profile. PyHermes evaluates
that kernel on the wavelet/Fourier grid and folds it into the convolution
kernel used by ``SFCField @ WindowFunc``.

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
       sfc_field.sfc_info,
       threads=8,
   )

Custom windows can also compose existing windows. For example, a
finite-thickness spherical shell can be written as a volume-normalized
difference of two spherical top-hats:

.. math::

   \widehat W_{\rm thick\ shell}
   =
   {R_{\rm out}^3 \widehat W_{\rm sphere}(k;R_{\rm out})
   -
   R_{\rm in}^3 \widehat W_{\rm sphere}(k;R_{\rm in})
   \over
   R_{\rm out}^3-R_{\rm in}^3}.

In Python:

.. code-block:: python

   import copy
   import numpy as np
   from numba import njit

   from pyhermes.utils.window_functions import window_function_sphere_numba

   @njit
   def window_function_thick_shell_numba(ki, kj, kk, R_in, R_out):
       V_in = R_in**3
       V_out = R_out**3
       denom = V_out - V_in
       if denom == 0.0:
           k = np.sqrt(ki**2 + kj**2 + kk**2)
           q_out = 2.0 * np.pi * k * R_out
           if q_out == 0.0:
               return 1.0
           return np.sin(q_out) / q_out

       W_in = window_function_sphere_numba(ki, kj, kk, R_in)
       W_out = window_function_sphere_numba(ki, kj, kk, R_out)
       return (W_out * V_out - W_in * V_in) / denom

   def mapping_s_to_R_thick_shell(sample, binning_window, T=10.0):
       params = copy.deepcopy(binning_window)
       params["len_args"]["R_in"] = sample["s"] - T / 2.0
       params["len_args"]["R_out"] = sample["s"] + T / 2.0
       return params

   binning_window = {
       "type": "thick_shell",
       "func": window_function_thick_shell_numba,
       "len_args": ["R_in", "R_out"],
       "mapping": mapping_s_to_R_thick_shell,
       "kernel_mode": "octant",
   }

The same idea works for line-of-sight windows. This example combines the
cylindrical side surface and two disk caps into a closed cylinder surface. The
weights follow the corresponding areas: the side area is proportional to
:math:`2H`, while the two caps are proportional to :math:`R` after the common
factor :math:`2\pi R` is removed:

.. code-block:: python

   from numba import njit

   from pyhermes.utils.window_functions import (
       window_function_cylshell_numba,
       window_function_disk_numba,
   )

   @njit
   def window_function_cylsurf_numba(
       ki, kj, kk, R, H, nx=0.0, ny=0.0, nz=1.0
   ):
       denom = 2.0 * H + R
       if denom == 0.0:
           return 1.0
       win_cylshell = window_function_cylshell_numba(ki, kj, kk, R, H, nx, ny, nz)
       win_disk = window_function_disk_numba(ki, kj, kk, R, H, nx, ny, nz)
       return (win_cylshell * 2.0 * H + win_disk * R) / denom

   binning_window = {
       "type": "cylsurf",
       "func": window_function_cylsurf_numba,
       "len_args": ["R", "H"],
       "los_args": {"nx": 0.0, "ny": 0.0, "nz": 1.0},
       "mapping": "smu_to_RH",
       "kernel_mode": "auto",
   }

Practical cautions:

- Use the ``@njit`` style shown above. The kernel is evaluated inside Numba
  loops, so a plain Python function will either fail during compilation or make
  kernel construction impractically slow.
- Handle the zero mode explicitly when the formula contains divisions by
  :math:`k`, :math:`q`, or Bessel-like factors. Most normalized smoothing or
  counting windows should return ``1`` at ``k = 0``.
- Return a real scalar for ordinary smoothing and binning windows. Complex
  Hermitian-preserving kernels are supported through ``kernel_mode:
  "complex_rfft"``; this is the mode used by the built-in
  ``directional_derivative`` window.
- Be careful with parameter placement. If a length scale is accidentally put in
  ``other_args``, PyHermes will not rescale it by ``J`` and ``box_size``.
- When adding or subtracting windows, combine normalized kernels with the
  correct volume, area, or line-of-sight weights if the result should remain
  normalized. For binning windows this usually means checking that
  :math:`\widehat W(0)=1`.
- Handle degenerate limits explicitly, such as ``R_in == R_out`` in a
  finite-thickness shell or ``q == 0`` in a sinc/Bessel factor. These limits are
  often the difference between a stable custom window and a noisy one.
- Keep physical length arguments in their valid domain. For the thick-shell
  example above, choose the sampling range and thickness so that ``R_in`` is
  non-negative, or clip it deliberately in the mapping if that is the desired
  bin definition.
- Custom mappings should fill only the length parameters that are meant to vary
  with the current sample. Fixed numeric values in ``len_args`` are left as
  user-defined constants.
- Custom windows default to the conservative ``full_rfft`` kernel mode. Only
  request ``octant`` if the Fourier kernel is invariant under independent sign
  flips of all grid axes:
  ``W(kx, ky, kz) = W(-kx, ky, kz) = W(kx, -ky, kz) = W(kx, ky, -kz)``.
- For oblique LOS choices, use ``full_rfft``. Axis-aligned even windows can use
  ``octant`` or ``auto``.
- For complex rFFT windows, request ``complex_rfft`` unless you are using a
  built-in window that selects that mode automatically.
- Full complex FFT kernels are used by specialized built-in windows such as
  ``legendre_multipole``. They are useful for spherical-harmonic multipole
  filters, but require an ``(L,L,L)`` complex kernel rather than the usual
  rFFT-shaped kernel.
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
     sfc_field: "./output/quijote8000_snap004_rsd_sfc.pkl"
     random: "uniform"
     window:
       type: "sphere"
       len_args:
         R: 5

This turns the input density into a smoothed field before the 2PCF products are
formed. ``window1`` and ``window2`` can be used when the two legs need
different preprocessing.

``binning_window`` defines the separation bin itself. For a real-space
:math:`\xi(s)` measurement, the default is a shell:

.. code-block:: yaml

   binning_window: "shell"
   sampling:
     s:
       min: 0.0
       max: 180.0
       n: 46

For redshift-space coordinates, use ``ring``, ``disk``, ``cylinder``, or
``cylshell``. The ``mapping`` field tells PyHermes how to convert sampling
coordinates into window lengths:

.. code-block:: yaml

   binning_window:
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
template ``binning_window`` dictionary, then returns the concrete window
dictionary for that sample:

.. code-block:: python

   import copy
   import numpy as np

   def custom_mapping(sample, binning_window):
       params = copy.deepcopy(binning_window)
       params.setdefault("len_args", {})
       params["len_args"]["R"] = sample["s"] * np.sqrt(1.0 - sample["mu"] ** 2)
       params["len_args"]["H"] = sample["s"] * sample["mu"]
       params["len_args"]["R"] += 2.0  # Example offset for a custom bin rule.
       return params

   task.run(
       binning_window={
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

Windows In 2PCF Multipoles
--------------------------

This section describes a planned extension rather than a released
``Corr_2PCF`` interface. 2PCF multipoles are a natural future use of the same
binning-window language. In the traditional workflow, one first measures
:math:`\xi(s,\mu)` on a two-dimensional grid and then projects along
:math:`\mu`:

.. math::

   \xi_\ell(s)
   =
   {2\ell+1\over 2}
   \int_{-1}^{1}d\mu\,
   \xi(s,\mu)\,\mathcal{L}_\ell(\mu).

That approach makes the result depend on the chosen :math:`(s,\mu)` sampling:
too few :math:`\mu` bins under-resolve the projection, while many narrow bins
increase cost and make each intermediate bin noisier.

In the PyHermes window-convolution picture, the projection can instead be
absorbed into the binning window itself. For a fixed plane-parallel line of sight
:math:`\widehat{\mathbf n}`, the real-space pair filter can be written
schematically as

.. math::

   W_\ell(\mathbf{r};s,\widehat{\mathbf n})
   =
   {1\over 4\pi s^2}
   \delta_{\rm D}(|\mathbf{r}|-s)\,
   \mathcal{L}_\ell(\widehat{\mathbf r}\cdot\widehat{\mathbf n}).

Then the multipole can be estimated directly as a windowed-field product,

.. math::

   \xi_\ell(s)
   \propto
   \left\langle
   \delta(\mathbf{x})\,
   (W_\ell\circ\delta)(\mathbf{x})
   \right\rangle,

with the conventional :math:`2\ell+1` normalization applied either in the
window or in the final estimator. The :math:`\ell=0` case reduces to the
ordinary shell binning window used by :math:`\xi(s)`, while higher multipoles use
Legendre-weighted shell windows.

This planned route is one of the advantages of the Hermes framework: it can
skip the coordinate-space :math:`(s,\mu)` sampling step and perform the angular
projection through a Fourier-space convolution. The resulting accuracy is then
controlled mainly by the field resolution ``J`` and by any smoothing/window
choice, rather than by an auxiliary :math:`\mu` grid. As with the current
redshift-space binning windows, this simple convolution form assumes a fixed LOS;
local-LOS survey estimators would require a separate treatment because the
window would no longer be purely translation invariant.

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
     sfc_field: "./output/quijote8000_snap004_sfc.pkl"
     random: "uniform"
     window:
       type: "sphere"
       len_args:
         R: 5
     r12: 20.0
     r13: 40.0
     l_max: 10

Second, the multipole calculation constructs angular filters internally as
``WindowFunc`` objects with ``type: legendre_multipole`` and
``kernel_mode: complex_full_fft``. For one leg, the filtered field has the
schematic form

.. math::

   n_{\ell m}(\mathbf{x};r)
   =
   (W_{\ell m}(r)\circ n)(\mathbf{x}),

where :math:`W_{\ell m}` contains the radial scale and spherical harmonic
angular dependence. The user controls this layer mainly through ``r12``,
``r13``, and ``l_max`` rather than by writing a separate YAML window for each
``(\ell,m)`` mode. Internally, the implementation still uses the optimized
Legendre kernels for supported low-order modes.

Practical Rules Of Thumb
------------------------

- Use ``sphere`` for smoothing before Counting, 2PCF, or 3PCF measurements.
- Use ``cubic`` when the desired smoothing support is an axis-aligned box rather
  than a radial ball.
- Use ``shell`` for real-space pair separations, or compose custom shell-like
  windows from ``sphere`` when a finite radial thickness is needed.
- Use ``ring`` for thin redshift-space pair bins, ``cylshell`` for cylindrical
  side surfaces, and ``disk`` or ``cylinder`` when a finite transverse or
  line-of-sight average is desired.
- Keep the LOS axis-aligned when possible if runtime matters. Oblique LOS is
  supported, but it uses a more general kernel path.
- Use ``directional_derivative``, ``laplacian``, and ``inverse_laplacian`` as
  operator windows, usually composed with a smoothing window, when you need
  gradients, divergence, curl, or Poisson-like operations on a represented
  field.
- Use ``binning_window_cache`` for heavy 2PCF runs when many binning windows are
  rebuilt under a memory-saving strategy.
- Treat window choices as part of the science definition of the statistic, not
  only as performance parameters.
