Hermes in one page
==================

Hermes and PyHermes
-------------------

**Hermes** is an in situ multiresolution framework for cosmic statistics.
**PyHermes** is its open-source Python implementation. The framework replaces
repeated ex situ enumeration of particle pairs or tuples with algebraic
operations among window-filtered continuous fields.

The central workflow is deliberately small:

1. A catalogue is projected onto a compact scaling-function basis.
2. A ``WindowFunc`` describes the spatial operation required by a statistic.
3. ``SFCField @ WindowFunc`` constructs a filtered field by FFT convolution.
4. Products and spatial averages of filtered fields produce the requested
   count-level quantity or connected statistic.

The important consequence is reuse. Once a catalogue has become an
``SFCField``, changing a separation bin, smoothing scale, multipole projector,
or differential operator does not require projecting the catalogue again.

Why a field--window language?
-----------------------------

Conventional estimators often arrive as independent counting algorithms: a
spherical shell for an isotropic 2PCF, rings for a redshift-space 2PCF,
rotated configurations for a standard 3PCF, and spherical harmonics for 3PCF
multipoles. Hermes treats these as different windows acting on the same field.

This viewpoint offers three practical advantages:

**Reuse**
   The catalogue-to-field projection is performed once. The same field can be
   used for one-point, two-point, three-point, weighted, and differential
   analyses.

**Flexibility**
   Binning is no longer restricted to one geometric convention. Thin shells,
   thick shells, Gaussian shells, rings, disks, cylindrical windows, and
   user-defined Fourier kernels use the same interface.

**Scalability**
   FFT window operations scale primarily with the number of MRA coefficients
   and requested windows, rather than directly with the number of catalogue
   pairs or triplets. MPI, Numba threads, and an optional CUDA summation backend
   are available for the expensive workflows.

What PyHermes currently computes
--------------------------------

The current release provides:

- ``SFCProjection`` for catalogue-to-field reconstruction;
- ``Counting`` for random-position sampling and one-point PDFs;
- ``Corr_2PCF`` for isotropic and anisotropic two-point statistics;
- ``Corr_3PCF`` for Monte Carlo standard 3PCFs with particle or box-random
  centres;
- ``Corr_3PCF_Multipole`` for 3PCF multipoles and radial scans;
- catalogue weights, physical field values, and marked fields;
- smoothing, binning, high-pass, differential, and inverse-Laplacian windows;
- CPU and GPU backends for the final 3PCF-multipole contraction.

The examples in the paper use periodic simulation boxes. Explicit random
fields can represent non-uniform reference catalogues, but a complete survey
analysis must still supply its own mask, selection function, and systematic
weights. PyHermes provides the field and estimator machinery; it does not
guess an observational selection function for you.

The four layers
---------------

Catalogue layer
~~~~~~~~~~~~~~~

Input positions may come from in-memory arrays or from raw binary, NumPy NPZ,
Gadget binary, Gadget HDF5, and FoF readers. ``catalog_weight`` represents a
selection or catalogue weight, while ``field_value`` carries a mark or a
physical quantity such as mass or one velocity component.

MRA-field layer
~~~~~~~~~~~~~~~

``SFCProjection`` returns an ``SFCField`` with coefficients
``epsilon``. The multiresolution level ``J`` gives ``L = 2**J`` coefficients
per axis and ``2**(3*J)`` coefficients in three dimensions. The default basis
is Daubechies D4, named ``db2`` by PyWavelets.

Window layer
~~~~~~~~~~~~

``WindowFunc`` stores a Fourier-space kernel compatible with one MRA field.
Windows fall into four roles:

- smoothing windows define a local average or filter;
- binning windows define pair or triangle separation support;
- multipole windows project angular structure;
- operator windows apply derivatives, the Laplacian, or the inverse Laplacian.

Task layer
~~~~~~~~~~

Task classes load or accept fields, apply the requested windows, compute the
necessary field products, and save typed result objects such as
``Corr2PCFData`` or ``Corr3PCFMultipoleData``. The task interface is the usual
production route; direct field arithmetic is invaluable for understanding and
extending an estimator.

Numerical scope
---------------

Hermes is not a promise that every one-off measurement is faster than every
specialised counter. For a small catalogue and one conventional statistic, a
highly tuned pair counter may be the simpler tool. PyHermes is strongest when
the reconstructed field is reused, when windows are non-standard, when many
configurations are required, or when higher-order tuple counting would become
the dominant cost.

Finite ``J`` also means finite spatial resolution. If the MRA cell size
``box_size / 2**J`` is comparable to the requested separation or bin width,
small-scale amplitudes will be suppressed. Convergence in ``J`` is therefore a
scientific check, not merely a performance setting.

Terminology used in this guide
------------------------------

``SFCField``
   A field represented by scaling-function coefficients. ``SFC`` means
   scaling-function coefficients.

``SFCProjection``
   The catalogue-to-field projection task.

``window``
   An optional smoothing or physical operator applied to an input vertex.

``binning_window``
   A window whose parameters are mapped from sampled pair or triangle
   coordinates. Older names such as "pair window" are not used.

``filtered field``
   The result of applying a window to an ``SFCField``.

``product``
   Either an intermediate count-level field product, such as ``dd`` or
   ``delta_ddd_l``, or a final statistic, such as ``xi``, ``Q``, or ``zeta_l``.

Executable tutorials
--------------------

The tracked notebooks in ``examples/notebooks/`` are part of the public user
interface. They share one foundation and then branch by scientific goal:

``quick_start.ipynb`` -> [``particle_io.ipynb`` when adapting input data] ->
``sfc_projection.ipynb`` -> ``window.ipynb``.

From there, use ``physical_fields.ipynb`` for differential operators and
derived fields, ``counting.ipynb`` for one-point distributions, or
``corr2pcf.ipynb`` -> ``corr3pcf.ipynb`` for correlation statistics.

See :doc:`get_start/get_start` for the full map and
:doc:`get_start/quick_start` for the first runnable workflow.
