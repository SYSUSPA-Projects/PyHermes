Getting Started
===============

This section walks through the standard PyHermes workflow:

1. create the multiresolution coefficient field with ``Convols``
2. sample the field with ``Counting`` if needed
3. measure the 2PCF with ``Corr_2PCF``
4. measure the 3PCF with ``Corr_3PCF``
5. measure 3PCF multipoles with ``Corr_3PCF_Multipole``
6. optionally run the full workflow from a single parameter file

The examples in this section match the scripts shipped in the repository under
``examples/``.

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   
   convols/convols
   counting/counting
   corr_2pcf/corr_2pcf
   corr_3pcf/corr_3pcf
   corr_3pcf_multipole/corr_3pcf_multipole
   allinone
