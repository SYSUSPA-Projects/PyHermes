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
``examples/``. They are also written around a common workflow ladder that
appears repeatedly throughout the documentation:

- **Workflow A. Command-Line Driver**
- **Workflow B. Config-Driven Python API**
- **Workflow C. Task Object with Attribute Overrides**
- **Workflow D. Manual Input Objects and Custom Preparation**
- **Workflow E. Low-Level Building Blocks**

You do not need to use every layer. A good rule of thumb is:

- start with Workflow A or B for routine usage
- move to Workflow C or D when you want interactive control
- use Workflow E only when you need a custom estimator or low-level extension

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   
   convols/convols
   counting/counting
   corr_2pcf/corr_2pcf
   corr_3pcf/corr_3pcf
   corr_3pcf_multipole/corr_3pcf_multipole
   allinone
