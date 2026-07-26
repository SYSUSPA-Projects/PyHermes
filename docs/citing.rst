Citing PyHermes
===============

If PyHermes contributes to a publication, presentation, or released data
product, please cite the software version used and the four papers below.
Together, they describe the MRACS foundation, the in-situ field--window
formulation, the isotropic 3PCF multipole algorithm, and the complete
Hermes/PyHermes framework. GitHub and compatible reference managers can read
the repository's ``CITATION.cff`` file directly.

Software citation
-----------------

Please include the version of PyHermes used in the analysis:

.. code-block:: bibtex

   @software{pyhermes_software_2026,
     author  = {Long-Long Feng and Tengpeng Xu and Tian-Cheng Luan and collaborators},
     title   = {PyHermes: High-performance multiresolution cosmic statistics in Python},
     year    = {2026},
     version = {1.1.0},
     url     = {https://github.com/SYSUSPA-Projects/PyHermes}
   }

Required papers
---------------

The Hermes/PyHermes manuscript has been submitted to arXiv. Its permanent
identifier will be inserted in the following entry as soon as it is assigned:

.. important::

   The author list and title below follow the submitted manuscript. The arXiv
   identifier and public URL are intentionally marked as pending.

.. code-block:: bibtex

   @article{Feng2026Hermes,
     author  = {Long-long Feng and Tengpeng Xu and Tian-Cheng Luan and Jiawei Li
                and Xin Sun and Wenjie Ju and Zhuoyang Li and Shiyu Yue
                and Weishan Zhu and Yan-Chuan Cai},
     title   = {Hermes -- Towards an Optimal High-Performance Algorithm for
                Cosmic Statistics of Large Data Sets},
     year    = {2026},
     note    = {Submitted to arXiv; identifier pending}
   }

   @article{Feng2007,
     author  = {Feng, Long-Long},
     title   = {The Beylkin-Cramer Summation Rule and a New Fast Algorithm of
                Cosmic Statistics for Large Data Sets},
     journal = {The Astrophysical Journal},
     volume  = {658},
     number  = {1},
     pages   = {25--35},
     year    = {2007},
     doi     = {10.1086/511024},
     eprint  = {astro-ph/0512167}
   }

   @article{Yue2024,
     author  = {Yue, Shiyu and Feng, Longlong and Ju, Wenjie and Pan, Jun
                and Huang, Zhiqi and Fang, Feng and Li, Zhuoyang
                and Cai, Yan-Chuan and Zhu, Weishan},
     title   = {Pair counting without binning -- a new approach to correlation
                functions in clustering statistics},
     journal = {Monthly Notices of the Royal Astronomical Society},
     volume  = {535},
     number  = {4},
     pages   = {3500--3516},
     year    = {2024},
     doi     = {10.1093/mnras/stae2513}
   }

   @article{Ju2026,
     author  = {Ju, Wenjie and Feng, Longlong and Huang, Zhiqi and Sun, Xin
                and Zhu, Weishan},
     title   = {An optimal in situ multipole algorithm for the isotropic
                three-point correlation function},
     journal = {Monthly Notices of the Royal Astronomical Society},
     volume  = {546},
     number  = {1},
     pages   = {staf2275},
     year    = {2026},
     doi     = {10.1093/mnras/staf2275}
   }

Please also cite the scientific datasets, simulations, and external methods
used in a particular analysis. A PyHermes citation does not replace those
method-specific references.
