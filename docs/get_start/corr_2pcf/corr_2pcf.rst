Corr_2pcf
=========


.. role:: strike
    :class: strike

Wanna calculate 2pcf with boost-speed? Come and try PyHermes.


Finally, the output file *corr2pcf_r5.txt* looks like:

.. code-block:: python

    # Corr_2PCF output from PyHermes v0.0.7, TIME: 2024.08.22-13:46:31
    # Parameters from input :
    #  R1            = 1.0
    #  R2            = 150.0
    #  xi_num        = 20
    #  threads       = 20
    #  fout_dir      = ./check_2pcf
    #  deltac_in_pat = ./convols_L512_r5_pywt.npy
    # Parameters from DeltaC:
    #  J             = 9
    #  Radius        = 5
    #  SimBoxL       = 1000
    #  SampRate      = 1024
    #  bandwidth     = 1
    #  fin_path      = https://pyhermes.astroslacker.com/_downloads/906e0695649e3634a5fe8081b9ab2086/quijote10000.bin
    #  fin_size      = 406793
    #  fin_format    = generic
    #  Window type   = shell
    #  wavelet_mode  = db2
    #  wavelet_level = 10

    ---------------------------
    r[h-1 Mpc]  , xi
    1.000000e+00, 4.849146e+00
    8.842105e+00, 1.291518e+00
    1.668421e+01, 2.761841e-01
    2.452632e+01, 1.201382e-01
    3.236842e+01, 6.153875e-02
    4.021053e+01, 3.495787e-02
    4.805263e+01, 2.120653e-02
    5.589474e+01, 1.290703e-02
    6.373684e+01, 7.618866e-03
    7.157895e+01, 4.722179e-03
    7.942105e+01, 3.620078e-03
    8.726316e+01, 3.336819e-03
    9.510526e+01, 2.975417e-03
    1.029474e+02, 2.706145e-03
    1.107895e+02, 1.797030e-03
    1.186316e+02, 3.404875e-04
    1.264737e+02, -6.665408e-04
    1.343158e+02, -1.202132e-03
    1.421579e+02, -1.517370e-03
    1.500000e+02, -1.325033e-03

Thats it, great! :strike:`Im wondering why this cannot displayed properly?`