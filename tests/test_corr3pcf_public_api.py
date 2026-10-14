"""Public projection-to-particle-output workflow, including directory output."""

from pathlib import Path

import numpy as np

from pyhermes.base import SFCProjection
from pyhermes.io import Corr3PCFMultipoleData, SFCField
from pyhermes.theory import Corr_3PCF_Multipole


def test_particle_companion_and_directory_output_roundtrip(tmp_path, monkeypatch):
    positions = np.random.default_rng(322).uniform(0, 32, (32, 3)).astype(np.float32)
    weights = np.linspace(0.5, 1.5, len(positions))
    field_path = tmp_path / "field.pkl"
    SFCProjection({"SFCProjection": {
        "particle_pos": positions, "catalog_weight": weights,
        "box_size": 32., "J": 3, "threads": 2,
        "wavelet_mode": "db2", "wavelet_level": 3, "phi_resolution": 8,
        "save_particle_data": True, "particle_data_path": str(tmp_path / "particles.npz"),
        "fout_path": str(field_path),
    }}).run()
    field = SFCField(data_path=str(field_path), threads=2)
    particles = field.get_particle_data()
    np.testing.assert_array_equal(particles["pos"], positions)
    np.testing.assert_allclose(particles["projection_weight"], weights / weights.sum())
    result = Corr_3PCF_Multipole({"Corr_3PCF_Multipole": {
        "sfc_field": str(field_path), "random": "uniform", "threads": 2,
        "summation_method": "particle", "summation_backend": "cpu",
        "window2": {"type": "sphere", "len_args": {"R": 1.}},
        "window3": {"type": "sphere", "len_args": {"R": 1.}},
        "binning_window12": {"type": "shell", "mapping": {"R": "r12"}},
        "binning_window13": {"type": "shell", "mapping": {"R": "r13"}},
        "sampling": {"r12": [4., 6.], "r13": [8., 10.], "mode": "paired"},
        "l_max": 2, "fout_path": str(tmp_path / "results") + "/",
    }}).run()
    files = list((tmp_path / "results").glob("*.pkl"))
    assert len(files) == 1 and Path(files[0]).name.startswith("Output_Corr3PCFMultipoleData_")
    restored = Corr3PCFMultipoleData(data_path=str(files[0]))
    assert restored.zeta_l.shape == (2, 3) and np.isfinite(restored.zeta_l).all()
    assert restored.task_params["summation_method"] == "particle"
    np.testing.assert_array_equal(restored.zeta_l, result.zeta_l)
    np.testing.assert_array_equal(restored.l, [0, 1, 2])

    monkeypatch.chdir(tmp_path)
    result.save_corr3pcf_multipole("plain.pkl")
    plain = Corr3PCFMultipoleData(data_path="plain.pkl")
    np.testing.assert_array_equal(plain.zeta_l, result.zeta_l)
    assert plain.task_params["summation_method"] == "particle"
