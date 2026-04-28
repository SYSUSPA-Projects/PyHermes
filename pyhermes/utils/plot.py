import numpy as np


def _as_1d_array(values, name):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr


def _get_smu_arrays(corr2pcf_smu):
    s = _as_1d_array(corr2pcf_smu.s, "corr2pcf_smu.s")
    mu = _as_1d_array(corr2pcf_smu.mu, "corr2pcf_smu.mu")
    xi = np.asarray(corr2pcf_smu.xi, dtype=np.float64)
    if xi.shape != (s.size, mu.size):
        raise ValueError(
            f"corr2pcf_smu.xi must have shape {(s.size, mu.size)}, got {xi.shape}."
        )
    if np.any(mu < 0.0) or np.any(mu > 1.0):
        raise ValueError("corr2pcf_smu.mu is expected to lie in [0, 1].")

    order = np.argsort(mu)
    return s, mu[order], xi[:, order]


def smu_to_half_plane(corr2pcf_smu, side="right", s_power=2):
    """
    Convert xi(s, mu) sampled on mu in [0, 1] to one half of the
    (s_perp, s_parallel) plane.
    """
    s, mu, xi = _get_smu_arrays(corr2pcf_smu)

    negative_mu_mask = mu != 0.0
    mu_full = np.concatenate([-mu[negative_mu_mask][::-1], mu])
    xi_full = np.concatenate([xi[:, negative_mu_mask][:, ::-1], xi], axis=1)

    S, MU = np.meshgrid(s, mu_full, indexing="ij")
    s_par = S * MU
    s_perp = S * np.sqrt(np.maximum(0.0, 1.0 - MU**2))

    if side == "left":
        s_perp = -s_perp
    elif side != "right":
        raise ValueError("side must be 'left' or 'right'.")

    scaled_xi = (S**s_power) * xi_full
    return s_perp, s_par, scaled_xi


def _combined_vmax(arrays, percentile):
    values = np.concatenate([np.abs(arr).ravel() for arr in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    vmax = np.nanpercentile(values, percentile)
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = np.nanmax(values)
    return float(vmax) if np.isfinite(vmax) and vmax > 0.0 else 1.0


def _grid_triangles(n_row, n_col):
    triangles = []
    for i in range(n_row - 1):
        for j in range(n_col - 1):
            p00 = i * n_col + j
            p01 = p00 + 1
            p10 = (i + 1) * n_col + j
            p11 = p10 + 1
            triangles.append((p00, p10, p11))
            triangles.append((p00, p11, p01))
    return np.asarray(triangles, dtype=np.int32)


def _triangulation(x, y):
    import matplotlib.tri as mtri

    return mtri.Triangulation(
        x.ravel(),
        y.ravel(),
        triangles=_grid_triangles(*x.shape),
    )


def plot_corr2pcf_smu(
    corr2pcf1,
    corr2pcf2=None,
    label1=None,
    label2=None,
    title=None,
    figsize=None,
    ax=None,
    cmap="RdBu_r",
    n_levels=81,
    vmax=None,
    percentile=98,
    s_power=2,
    colorbar=True,
    colorbar_kwargs=None,
    label_fontsize=16,
    tick_fontsize=12,
    title_fontsize=18,
):
    """
    Plot s**s_power * xi(s_perp, s_parallel).

    If corr2pcf2 is provided, corr2pcf1 is shown on the left half-plane and
    corr2pcf2 on the right half-plane. If corr2pcf2 is omitted, corr2pcf1 is
    mirrored to show the full plane.
    """
    import matplotlib.pyplot as plt

    if figsize is None:
        figsize = (7, 7)
    if colorbar_kwargs is None:
        colorbar_kwargs = {}

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x_left, y_left, z_left = smu_to_half_plane(corr2pcf1, side="left", s_power=s_power)
    if corr2pcf2 is None:
        x_right, y_right, z_right = smu_to_half_plane(corr2pcf1, side="right", s_power=s_power)
    else:
        x_right, y_right, z_right = smu_to_half_plane(corr2pcf2, side="right", s_power=s_power)

    if vmax is None:
        vmax = _combined_vmax([z_left, z_right], percentile)
    levels = np.linspace(-vmax, vmax, n_levels)

    cf = ax.tricontourf(
        _triangulation(x_left, y_left),
        z_left.ravel(),
        levels=levels,
        cmap=cmap,
        extend="both",
    )
    ax.tricontourf(
        _triangulation(x_right, y_right),
        z_right.ravel(),
        levels=levels,
        cmap=cmap,
        extend="both",
    )

    cbar = None
    if colorbar:
        default_cbar_kwargs = {"shrink": 0.75, "fraction": 0.05, "pad": 0.04, "aspect": 25}
        default_cbar_kwargs.update(colorbar_kwargs)
        cbar = fig.colorbar(cf, ax=ax, label=rf"$s^{s_power} \xi$", **default_cbar_kwargs)
        cbar.set_label(rf"$s^{s_power} \xi$", fontsize=label_fontsize)
        cbar.ax.tick_params(labelsize=tick_fontsize)

    ax.axvline(0.0, color="k", lw=0.8, alpha=0.5)
    if label1:
        ax.text(0, 1.02, label1, transform=ax.transAxes, ha="left", va="bottom", fontsize=14)
    if corr2pcf2 is not None and label2:
        ax.text(1, 1.02, label2, transform=ax.transAxes, ha="right", va="bottom", fontsize=14)

    ax.set_xlabel(r"$s_\perp$", fontsize=label_fontsize)
    ax.set_ylabel(r"$s_\parallel$", fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    if title:
        ax.set_title(title, fontsize=title_fontsize)
    ax.set_aspect("equal", adjustable="box")
    return fig, ax, cbar
