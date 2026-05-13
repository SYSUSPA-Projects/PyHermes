"""
Plot helpers for PyHermes.

``plot_styles.json`` stores the built-in defaults for ``plot_corr2pcf_2d``:
base plot values, accepted ``**kwargs``, nested kwargs that should be merged,
and corner-label appearance. Users normally override these values at call time
instead of editing the JSON.

Direct ``plot_corr2pcf_2d`` arguments:
    corr2pcf1, corr2pcf2, quadrants, coordinates, add_contour, value,
    s_range, s_min, s_max, quadrant_labels, label1, label2, title, ax,
    add_colorbar.

Accepted ``**kwargs``:
    figsize, cmap, n_levels, vmin, vmax, percentile, s_power,
    colorbar_kwargs, colorbar_nbins, colorbar_tick_prune, label_fontsize,
    tick_fontsize, title_fontsize, text_fontsize, xlabel, ylabel,
    tick_params, spine_linewidth, symmetric_limits, contour_kwargs,
    contour_levels, center_lines, center_line_kwargs,
    s_min_mask, s_min_mask_kwargs.

Examples:
    plot_corr2pcf_2d(corr, title="redshift space")

    plot_corr2pcf_2d(
        corr,
        add_contour=True,
        value="log10_1p_xi",
        s_min=20,
        contour_levels=[0.001, 0.01, 0.05, 0.1],
    )

    plot_corr2pcf_2d(
        corr_smu,
        corr_rppi,
        coordinates={"left": "smu", "right": "rppi"},
        label1="smu mode",
        label2="rppi mode",
        cmap="viridis",
        colorbar_kwargs={"shrink": 0.7},
        contour_kwargs={"colors": "black", "linewidths": 0.8},
    )
"""

import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_plot_config():
    with open(Path(__file__).with_name("plot_styles.json"), encoding="utf-8") as f:
        return json.load(f)["corr2pcf_2d"]


PLOT_CONFIG = _load_plot_config()
PLOT_BASE_OPTIONS = PLOT_CONFIG["base_options"]
PLOT_KWARG_DEFAULTS = PLOT_CONFIG["kwarg_defaults"]
PLOT_OPTION_KWARGS = tuple(PLOT_CONFIG["option_kwargs"])
PLOT_MERGE_KWARGS = set(PLOT_CONFIG["merge_kwargs"])
CORNER_LABEL_BBOX = PLOT_CONFIG["corner_label_bbox"]
CORNER_LABEL_POSITIONS = PLOT_CONFIG["corner_label_positions"]
QUADRANTS = ("upper_left", "upper_right", "lower_left", "lower_right")

PLOT_VALUE_NAMES = {"s_power_xi", "xi", "log10_1p_xi"}
COORDINATE_NAMES = {"smu", "rppi"}


def _corr_array(corr2pcf, name, object_name):
    try:
        values = getattr(corr2pcf, name)
    except AttributeError as exc:
        raise ValueError(f"{object_name}.{name} is required.") from exc

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{object_name}.{name} must be a 1D array.")
    if arr.size == 0:
        raise ValueError(f"{object_name}.{name} must not be empty.")
    return arr


def _check_plot_value(value):
    if value not in PLOT_VALUE_NAMES:
        raise ValueError(
            "value must be 's_power_xi', 'xi', or 'log10_1p_xi'."
        )
    return value


def _plot_values(xi, radius, s_power, value):
    value = _check_plot_value(value)
    if value == "s_power_xi":
        return (radius**s_power) * xi
    if value == "xi":
        return xi
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(1.0 + xi)


def _plot_value_label(value, s_power):
    value = _check_plot_value(value)
    if value == "s_power_xi":
        return rf"$s^{s_power} \xi$"
    if value == "xi":
        return r"$\xi$"
    return r"$\log (1 + \xi)$"


def _plot_options(add_contour, value, title, kwargs):
    unexpected = sorted(set(kwargs) - set(PLOT_KWARG_DEFAULTS))
    if unexpected:
        names = ", ".join(unexpected)
        raise TypeError(f"Unexpected plot_corr2pcf_2d keyword argument(s): {names}.")
    if not isinstance(add_contour, bool):
        raise ValueError("add_contour must be True or False.")

    plot_kwargs = dict(PLOT_KWARG_DEFAULTS)
    plot_kwargs.update(kwargs)

    options = copy.deepcopy(PLOT_BASE_OPTIONS)
    options["add_contour"] = add_contour
    options["value"] = _check_plot_value(value)
    options["title"] = title

    for name in PLOT_OPTION_KWARGS:
        override = plot_kwargs[name]
        if override is None:
            continue
        if name in PLOT_MERGE_KWARGS:
            options[name].update(override)
        else:
            options[name] = override

    if plot_kwargs["colorbar_kwargs"] is not None:
        options["colorbar_kwargs"].update(plot_kwargs["colorbar_kwargs"])
    return options, plot_kwargs


def _s_limits(s_range, s_min, s_max):
    if s_range is not None:
        if s_min is not None or s_max is not None:
            raise ValueError("Use either s_range or s_min/s_max, not both.")
        if len(s_range) != 2:
            raise ValueError("s_range must be a 2-item sequence.")
        s_min, s_max = s_range

    if s_min is not None:
        s_min = float(s_min)
        if s_min < 0.0:
            raise ValueError("s_min must be non-negative.")
    if s_max is not None:
        s_max = float(s_max)
        if s_max < 0.0:
            raise ValueError("s_max must be non-negative.")
    if s_min is not None and s_max is not None and s_min > s_max:
        raise ValueError("s_min must be smaller than or equal to s_max.")
    return s_min, s_max


def _prepare_plot_item(name, x, y, z, s_min, s_max, value):
    radius = np.sqrt(x**2 + y**2)
    point_mask = np.ones(radius.shape, dtype=bool)
    if s_min is not None:
        point_mask &= radius >= s_min
    if s_max is not None:
        point_mask &= radius <= s_max
    if not np.any(point_mask):
        raise ValueError("s range does not include any plotted points.")

    z_plot = np.asarray(z, dtype=np.float64).copy()
    selected = z_plot[point_mask]
    if np.any(~np.isfinite(selected)):
        if _check_plot_value(value) == "log10_1p_xi":
            raise ValueError(
                "value='log10_1p_xi' requires xi > -1 in the selected s range."
            )
        raise ValueError("plot values must be finite in the selected s range.")
    z_plot[~point_mask] = 0.0
    return name, x, y, z_plot, point_mask


def _get_smu_arrays(corr2pcf_smu):
    s = _corr_array(corr2pcf_smu, "s", "corr2pcf_smu")
    mu = _corr_array(corr2pcf_smu, "mu", "corr2pcf_smu")
    xi = np.asarray(corr2pcf_smu.xi, dtype=np.float64)
    if xi.shape != (s.size, mu.size):
        raise ValueError(
            f"corr2pcf_smu.xi must have shape {(s.size, mu.size)}, got {xi.shape}."
        )
    if np.any(mu < 0.0) or np.any(mu > 1.0):
        raise ValueError("corr2pcf_smu.mu is expected to lie in [0, 1].")

    order = np.argsort(mu)
    return s, mu[order], xi[:, order]


def _get_rppi_arrays(corr2pcf_rppi):
    rp = _corr_array(corr2pcf_rppi, "rp", "corr2pcf_rppi")
    pi = _corr_array(corr2pcf_rppi, "pi", "corr2pcf_rppi")
    xi = np.asarray(corr2pcf_rppi.xi, dtype=np.float64)
    if xi.shape != (rp.size, pi.size):
        raise ValueError(
            f"corr2pcf_rppi.xi must have shape {(rp.size, pi.size)}, got {xi.shape}."
        )

    rp_order = np.argsort(rp)
    pi_order = np.argsort(pi)
    return rp[rp_order], pi[pi_order], xi[np.ix_(rp_order, pi_order)]


def smu_to_half_plane(corr2pcf_smu, side="right", s_power=2, value="s_power_xi"):
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

    return s_perp, s_par, _plot_values(xi_full, S, s_power, value)


def rppi_to_half_plane(corr2pcf_rppi, side="right", s_power=2, value="s_power_xi"):
    """
    Convert xi(rp, pi) sampled on pi >= 0 to one half of the
    (rp, pi) plane.
    """
    rp, pi, xi = _get_rppi_arrays(corr2pcf_rppi)

    negative_pi_mask = pi != 0.0
    pi_full = np.concatenate([-pi[negative_pi_mask][::-1], pi])
    xi_full = np.concatenate([xi[:, negative_pi_mask][:, ::-1], xi], axis=1)

    RP, PI = np.meshgrid(rp, pi_full, indexing="ij")
    x = RP.copy()
    if side == "left":
        x = -x
    elif side != "right":
        raise ValueError("side must be 'left' or 'right'.")

    s = np.sqrt(RP**2 + PI**2)
    return x, PI, _plot_values(xi_full, s, s_power, value)


def smu_to_quadrant(corr2pcf_smu, quadrant="upper_right", s_power=2, value="s_power_xi"):
    """
    Convert xi(s, mu) sampled on mu in [0, 1] to one quadrant of the
    (s_perp, s_parallel) plane.
    """
    quadrant = _check_quadrant_name(quadrant)
    s, mu, xi = _get_smu_arrays(corr2pcf_smu)

    S, MU = np.meshgrid(s, mu, indexing="ij")
    s_par = S * MU
    s_perp = S * np.sqrt(np.maximum(0.0, 1.0 - MU**2))

    if "left" in quadrant:
        s_perp = -s_perp
    if "lower" in quadrant:
        s_par = -s_par

    return s_perp, s_par, _plot_values(xi, S, s_power, value)


def rppi_to_quadrant(corr2pcf_rppi, quadrant="upper_right", s_power=2, value="s_power_xi"):
    """
    Convert xi(rp, pi) sampled on pi >= 0 to one quadrant.
    """
    quadrant = _check_quadrant_name(quadrant)
    rp, pi, xi = _get_rppi_arrays(corr2pcf_rppi)

    RP, PI = np.meshgrid(rp, pi, indexing="ij")
    x = RP.copy()
    if "left" in quadrant:
        x = -x
    if "lower" in quadrant:
        PI = -PI

    s = np.sqrt(RP**2 + PI**2)
    return x, PI, _plot_values(xi, s, s_power, value)


COORDINATE_CONVERTERS = {
    "smu": (smu_to_half_plane, smu_to_quadrant),
    "rppi": (rppi_to_half_plane, rppi_to_quadrant),
}


def _check_quadrant_name(name):
    if name not in QUADRANTS:
        raise ValueError(
            "quadrant must be one of upper_left, upper_right, lower_left, lower_right."
        )
    return name


def _check_coordinate_name(name):
    if name not in COORDINATE_NAMES:
        raise ValueError("coordinates must contain only 'smu' or 'rppi'.")
    return name


def _half_plane_coordinates(coordinates):
    if isinstance(coordinates, str):
        name = _check_coordinate_name(coordinates)
        return name, name
    if isinstance(coordinates, dict):
        required = {"left", "right"}
        if set(coordinates) != required:
            raise ValueError(
                "coordinates dict must contain exactly 'left' and 'right'."
            )
        return (
            _check_coordinate_name(coordinates["left"]),
            _check_coordinate_name(coordinates["right"]),
        )
    if len(coordinates) != 2:
        raise ValueError(
            "coordinates must be 'smu', 'rppi', or a 2-item sequence for "
            "(corr2pcf1, corr2pcf2)."
        )
    return tuple(_check_coordinate_name(name) for name in coordinates)


def _quadrant_map(quadrants):
    if isinstance(quadrants, dict):
        quadrant_map = {
            _check_quadrant_name(name): corr
            for name, corr in quadrants.items()
            if corr is not None
        }
    else:
        if len(quadrants) != 4:
            raise ValueError(
                "quadrants must be a dict or a 4-item sequence ordered as "
                "(upper_left, upper_right, lower_left, lower_right)."
            )
        quadrant_map = {
            quadrant: corr
            for quadrant, corr in zip(QUADRANTS, quadrants)
            if corr is not None
        }

    if not quadrant_map:
        raise ValueError("quadrants must contain at least one corr2pcf object.")
    return quadrant_map


def _quadrant_coordinates(coordinates):
    if isinstance(coordinates, str):
        name = _check_coordinate_name(coordinates)
        return {quadrant: name for quadrant in QUADRANTS}
    if isinstance(coordinates, dict):
        return {
            _check_quadrant_name(quadrant): _check_coordinate_name(name)
            for quadrant, name in coordinates.items()
        }
    if len(coordinates) != 4:
        raise ValueError(
            "coordinates must be 'smu', 'rppi', or a 4-item sequence ordered as "
            "(upper_left, upper_right, lower_left, lower_right)."
        )
    return {
        quadrant: _check_coordinate_name(name)
        for quadrant, name in zip(QUADRANTS, coordinates)
    }


def _plot_items(corr2pcf1, corr2pcf2, quadrants, coordinates, s_power, value):
    if quadrants is None:
        if corr2pcf1 is None:
            raise ValueError("corr2pcf1 is required when quadrants is not provided.")

        coordinates_left, coordinates_right = _half_plane_coordinates(coordinates)
        left_converter = COORDINATE_CONVERTERS[coordinates_left][0]
        x_left, y_left, z_left = left_converter(
            corr2pcf1,
            side="left",
            s_power=s_power,
            value=value,
        )

        if corr2pcf2 is None:
            right_corr = corr2pcf1
            right_converter = left_converter
        else:
            right_corr = corr2pcf2
            right_converter = COORDINATE_CONVERTERS[coordinates_right][0]
        x_right, y_right, z_right = right_converter(
            right_corr,
            side="right",
            s_power=s_power,
            value=value,
        )
        return [
            ("upper_left", x_left, y_left, z_left),
            ("upper_right", x_right, y_right, z_right),
        ]

    quadrant_map = _quadrant_map(quadrants)
    coordinates_map = _quadrant_coordinates(coordinates)
    plot_items = []
    for quadrant in QUADRANTS:
        corr = quadrant_map.get(quadrant)
        if corr is None:
            continue
        coordinates_name = coordinates_map.get(quadrant)
        if coordinates_name is None:
            raise ValueError(f"coordinates is missing an entry for {quadrant}.")
        converter = COORDINATE_CONVERTERS[coordinates_name][1]
        x, y, z = converter(corr, quadrant=quadrant, s_power=s_power, value=value)
        plot_items.append((quadrant, x, y, z))
    return plot_items


def _value_limits(plot_items, vmin, vmax, percentile, symmetric_limits):
    values = np.concatenate(
        [z[point_mask].ravel() for _, _, _, z, point_mask in plot_items]
    )
    values = values[np.isfinite(values)]
    if symmetric_limits and vmin is None:
        if vmax is None:
            if values.size == 0:
                auto_vmax = 1.0
            else:
                auto_vmax = np.nanpercentile(np.abs(values), percentile)
                if not np.isfinite(auto_vmax) or auto_vmax == 0.0:
                    auto_vmax = np.nanmax(np.abs(values))
                auto_vmax = float(auto_vmax) if auto_vmax > 0.0 else 1.0
        else:
            auto_vmax = vmax
        auto_vmin = -auto_vmax
    elif values.size == 0:
        auto_vmin, auto_vmax = -1.0, 1.0
    else:
        percentile = float(percentile)
        if percentile <= 0.0 or percentile > 100.0:
            raise ValueError("percentile must be in (0, 100].")
        lower_percentile = max(0.0, 100.0 - percentile)
        auto_vmin = float(np.nanpercentile(values, lower_percentile))
        auto_vmax = float(np.nanpercentile(values, percentile))
        if auto_vmin == auto_vmax:
            delta = abs(auto_vmax) * 0.1 if auto_vmax != 0.0 else 1.0
            auto_vmin -= delta
            auto_vmax += delta

    if vmin is None:
        vmin = auto_vmin
    if vmax is None:
        vmax = auto_vmax
    if vmin >= vmax:
        raise ValueError("vmin must be smaller than vmax.")
    return vmin, vmax


def _grid_triangles(x, y):
    n_row, n_col = x.shape
    triangles = []
    for i in range(n_row - 1):
        for j in range(n_col - 1):
            p00 = i * n_col + j
            p01 = p00 + 1
            p10 = (i + 1) * n_col + j
            p11 = p10 + 1

            y_mid = 0.25 * (y[i, j] + y[i, j + 1] + y[i + 1, j] + y[i + 1, j + 1])
            if y_mid < 0.0:
                triangles.append((p00, p10, p01))
                triangles.append((p01, p10, p11))
            else:
                triangles.append((p00, p10, p11))
                triangles.append((p00, p11, p01))
    return np.asarray(triangles, dtype=np.int32).reshape(-1, 3)


def _triangulation(x, y, point_mask):
    import matplotlib.tri as mtri

    triangles = _grid_triangles(x, y)
    triangulation = mtri.Triangulation(
        x.ravel(),
        y.ravel(),
        triangles=triangles,
    )
    triangle_mask = ~np.all(point_mask.ravel()[triangles], axis=1)
    if np.all(triangle_mask):
        raise ValueError("s range does not include any complete plot cells.")
    triangulation.set_mask(triangle_mask)
    return triangulation


def _add_s_min_mask(ax, s_min, options):
    if not options["s_min_mask"] or s_min is None:
        return

    circle = plt.Circle((0.0, 0.0), s_min, **options["s_min_mask_kwargs"])
    ax.add_patch(circle)


def _draw_plot_items(ax, plot_items, levels, vmin, vmax, options, contour_levels):
    cf = None
    for _, x, y, z, point_mask in plot_items:
        triangulation = _triangulation(x, y, point_mask)
        cf = ax.tricontourf(
            triangulation,
            z.ravel(),
            levels=levels,
            cmap=options["cmap"],
            extend="both",
        )
        if not options["add_contour"]:
            continue

        contour_levels = (
            np.linspace(vmin, vmax, 7)[1:-1]
            if contour_levels is None
            else contour_levels
        )
        cs = ax.tricontour(
            triangulation,
            z.ravel(),
            levels=contour_levels,
            **options["contour_kwargs"],
        )
        label_levels = cs.levels[::2] if cs.levels.size > 5 else cs.levels
        ax.clabel(
            cs,
            label_levels,
            inline=True,
            fontsize=max(9, options["tick_fontsize"] - 1),
            fmt="%.3f",
            colors=options["contour_kwargs"].get("colors"),
            inline_spacing=4,
        )
    return cf


def _add_colorbar(fig, ax, cf, options, s_power):
    cbar_label = _plot_value_label(options["value"], s_power)
    cbar = fig.colorbar(
        cf,
        ax=ax,
        label=cbar_label,
        **options["colorbar_kwargs"],
    )
    cbar.set_label(cbar_label, fontsize=options["label_fontsize"])
    cbar.ax.tick_params(labelsize=options["tick_fontsize"])

    if (
        options["colorbar_nbins"] is not None
        and "ticks" not in options["colorbar_kwargs"]
    ):
        import matplotlib.ticker as mticker

        locator_kwargs = {"nbins": options["colorbar_nbins"]}
        if options["colorbar_tick_prune"]:
            locator_kwargs["prune"] = options["colorbar_tick_prune"]
        cbar.locator = mticker.MaxNLocator(**locator_kwargs)
        cbar.update_ticks()
    return cbar


def _add_corner_label(ax, text, corner, options):
    x_text, y_text, ha, va = CORNER_LABEL_POSITIONS[corner]
    ax.text(
        x_text,
        y_text,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=options["text_fontsize"],
        bbox=CORNER_LABEL_BBOX,
    )


def _add_plot_labels(ax, plot_items, quadrants, corr2pcf2, labels, options):
    if quadrants is None:
        if corr2pcf2 is None:
            if labels["label1"]:
                _add_corner_label(ax, labels["label1"], "upper_right", options)
            return
        if labels["label1"]:
            _add_corner_label(ax, labels["label1"], "upper_left", options)
        if labels["label2"]:
            _add_corner_label(ax, labels["label2"], "upper_right", options)
        return

    quadrant_labels = labels["quadrant_labels"]
    if not quadrant_labels:
        return
    plotted_quadrants = {item[0] for item in plot_items}
    for name, text in quadrant_labels.items():
        quadrant = _check_quadrant_name(name)
        if quadrant in plotted_quadrants:
            _add_corner_label(ax, text, quadrant, options)


def _style_axes(ax, options):
    if options["center_lines"]:
        ax.axvline(0.0, **options["center_line_kwargs"])
        ax.axhline(0.0, **options["center_line_kwargs"])

    tick_params = {"labelsize": options["tick_fontsize"]}
    tick_params.update(options["tick_params"])
    ax.tick_params(**tick_params)

    spine_linewidth = options["spine_linewidth"]
    if spine_linewidth is not None:
        for spine in ax.spines.values():
            spine.set_linewidth(spine_linewidth)

    ax.set_xlabel(options["xlabel"], fontsize=options["label_fontsize"])
    ax.set_ylabel(options["ylabel"], fontsize=options["label_fontsize"])
    if options["title"]:
        ax.set_title(options["title"], fontsize=options["title_fontsize"])
    ax.set_aspect("equal", adjustable="box")


def plot_corr2pcf_2d(
    corr2pcf1=None,
    corr2pcf2=None,
    *,
    quadrants=None,
    coordinates="smu",
    add_contour=False,
    value="s_power_xi",
    s_range=None,
    s_min=None,
    s_max=None,
    quadrant_labels=None,
    label1=None,
    label2=None,
    title=None,
    ax=None,
    add_colorbar=True,
    **kwargs,
):
    """
    Plot a 2D Corr_2PCF value on the LOS plane.

    If corr2pcf2 is provided, corr2pcf1 is shown on the left half-plane and
    corr2pcf2 on the right half-plane. If corr2pcf2 is omitted, corr2pcf1 is
    mirrored to show the full plane. ``coordinates`` may be either "smu",
    "rppi", a dict keyed by left/right, or a 2-item sequence giving the
    coordinate type for (corr2pcf1, corr2pcf2).

    If quadrants is provided, it should be either a dict keyed by
    upper_left, upper_right, lower_left, lower_right, or a 4-item sequence in
    that order. Each quadrant is drawn from one corr2pcf object without
    mirroring across s_parallel=0. ``coordinates`` may be a matching dict or
    4-item sequence.

    ``value`` may be "s_power_xi" for ``s**s_power * xi``, "xi" for raw xi,
    or "log10_1p_xi" for ``log10(1 + xi)``.

    ``s_range=(s_min, s_max)`` or ``s_min``/``s_max`` can restrict the plotted
    total separation. For rppi data this still uses ``sqrt(rp**2 + pi**2)``.

    ``add_contour=True`` enables contour lines. ``contour_levels`` and
    ``contour_kwargs`` control the contour details.
    """
    options, plot_kwargs = _plot_options(add_contour, value, title, kwargs)
    s_power = plot_kwargs["s_power"]
    s_min, s_max = _s_limits(s_range, s_min, s_max)

    if ax is None:
        fig, ax = plt.subplots(figsize=options["figsize"])
    else:
        fig = ax.figure

    plot_items = _plot_items(
        corr2pcf1=corr2pcf1,
        corr2pcf2=corr2pcf2,
        quadrants=quadrants,
        coordinates=coordinates,
        s_power=s_power,
        value=options["value"],
    )
    plot_items = [
        _prepare_plot_item(name, x, y, z, s_min, s_max, options["value"])
        for name, x, y, z in plot_items
    ]

    vmin, vmax = _value_limits(
        plot_items=plot_items,
        vmin=plot_kwargs["vmin"],
        vmax=plot_kwargs["vmax"],
        percentile=plot_kwargs["percentile"],
        symmetric_limits=options["symmetric_limits"],
    )
    levels = np.linspace(vmin, vmax, options["n_levels"])

    _add_s_min_mask(ax, s_min, options)
    cf = _draw_plot_items(
        ax=ax,
        plot_items=plot_items,
        levels=levels,
        vmin=vmin,
        vmax=vmax,
        options=options,
        contour_levels=plot_kwargs["contour_levels"],
    )

    cbar = _add_colorbar(fig, ax, cf, options, s_power) if add_colorbar else None
    _add_plot_labels(
        ax,
        plot_items,
        quadrants,
        corr2pcf2,
        {
            "label1": label1,
            "label2": label2,
            "quadrant_labels": quadrant_labels,
        },
        options,
    )
    _style_axes(ax, options)
    return fig, ax, cbar
