import numpy as np


def _as_1d_array(values, name):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr


def _sampling_array(corr2pcf, name, object_name):
    if hasattr(corr2pcf, name):
        return _as_1d_array(getattr(corr2pcf, name), f"{object_name}.{name}")
    sampling = getattr(corr2pcf, "sampling", {})
    return _as_1d_array(sampling.get(name), f"{object_name}.sampling['{name}']")


def _normalize_plot_value(value):
    aliases = {
        "s_power_xi": "s_power_xi",
        "xi": "xi",
        "log10": "log10_1p_xi",
        "log10_1p_xi": "log10_1p_xi",
        "log10(1+xi)": "log10_1p_xi",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(
            "value must be 's_power_xi', 'xi', or 'log10_1p_xi'."
        ) from exc


def _plot_values(xi, radius, s_power, value):
    value = _normalize_plot_value(value)
    if value == "s_power_xi":
        return (radius**s_power) * xi
    if value == "xi":
        return xi
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(1.0 + xi)


def _plot_value_label(value, s_power):
    value = _normalize_plot_value(value)
    if value == "s_power_xi":
        return rf"$s^{s_power} \xi$"
    if value == "xi":
        return r"$\xi$"
    return r"$\log (1 + \xi)$"


def _plot_title_label(value, s_power):
    value = _normalize_plot_value(value)
    if value == "s_power_xi":
        return rf"$s^{s_power}\xi(s_\perp, s_\parallel)$"
    if value == "xi":
        return r"$\xi(s_\perp, s_\parallel)$"
    return r"$\log (1+\xi(s_\perp, s_\parallel))$"


def _normalize_plot_style(style):
    if style is None:
        return "default"
    aliases = {
        "default": "default",
        "contour": "contour",
    }
    try:
        return aliases[style]
    except KeyError as exc:
        raise ValueError("style must be None, 'default', or 'contour'.") from exc


PLOT_CORR2PCF_2D_BASE_STYLE = {
    "figsize": (6.8, 6.8),
    "label_fontsize": 18,
    "tick_fontsize": 13,
    "title_fontsize": 19,
    "text_fontsize": 14,
    "xlabel": r"$s_\perp\,(Mpc/h)$",
    "ylabel": r"$s_\parallel\,(Mpc/h)$",
    "tick_params": {
        "direction": "in",
        "top": True,
        "right": True,
    },
    "spine_linewidth": 1.0,
}


PLOT_CORR2PCF_2D_STYLES = {
    "default": {
        **PLOT_CORR2PCF_2D_BASE_STYLE,
        "value": "s_power_xi",
        "cmap": "RdBu_r",
        "n_levels": 81,
        "center_lines": True,
        "symmetric_limits": True,
        "draw_contours": False,
        "contour_kwargs": {},
        "colorbar_kwargs": {
            "shrink": 0.75,
            "fraction": 0.05,
            "pad": 0.04,
            "aspect": 25,
        },
        "colorbar_nbins": None,
    },
    "contour": {
        **PLOT_CORR2PCF_2D_BASE_STYLE,
        "value": "log10_1p_xi",
        "cmap": "plasma",
        "n_levels": 121,
        "center_lines": False,
        "symmetric_limits": False,
        "draw_contours": True,
        "contour_kwargs": {
            "colors": "yellow",
            "linewidths": 1.1,
            "linestyles": "dashdot",
        },
        "colorbar_kwargs": {
            "orientation": "horizontal",
            "shrink": 0.78,
            "fraction": 0.07,
            "pad": 0.12,
            "aspect": 32,
        },
        "colorbar_nbins": 6,
    },
}


CORNER_LABEL_BBOX = {
    "boxstyle": "round,pad=0.25",
    "facecolor": "white",
    "edgecolor": "0.75",
    "alpha": 0.95,
}


CORNER_LABEL_POSITIONS = {
    "upper_left": (0.02, 0.98, "left", "top"),
    "upper_right": (0.98, 0.98, "right", "top"),
    "lower_left": (0.02, 0.02, "left", "bottom"),
    "lower_right": (0.98, 0.02, "right", "bottom"),
}


PLOT_CORR2PCF_2D_KWARG_DEFAULTS = {
    "figsize": None,
    "cmap": None,
    "n_levels": None,
    "vmin": None,
    "vmax": None,
    "percentile": 98,
    "s_power": 2,
    "colorbar_kwargs": None,
    "label_fontsize": None,
    "tick_fontsize": None,
    "title_fontsize": None,
    "text_fontsize": None,
    "contour_levels": None,
    "center_lines": None,
}


PLOT_CORR2PCF_2D_STYLE_KWARGS = (
    "figsize",
    "cmap",
    "n_levels",
    "label_fontsize",
    "tick_fontsize",
    "title_fontsize",
    "text_fontsize",
    "center_lines",
)


def _normalize_plot_kwargs(kwargs):
    unexpected = sorted(set(kwargs) - set(PLOT_CORR2PCF_2D_KWARG_DEFAULTS))
    if unexpected:
        names = ", ".join(unexpected)
        raise TypeError(f"Unexpected plot_corr2pcf_2d keyword argument(s): {names}.")

    options = dict(PLOT_CORR2PCF_2D_KWARG_DEFAULTS)
    options.update(kwargs)
    return options


def _style_options(style):
    style = _normalize_plot_style(style)
    options = dict(PLOT_CORR2PCF_2D_STYLES[style])
    options["style"] = style
    options["colorbar_kwargs"] = dict(options["colorbar_kwargs"])
    options["contour_kwargs"] = dict(options["contour_kwargs"])
    options["tick_params"] = dict(options["tick_params"])
    return options


def _normalize_s_limits(s_range, s_min, s_max):
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


def _s_range_mask(x, y, s_min, s_max):
    radius = np.sqrt(x**2 + y**2)
    mask = np.ones(radius.shape, dtype=bool)
    if s_min is not None:
        mask &= radius >= s_min
    if s_max is not None:
        mask &= radius <= s_max
    if not np.any(mask):
        raise ValueError("s range does not include any plotted points.")
    return mask


def _prepare_plot_item(name, x, y, z, s_min, s_max, value):
    point_mask = _s_range_mask(x, y, s_min, s_max)
    z_plot = np.asarray(z, dtype=np.float64).copy()
    selected = z_plot[point_mask]
    if np.any(~np.isfinite(selected)):
        if _normalize_plot_value(value) == "log10_1p_xi":
            raise ValueError(
                "value='log10_1p_xi' requires xi > -1 in the selected s range."
            )
        raise ValueError("plot values must be finite in the selected s range.")
    z_plot[~point_mask] = 0.0
    return name, x, y, z_plot, point_mask


def _get_smu_arrays(corr2pcf_smu):
    s = _sampling_array(corr2pcf_smu, "s", "corr2pcf_smu")
    mu = _sampling_array(corr2pcf_smu, "mu", "corr2pcf_smu")
    xi = np.asarray(corr2pcf_smu.xi, dtype=np.float64)
    if xi.shape != (s.size, mu.size):
        raise ValueError(
            f"corr2pcf_smu.xi must have shape {(s.size, mu.size)}, got {xi.shape}."
        )
    if np.any(mu < 0.0) or np.any(mu > 1.0):
        raise ValueError("corr2pcf_smu.sampling['mu'] is expected to lie in [0, 1].")

    order = np.argsort(mu)
    return s, mu[order], xi[:, order]


def _get_rppi_arrays(corr2pcf_rppi):
    rp = _sampling_array(corr2pcf_rppi, "rp", "corr2pcf_rppi")
    pi = _sampling_array(corr2pcf_rppi, "pi", "corr2pcf_rppi")
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

    values = _plot_values(xi_full, S, s_power, value)
    return s_perp, s_par, values


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
    values = _plot_values(xi_full, s, s_power, value)
    return x, PI, values


def smu_to_quadrant(corr2pcf_smu, quadrant="upper_right", s_power=2, value="s_power_xi"):
    """
    Convert xi(s, mu) sampled on mu in [0, 1] to one quadrant of the
    (s_perp, s_parallel) plane.
    """
    quadrant = _normalize_quadrant_name(quadrant)
    s, mu, xi = _get_smu_arrays(corr2pcf_smu)

    S, MU = np.meshgrid(s, mu, indexing="ij")
    s_par = S * MU
    s_perp = S * np.sqrt(np.maximum(0.0, 1.0 - MU**2))

    if "left" in quadrant:
        s_perp = -s_perp
    if "lower" in quadrant:
        s_par = -s_par

    values = _plot_values(xi, S, s_power, value)
    return s_perp, s_par, values


def rppi_to_quadrant(corr2pcf_rppi, quadrant="upper_right", s_power=2, value="s_power_xi"):
    """
    Convert xi(rp, pi) sampled on pi >= 0 to one quadrant.
    """
    quadrant = _normalize_quadrant_name(quadrant)
    rp, pi, xi = _get_rppi_arrays(corr2pcf_rppi)

    RP, PI = np.meshgrid(rp, pi, indexing="ij")
    x = RP.copy()
    if "left" in quadrant:
        x = -x
    if "lower" in quadrant:
        PI = -PI

    s = np.sqrt(RP**2 + PI**2)
    values = _plot_values(xi, s, s_power, value)
    return x, PI, values


COORDINATE_CONVERTERS = {
    "smu": (smu_to_half_plane, smu_to_quadrant),
    "rppi": (rppi_to_half_plane, rppi_to_quadrant),
}


def _normalize_quadrant_name(name):
    aliases = {
        "ul": "upper_left",
        "upper-left": "upper_left",
        "upper_left": "upper_left",
        "top_left": "upper_left",
        "tl": "upper_left",
        "ur": "upper_right",
        "upper-right": "upper_right",
        "upper_right": "upper_right",
        "top_right": "upper_right",
        "tr": "upper_right",
        "ll": "lower_left",
        "lower-left": "lower_left",
        "lower_left": "lower_left",
        "bottom_left": "lower_left",
        "bl": "lower_left",
        "lr": "lower_right",
        "lower-right": "lower_right",
        "lower_right": "lower_right",
        "bottom_right": "lower_right",
        "br": "lower_right",
    }
    try:
        return aliases[name]
    except KeyError as exc:
        raise ValueError(
            "quadrant must be one of upper_left, upper_right, lower_left, lower_right."
        ) from exc


def _normalize_quadrants(quadrants):
    if isinstance(quadrants, dict):
        return {
            _normalize_quadrant_name(name): corr
            for name, corr in quadrants.items()
            if corr is not None
        }
    if len(quadrants) != 4:
        raise ValueError(
            "quadrants must be a dict or a 4-item sequence ordered as "
            "(upper_left, upper_right, lower_left, lower_right)."
        )
    names = ("upper_left", "upper_right", "lower_left", "lower_right")
    return {name: corr for name, corr in zip(names, quadrants) if corr is not None}


def _normalize_coordinates_name(name):
    if name not in COORDINATE_CONVERTERS:
        raise ValueError("coordinates must contain only 'smu' or 'rppi'.")
    return name


def _half_plane_coordinates(coordinates):
    if isinstance(coordinates, str):
        name = _normalize_coordinates_name(coordinates)
        return name, name
    if isinstance(coordinates, dict):
        left = coordinates.get("left", coordinates.get("corr2pcf1"))
        right = coordinates.get("right", coordinates.get("corr2pcf2", left))
        if left is None or right is None:
            raise ValueError(
                "coordinates dict must define left/right or corr2pcf1/corr2pcf2."
            )
        return _normalize_coordinates_name(left), _normalize_coordinates_name(right)
    if len(coordinates) != 2:
        raise ValueError(
            "coordinates must be 'smu', 'rppi', or a 2-item sequence for "
            "(corr2pcf1, corr2pcf2)."
        )
    return tuple(_normalize_coordinates_name(name) for name in coordinates)


def _quadrant_coordinates(coordinates):
    if isinstance(coordinates, str):
        name = _normalize_coordinates_name(coordinates)
        return {
            "upper_left": name,
            "upper_right": name,
            "lower_left": name,
            "lower_right": name,
        }
    if isinstance(coordinates, dict):
        return {
            _normalize_quadrant_name(quadrant): _normalize_coordinates_name(name)
            for quadrant, name in coordinates.items()
        }
    if len(coordinates) != 4:
        raise ValueError(
            "coordinates must be 'smu', 'rppi', or a 4-item sequence ordered as "
            "(upper_left, upper_right, lower_left, lower_right)."
        )
    names = ("upper_left", "upper_right", "lower_left", "lower_right")
    return {
        quadrant: _normalize_coordinates_name(name)
        for quadrant, name in zip(names, coordinates)
    }


def _combined_vmax(arrays, percentile):
    values = np.concatenate([np.abs(arr).ravel() for arr in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    vmax = np.nanpercentile(values, percentile)
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = np.nanmax(values)
    return float(vmax) if np.isfinite(vmax) and vmax > 0.0 else 1.0


def _combined_vrange(arrays, percentile):
    values = np.concatenate([arr.ravel() for arr in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0

    percentile = float(percentile)
    if percentile <= 0.0 or percentile > 100.0:
        raise ValueError("percentile must be in (0, 100].")
    lower_percentile = max(0.0, 100.0 - percentile)
    vmin = np.nanpercentile(values, lower_percentile)
    vmax = np.nanpercentile(values, percentile)

    if not np.isfinite(vmin):
        vmin = np.nanmin(values)
    if not np.isfinite(vmax):
        vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return -1.0, 1.0
    if vmin == vmax:
        delta = abs(vmax) * 0.1 if vmax != 0.0 else 1.0
        vmin -= delta
        vmax += delta
    return float(vmin), float(vmax)


def _default_contour_levels(vmin, vmax):
    return np.linspace(vmin, vmax, 7)[1:-1]


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


def _triangulation(x, y, point_mask=None):
    import matplotlib.tri as mtri

    triangles = _grid_triangles(x, y)
    triangulation = mtri.Triangulation(
        x.ravel(),
        y.ravel(),
        triangles=triangles,
    )
    if point_mask is not None:
        triangle_mask = ~np.all(point_mask.ravel()[triangles], axis=1)
        if np.all(triangle_mask):
            raise ValueError("s range does not include any complete plot cells.")
        triangulation.set_mask(triangle_mask)
    return triangulation


def _resolve_plot_options(
    style,
    value,
    title,
    s_power,
    plot_kwargs,
):
    options = _style_options(style)
    if value is not None:
        options["value"] = value
    for name in PLOT_CORR2PCF_2D_STYLE_KWARGS:
        override = plot_kwargs[name]
        if override is not None:
            options[name] = override

    options["value"] = _normalize_plot_value(options["value"])
    if plot_kwargs["colorbar_kwargs"] is not None:
        options["colorbar_kwargs"].update(plot_kwargs["colorbar_kwargs"])
    if title is None and options["style"] == "contour":
        title = _plot_title_label(options["value"], s_power)
    options["title"] = title
    return options


def _converted_plot_items(
    corr2pcf1,
    corr2pcf2,
    quadrants,
    coordinates,
    s_power,
    value,
):
    if quadrants is None:
        if corr2pcf1 is None:
            raise ValueError("corr2pcf1 is required when quadrants is not provided.")
        coordinates_left, coordinates_right = _half_plane_coordinates(coordinates)
        x_left, y_left, z_left = COORDINATE_CONVERTERS[coordinates_left][0](
            corr2pcf1,
            side="left",
            s_power=s_power,
            value=value,
        )
        if corr2pcf2 is None:
            x_right, y_right, z_right = COORDINATE_CONVERTERS[coordinates_left][0](
                corr2pcf1,
                side="right",
                s_power=s_power,
                value=value,
            )
        else:
            x_right, y_right, z_right = COORDINATE_CONVERTERS[coordinates_right][0](
                corr2pcf2,
                side="right",
                s_power=s_power,
                value=value,
            )
        return [
            ("upper_left", x_left, y_left, z_left),
            ("upper_right", x_right, y_right, z_right),
        ]

    quadrant_map = _normalize_quadrants(quadrants)
    if not quadrant_map:
        raise ValueError("quadrants must contain at least one corr2pcf object.")
    coordinates_map = _quadrant_coordinates(coordinates)
    plot_items = []
    for quadrant in ("upper_left", "upper_right", "lower_left", "lower_right"):
        corr = quadrant_map.get(quadrant)
        if corr is None:
            continue
        coordinates_name = coordinates_map.get(quadrant)
        if coordinates_name is None:
            raise ValueError(f"coordinates is missing an entry for {quadrant}.")
        x, y, z = COORDINATE_CONVERTERS[coordinates_name][1](
            corr,
            quadrant=quadrant,
            s_power=s_power,
            value=value,
        )
        plot_items.append((quadrant, x, y, z))
    return plot_items


def _prepared_plot_items(plot_items, s_min, s_max, value):
    return [
        _prepare_plot_item(name, x, y, z, s_min, s_max, value)
        for name, x, y, z in plot_items
    ]


def _plot_value_arrays(plot_items):
    return [z[point_mask] for _, _, _, z, point_mask in plot_items]


def _resolve_value_limits(plot_items, vmin, vmax, percentile, symmetric_limits):
    plot_value_arrays = _plot_value_arrays(plot_items)
    if symmetric_limits and vmin is None:
        if vmax is None:
            vmax = _combined_vmax(plot_value_arrays, percentile)
        vmin = -vmax
    else:
        auto_vmin, auto_vmax = _combined_vrange(plot_value_arrays, percentile)
        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax

    if vmin >= vmax:
        raise ValueError("vmin must be smaller than vmax.")
    return vmin, vmax


def _add_s_min_mask(ax, s_min, options):
    if options["style"] != "contour" or s_min is None:
        return

    import matplotlib.pyplot as plt

    circle = plt.Circle(
        (0.0, 0.0),
        s_min,
        facecolor="white",
        # facecolor=plt.get_cmap(options["cmap"])(1.0),
        edgecolor="none",
        zorder=0.5,
    )
    ax.add_patch(circle)


def _draw_contours(ax, triangulation, z, vmin, vmax, options, contour_levels):
    if not options["draw_contours"] and contour_levels is None:
        return

    levels = (
        _default_contour_levels(vmin, vmax)
        if contour_levels is None
        else contour_levels
    )
    cs = ax.tricontour(
        triangulation,
        z.ravel(),
        levels=levels,
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


def _draw_plot_items(ax, plot_items, levels, vmin, vmax, options, contour_levels):
    cf = None
    for _, x, y, z, point_mask in plot_items:
        triangulation = _triangulation(x, y, point_mask=point_mask)
        cf = ax.tricontourf(
            triangulation,
            z.ravel(),
            levels=levels,
            cmap=options["cmap"],
            extend="both",
        )
        _draw_contours(ax, triangulation, z, vmin, vmax, options, contour_levels)
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

    if options["colorbar_nbins"] is not None:
        import matplotlib.ticker as mticker

        cbar.locator = mticker.MaxNLocator(nbins=options["colorbar_nbins"])
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


def _add_half_plane_labels(ax, quadrants, corr2pcf2, label1, label2, options):
    if quadrants is not None:
        return
    if corr2pcf2 is None:
        if label1:
            _add_corner_label(ax, label1, "upper_right", options)
        return

    if label1:
        _add_corner_label(ax, label1, "upper_left", options)
    if label2:
        _add_corner_label(ax, label2, "upper_right", options)


def _add_quadrant_labels(ax, plot_items, quadrants, quadrant_labels, options):
    if quadrants is None or not quadrant_labels:
        return

    plotted_quadrants = {item[0] for item in plot_items}
    for name, text in quadrant_labels.items():
        quadrant = _normalize_quadrant_name(name)
        if quadrant not in plotted_quadrants:
            continue
        _add_corner_label(ax, text, quadrant, options)


def _style_axes(ax, options):
    if options["center_lines"]:
        ax.axvline(0.0, color="k", lw=0.8, alpha=0.5)
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)

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
    style=None,
    value=None,
    s_range=None,
    s_min=None,
    s_max=None,
    quadrant_labels=None,
    label1=None,
    label2=None,
    title=None,
    ax=None,
    colorbar=True,
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

    ``style="contour"`` switches to a single-panel contour style with a
    plasma colormap, horizontal colorbar, yellow dash-dot contours, and
    projected-separation axis labels.

    Extra plotting controls are passed as keyword arguments: ``figsize``,
    ``cmap``, ``n_levels``, ``vmin``, ``vmax``, ``percentile``, ``s_power``,
    ``colorbar_kwargs``, font sizes, ``contour_levels``, and ``center_lines``.
    """
    import matplotlib.pyplot as plt

    plot_kwargs = _normalize_plot_kwargs(kwargs)
    s_power = plot_kwargs["s_power"]
    options = _resolve_plot_options(
        style=style,
        value=value,
        title=title,
        s_power=s_power,
        plot_kwargs=plot_kwargs,
    )
    s_min, s_max = _normalize_s_limits(s_range, s_min, s_max)

    if ax is None:
        fig, ax = plt.subplots(figsize=options["figsize"])
    else:
        fig = ax.figure

    plot_items = _converted_plot_items(
        corr2pcf1=corr2pcf1,
        corr2pcf2=corr2pcf2,
        quadrants=quadrants,
        coordinates=coordinates,
        s_power=s_power,
        value=options["value"],
    )
    plot_items = _prepared_plot_items(plot_items, s_min, s_max, options["value"])

    vmin, vmax = _resolve_value_limits(
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

    cbar = None
    if colorbar:
        cbar = _add_colorbar(fig, ax, cf, options, s_power)

    _add_half_plane_labels(ax, quadrants, corr2pcf2, label1, label2, options)
    _add_quadrant_labels(ax, plot_items, quadrants, quadrant_labels, options)
    _style_axes(ax, options)
    return fig, ax, cbar
