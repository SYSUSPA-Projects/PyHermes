"""Replot archived numeric timing/RSS samples; no scientific jobs are run."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(data_path, output):
    data = json.loads(Path(data_path).read_text())
    methods = ["grid_cpu", "grid_gpu", "particle_cpu"]
    colors = ["#4169a1", "#16978a", "#d78935"]
    names = ["Grid CPU", "Grid CPU + GPU", "Particle CPU"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, scale, label in zip(axes,
            ["end_to_end_seconds", "sampled_peak_rss_bytes"], [1., 2**30],
            ["Preparation + measurement [s]", "Sampled process-tree RSS [GiB]"]):
        for i, method in enumerate(methods):
            for j, layout in enumerate(["4x8", "8x4"]):
                values = np.array([r[key]/scale for r in data["samples"]
                                   if r["layout"] == layout and r["implementation"] == method])
                assert len(values) == 3 and np.isfinite(values).all()
                x = j + (i-1)*0.24
                median = np.median(values)
                ax.bar(x, median, width=0.22, color=colors[i], label=names[i] if j==0 else None)
                ax.errorbar(x, median, yerr=[[median-values.min()], [values.max()-median]],
                            fmt="none", color="black", capsize=3)
                ax.scatter(x+np.array([-0.05, 0, 0.05]), values, color="black", s=14, zorder=4)
        ax.set(xticks=[0, 1], xticklabels=["4 ranks x 8 threads", "8 ranks x 4 threads"],
               ylabel=label, ylim=(0, None))
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(.5, .93), ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Corrected PyHermes: J=8, ell=0..7, (20,40), endpoint sphere R5")
    fig.text(.5, .025, "32 physical cores; median, three samples and min/max. GPU memory is separate.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .87))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.data, args.output)
