"""
Comparison visualization: 2017-2022 original analysis vs 2023 extension.

Creates a publication-quality multi-panel figure comparing the two periods:
  Panel A (left, wide): Side-by-side event study plots
  Panel B (top-right):  Grouped bar chart of release-day vs surrounding-day fatalities
  Panel C (bottom-right): Summary statistics table

Reads:
  output/results_2017_2022.json
  output/results_2023.json

Writes:
  output/comparison_2017_2022_vs_2023.png
"""

import os
import sys
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

RESULTS_ORIG = os.path.join(OUTPUT_DIR, "results_2017_2022.json")
RESULTS_2023 = os.path.join(OUTPUT_DIR, "results_2023.json")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "comparison_2017_2022_vs_2023.png")

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------
# Blues for 2017-2022
BLUE_DARK = "#1b4f72"
BLUE_MED = "#2980b9"
BLUE_LIGHT = "#aed6f1"

# Orange/red for 2023
ORANGE_DARK = "#922b21"
ORANGE_MED = "#e74c3c"
ORANGE_LIGHT = "#f5b7b1"


def load_results():
    """Load the two JSON result files and return them as dicts."""
    for path, label in [(RESULTS_ORIG, "2017-2022"), (RESULTS_2023, "2023")]:
        if not os.path.isfile(path):
            print(f"ERROR: {path} not found. Run analyze_2023.py first to "
                  f"generate the {label} results.")
            sys.exit(1)

    with open(RESULTS_ORIG, "r") as f:
        orig = json.load(f)
    with open(RESULTS_2023, "r") as f:
        ext = json.load(f)

    return orig, ext


def _event_study_arrays(event_study_list):
    """Extract parallel numpy arrays from a list-of-dicts event study."""
    days = np.array([d["day"] for d in event_study_list])
    coefs = np.array([d["coef"] for d in event_study_list])
    ci_lo = np.array([d["ci_lower"] for d in event_study_list])
    ci_hi = np.array([d["ci_upper"] for d in event_study_list])
    return days, coefs, ci_lo, ci_hi


def plot_panel_a(ax, orig, ext):
    """Panel A: side-by-side event study plots for both periods."""
    days_o, coefs_o, ci_lo_o, ci_hi_o = _event_study_arrays(orig["event_study"])
    days_e, coefs_e, ci_lo_e, ci_hi_e = _event_study_arrays(ext["event_study"])

    offset = 0.15  # horizontal jitter to avoid overlap

    # 2017-2022
    err_lo_o = coefs_o - ci_lo_o
    err_hi_o = ci_hi_o - coefs_o
    ax.errorbar(
        days_o - offset, coefs_o,
        yerr=[err_lo_o, err_hi_o],
        fmt='o', markersize=5, capsize=3, capthick=1.2, linewidth=1.2,
        color=BLUE_MED, ecolor=BLUE_LIGHT, markeredgecolor=BLUE_DARK,
        label=f"2017\u20132022 (n={orig['n_albums']} albums)",
        zorder=3,
    )

    # 2023
    err_lo_e = coefs_e - ci_lo_e
    err_hi_e = ci_hi_e - coefs_e
    ax.errorbar(
        days_e + offset, coefs_e,
        yerr=[err_lo_e, err_hi_e],
        fmt='s', markersize=5, capsize=3, capthick=1.2, linewidth=1.2,
        color=ORANGE_MED, ecolor=ORANGE_LIGHT, markeredgecolor=ORANGE_DARK,
        label=f"2023 (n={ext['n_albums']} albums)",
        zorder=3,
    )

    # Reference line at zero
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", zorder=1)

    # Highlight day 0
    ax.axvline(0, color="grey", linewidth=0.6, linestyle=":", alpha=0.6, zorder=1)

    ax.set_xlabel("Day Relative to Album Release", fontsize=11)
    ax.set_ylabel("Adjusted Fatalities\n(relative to day \u22121)", fontsize=11)
    ax.set_title("A.  Event Study: Daily Fatality Coefficients", fontsize=13,
                 fontweight="bold", loc="left")
    ax.set_xticks(range(-10, 11))
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.set_xlim(-10.8, 10.8)

    # Light grid
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)


def plot_panel_b(ax, orig, ext):
    """Panel B: grouped bar chart -- release day vs surrounding days."""
    labels = ["2017\u20132022", "2023"]
    release_vals = [orig["comparison"]["release_mean"],
                    ext["comparison"]["release_mean"]]
    surround_vals = [orig["comparison"]["surrounding_mean"],
                     ext["comparison"]["surrounding_mean"]]

    x = np.arange(len(labels))
    bar_w = 0.32

    bars_surr = ax.bar(
        x - bar_w / 2, surround_vals, bar_w,
        label="Surrounding days", color=[BLUE_LIGHT, ORANGE_LIGHT],
        edgecolor=[BLUE_DARK, ORANGE_DARK], linewidth=1.0,
    )
    bars_rel = ax.bar(
        x + bar_w / 2, release_vals, bar_w,
        label="Release day", color=[BLUE_MED, ORANGE_MED],
        edgecolor=[BLUE_DARK, ORANGE_DARK], linewidth=1.0,
    )

    # Effect-size annotations
    for i, (period_data, color) in enumerate(
            [(orig, BLUE_DARK), (ext, ORANGE_DARK)]):
        c = period_data["comparison"]
        top = max(release_vals[i], surround_vals[i])
        annot_y = top + 3
        sign = "+" if c["absolute_increase"] >= 0 else ""
        ax.annotate(
            f"{sign}{c['absolute_increase']:.1f} ({sign}{c['relative_increase']:.1f}%)\n"
            f"p={c['p_value']:.4f}",
            xy=(x[i], annot_y),
            ha="center", va="bottom", fontsize=8.5, color=color,
            fontweight="bold",
        )

    ax.set_ylabel("Mean Daily Fatalities", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("B.  Release Day vs Surrounding Days", fontsize=12,
                 fontweight="bold", loc="left")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.tick_params(axis="both", labelsize=9)

    # Start y-axis a bit below the minimum to emphasise differences
    y_min = min(surround_vals) * 0.90
    y_max = max(release_vals) * 1.15
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)


def plot_panel_c(ax, orig, ext):
    """Panel C: summary statistics table."""
    ax.axis("off")
    ax.set_title("C.  Summary Statistics", fontsize=12,
                 fontweight="bold", loc="left", pad=12)

    co = orig["comparison"]
    ce = ext["comparison"]

    def fmt_p(p):
        if p < 0.0001:
            return "<0.0001"
        return f"{p:.4f}"

    row_labels = [
        "Release-day fatalities",
        "Surrounding-day fatalities",
        "Absolute increase",
        "95% CI",
        "Relative increase",
        "p-value",
        "No. of albums",
        "Mean daily fatalities",
    ]
    col_labels = ["", "2017\u20132022", "2023"]

    cell_text = [
        [row_labels[0], f"{co['release_mean']:.1f}", f"{ce['release_mean']:.1f}"],
        [row_labels[1], f"{co['surrounding_mean']:.1f}", f"{ce['surrounding_mean']:.1f}"],
        [row_labels[2], f"{co['absolute_increase']:.1f}", f"{ce['absolute_increase']:.1f}"],
        [row_labels[3],
         f"[{co['ci_lower']:.1f}, {co['ci_upper']:.1f}]",
         f"[{ce['ci_lower']:.1f}, {ce['ci_upper']:.1f}]"],
        [row_labels[4], f"{co['relative_increase']:.1f}%", f"{ce['relative_increase']:.1f}%"],
        [row_labels[5], fmt_p(co['p_value']), fmt_p(ce['p_value'])],
        [row_labels[6], str(orig['n_albums']), str(ext['n_albums'])],
        [row_labels[7],
         f"{orig['mean_daily_fatalities']:.1f}",
         f"{ext['mean_daily_fatalities']:.1f}"],
    ]

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.44, 0.28, 0.28],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.55)

    # Style header row
    for j in range(3):
        cell = table[0, j]
        cell.set_text_props(fontweight="bold", color="white")
        cell.set_facecolor("#34495e")
        cell.set_edgecolor("white")

    # Style metric-label column and data columns
    for i in range(1, len(cell_text) + 1):
        # Row label cell
        table[i, 0].set_text_props(ha="left", fontweight="semibold")
        table[i, 0].set_facecolor("#f8f9fa")
        table[i, 0].set_edgecolor("#dee2e6")
        # 2017-2022 data cell
        table[i, 1].set_facecolor("#eaf2f8")
        table[i, 1].set_edgecolor("#dee2e6")
        # 2023 data cell
        table[i, 2].set_facecolor("#fef5f4")
        table[i, 2].set_edgecolor("#dee2e6")


def print_text_summary(orig, ext):
    """Print a concise text comparison to stdout."""
    co = orig["comparison"]
    ce = ext["comparison"]

    print("=" * 72)
    print("COMPARISON: 2017-2022 (Original) vs 2023 (Extension)")
    print("=" * 72)

    header = f"{'Metric':<36} {'2017-2022':>14} {'2023':>14}"
    print(header)
    print("-" * 72)
    print(f"{'No. of albums':<36} {orig['n_albums']:>14} {ext['n_albums']:>14}")
    print(f"{'Mean daily fatalities (baseline)':<36} "
          f"{orig['mean_daily_fatalities']:>14.1f} "
          f"{ext['mean_daily_fatalities']:>14.1f}")
    print(f"{'Release-day fatalities (adjusted)':<36} "
          f"{co['release_mean']:>14.1f} {ce['release_mean']:>14.1f}")
    print(f"{'Surrounding-day fatalities (adj.)':<36} "
          f"{co['surrounding_mean']:>14.1f} {ce['surrounding_mean']:>14.1f}")
    print(f"{'Absolute increase':<36} "
          f"{co['absolute_increase']:>14.1f} {ce['absolute_increase']:>14.1f}")
    print(f"{'95% CI':<36} "
          f"{'[{:.1f}, {:.1f}]'.format(co['ci_lower'], co['ci_upper']):>14} "
          f"{'[{:.1f}, {:.1f}]'.format(ce['ci_lower'], ce['ci_upper']):>14}")
    print(f"{'Relative increase':<36} "
          f"{co['relative_increase']:>13.1f}% {ce['relative_increase']:>13.1f}%")

    def fmt_p(p):
        return "<0.0001" if p < 0.0001 else f"{p:.4f}"

    print(f"{'p-value':<36} {fmt_p(co['p_value']):>14} {fmt_p(ce['p_value']):>14}")
    print("-" * 72)

    # Directional interpretation
    if ce['absolute_increase'] > 0 and ce['p_value'] < 0.05:
        conclusion = ("The 2023 data REPLICATES the original finding: album releases "
                      "are associated with a statistically significant increase in "
                      "traffic fatalities.")
    elif ce['absolute_increase'] > 0 and ce['p_value'] >= 0.05:
        conclusion = ("The 2023 data shows a POSITIVE but non-significant effect. "
                      "The direction is consistent with the original finding, but "
                      "the effect does not reach statistical significance.")
    elif ce['absolute_increase'] <= 0:
        conclusion = ("The 2023 data does NOT replicate the original finding: "
                      "no positive association between album releases and fatalities "
                      "was observed.")
    else:
        conclusion = "Unable to draw a clear conclusion from the 2023 data."

    # Effect size comparison
    ratio = ce['absolute_increase'] / co['absolute_increase'] if co['absolute_increase'] != 0 else float('nan')
    print(f"\nEffect size ratio (2023 / 2017-2022): {ratio:.2f}")
    print(f"\nConclusion: {conclusion}")
    print("=" * 72)


def main():
    # Load data
    orig, ext = load_results()

    # Print text summary
    print_text_summary(orig, ext)

    # Build figure -----------------------------------------------------------
    fig = plt.figure(figsize=(16, 8), dpi=150, facecolor="white")

    # Layout: Panel A occupies the left 55%, Panels B & C share the right 45%
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[1.25, 1],
        height_ratios=[1, 1],
        hspace=0.38, wspace=0.30,
        left=0.06, right=0.97, top=0.92, bottom=0.08,
    )

    ax_a = fig.add_subplot(gs[:, 0])     # full left column
    ax_b = fig.add_subplot(gs[0, 1])     # top-right
    ax_c = fig.add_subplot(gs[1, 1])     # bottom-right

    plot_panel_a(ax_a, orig, ext)
    plot_panel_b(ax_b, orig, ext)
    plot_panel_c(ax_c, orig, ext)

    fig.suptitle(
        "Album Releases and Traffic Fatalities: 2017\u20132022 vs 2023",
        fontsize=15, fontweight="bold", y=0.98,
    )

    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nFigure saved to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
