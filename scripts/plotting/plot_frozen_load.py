"""Plot completed frozen-policy load tests with shared paper styles."""
import argparse
from collections import defaultdict
import csv
from importlib import metadata
import json
from pathlib import Path
import sys

import numpy as np

from scripts.evaluation.evaluate_frozen import ROOT, aggregate_group, digest, read_json, write_csv, write_json


METHODS = {
    "hpgi": "HPGI", "fixed_split": "Fixed-Split", "single_timescale": "Single-Timescale",
    "always_on": "Always-On", "fixed_window": "Fixed-Window", "upper_ppo_greedy": "Upper-PPO+Greedy",
    "greedy_consolidation": "Greedy",
}
LOAD_ORDER = ("low", "middle", "high")
PANELS = {
    "cost_per_slot": ("综合成本", "Composite cost", "成本 / 时隙", "Cost per slot", 1.),
    "energy_per_slot": ("实际能耗", "Actual energy", "能耗 / (J/时隙)", "Energy (J/slot)", 1.),
    "on_time_rate": ("全部到达请求的按时率", "Service across all arrivals", "按时完成率 / %", "On-time completion (%)", 100.),
    "on_time_rate_published": ("已发布请求的按时率", "Service among released requests", "按时完成率 / %", "On-time completion (%)", 100.),
}


def pooled_interval(rows, metric, replicates, seed):
    rows = sorted(rows, key=lambda row: row["seed"])
    if metric in ("cost_per_slot", "energy_per_slot"):
        name = "conservative_cost" if metric == "cost_per_slot" else "total_energy"
        numerators = np.array([row["metrics"][name] for row in rows], dtype=float)
        denominators = np.array([row["slots"] for row in rows], dtype=float)
    else:
        numerators = np.array([row["metrics"]["completed"] for row in rows], dtype=float)
        denominators = np.array([row["metrics"]["arrivals"] -
            (row["metrics"]["suppressed"] if metric == "on_time_rate_published" else 0) for row in rows], dtype=float)
    if not np.all(denominators > 0):
        raise ValueError("Cannot bootstrap a ratio with an empty trajectory denominator")
    indices = np.random.default_rng(seed).integers(0, len(rows), (replicates, len(rows)))
    resampled = numerators[indices].sum(1) / denominators[indices].sum(1)
    lower, upper = np.quantile(resampled, [.025, .975])
    return float(numerators.sum() / denominators.sum()), float(lower), float(upper)


def load_plot_data(directory):
    provenance = read_json(directory / "provenance.json")
    specification = read_json(provenance["spec"])
    if digest(provenance["spec"]) != provenance["spec_sha256"]:
        raise ValueError("Evaluation specification changed")
    summary_path = directory / "summary.csv"
    if digest(summary_path) != provenance["result_sha256"]["summary.csv"]:
        raise ValueError("Evaluation summary changed")
    with summary_path.open(encoding="utf-8", newline="") as stream:
        summaries = {(row["load"], row["decode"], row["method"]): row for row in csv.DictReader(stream)}
    expected = {task["id"]: task for task in specification["tasks"] if task["batch"] in ("A", "B")}
    groups = defaultdict(list)
    seen = set()
    for name, expected_hash in provenance["sources"].items():
        if digest(name) != expected_hash:
            raise ValueError("Evaluation observations changed")
        with Path(name).open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["id"] in seen or row["id"] not in expected:
                    raise ValueError("Unexpected or duplicate evaluation observation")
                seen.add(row["id"])
                task = expected[row["id"]]
                if row["slots"] != 128 or row["input_sha256"] != task["input"]["sha256"]:
                    raise ValueError("Incomplete or mismatched evaluation input")
                if not row["frozen_state_verified"] or not row["physics"]["constraints_and_kkt"]:
                    raise ValueError("Unverified evaluation observation")
                if row["method"] in METHODS and row["decode"] == "deterministic":
                    if row["information"] != "robust":
                        raise ValueError("Main load comparison requires the same information model")
                    groups[row["load"], row["method"]].append(row)
    if seen != set(expected):
        raise ValueError("Evaluation suite is incomplete")
    points, load_stats = [], []
    for load in LOAD_ORDER:
        reference = sorted((row["seed"], row["input_sha256"]) for row in groups[load, "hpgi"])
        for method, label in METHODS.items():
            rows = groups[load, method]
            if len(rows) != 20 or sorted((row["seed"], row["input_sha256"]) for row in rows) != reference:
                raise ValueError("Methods must share twenty complete trajectories at each load")
            pooled = aggregate_group(rows)
            summary = summaries[load, "deterministic", method]
            for metric in PANELS:
                estimate, lower, upper = pooled_interval(rows, metric, specification["bootstrap_replicates"],
                                                         specification["bootstrap_seed"])
                if not np.isclose(estimate, float(summary[metric]), rtol=1e-12, atol=1e-12):
                    raise ValueError("Plot aggregation differs from the completed evaluation summary")
                task = expected[rows[0]["id"]]
                points.append({"load": load, "method": method, "label": label, "metric": metric,
                    "arrival_probability": task["probability"],
                    "expected_arrivals": specification["methods"][method]["config"]["terminals"] * task["probability"],
                    "estimate": estimate, "ci95_low": lower, "ci95_high": upper, "trajectories": len(rows),
                    "slots": pooled["slots"], "actual_arrivals_per_slot": pooled["arrivals_per_slot"]})
            if method == "hpgi":
                load_stats.append({"load": load, "suppression_rate": pooled["suppressed_rate"],
                                   "actual_arrivals_per_slot": pooled["arrivals_per_slot"]})
    return points, load_stats, provenance


def plot(directory, config, output, language, errorbars=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, ticker
    import scienceplots

    points, load_stats, source = load_plot_data(directory)
    settings = read_json(config)
    settings["curve_styles"]["Greedy"] = {"color": "#343434", "linestyle": [0, [4, 1.5, 1, 1.5]], "marker": "P"}
    available = {font.name for font in font_manager.fontManager.ttflist}
    chinese = next((name for name in settings["chinese_fonts"] if name in available), None)
    if language == "zh" and chinese is None:
        raise ValueError("A configured Chinese font is required")
    fonts = [settings["latin_font"]] + ([chinese] if chinese else []) + ["DejaVu Serif"]
    rc = {"font.family": fonts, "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
          "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9, "mathtext.fontset": "stix",
          "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True}
    output.mkdir(parents=True, exist_ok=True)
    zh = language == "zh"
    with plt.style.context(["science", "ieee", "no-latex"]), plt.rc_context(rc):
        figure, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 142 / 25.4))
        figure.subplots_adjust(left=.085, right=.975, top=.845, bottom=.16, wspace=.30, hspace=.40)
        handles = []
        for panel_index, (axis, (metric, labels)) in enumerate(zip(axes.flat, PANELS.items())):
            for method, label in METHODS.items():
                selected = [row for load in LOAD_ORDER for row in points
                            if row["load"] == load and row["method"] == method and row["metric"] == metric]
                multiplier = labels[4]
                values = np.array([row["estimate"] for row in selected]) * multiplier
                lower = np.array([row["ci95_low"] for row in selected]) * multiplier
                upper = np.array([row["ci95_high"] for row in selected]) * multiplier
                style = settings["curve_styles"][label]
                linestyle = style["linestyle"]
                if isinstance(linestyle, list):
                    linestyle = (linestyle[0], tuple(linestyle[1]))
                line, = axis.plot([row["expected_arrivals"] for row in selected], values, label=label,
                    color=style["color"], linestyle=linestyle, linewidth=style.get("line_width", 1.25),
                    marker=style["marker"], markersize=4.3, markerfacecolor="white", markeredgewidth=.95,
                    zorder=5 if method == "hpgi" else 3)
                if errorbars:
                    axis.errorbar([row["expected_arrivals"] for row in selected], values,
                        yerr=[np.maximum(0, values - lower), np.maximum(0, upper - values)], fmt="none",
                        color=style["color"], elinewidth=.7, capsize=2, capthick=.7, zorder=2)
                if panel_index == 0:
                    handles.append(line)
            axis.set_title(f"({chr(97 + panel_index)}) {labels[0 if zh else 1]}", loc="left", pad=8)
            axis.set_ylabel(labels[2 if zh else 3])
            axis.set_xticks([10, 20, 30])
            axis.set_xlim(8.8, 31.2)
            axis.xaxis.set_minor_locator(ticker.NullLocator())
            axis.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            axis.grid(axis="y", color=".88", linewidth=.45, linestyle="--", zorder=0)
            axis.set_xlabel("期望到达数 / (请求/时隙)" if zh else "Expected arrivals (requests/slot)")
            if metric == "energy_per_slot":
                axis.set_ylim(0, 190)
            elif metric == "cost_per_slot":
                axis.set_ylim(45, 275)
            elif metric == "on_time_rate":
                axis.set_ylim(35, 100)
            else:
                axis.set_ylim(60, 102)
        figure.legend(handles, list(METHODS.values()), loc="upper center", bbox_to_anchor=(.51, .99),
                      ncol=4, frameon=False, handlelength=3.1, columnspacing=1.4, labelspacing=.8)
        suppression = " / ".join(f"{100 * row['suppression_rate']:.2f}%" for row in load_stats)
        note = (f"低 / 中 / 高负载的匿名抑制率：{suppression}。" if zh else
                f"Anonymity suppression, low / medium / high load: {suppression}.")
        uncertainty = ("固定权重、确定性决策；每点20条独立请求轨迹，误差棒为轨迹bootstrap 95%区间。" if zh else
                       "Frozen policies, deterministic actions; 20 request traces per point, 95% trace-bootstrap intervals.")
        if not errorbars:
            uncertainty = ("固定权重、确定性决策；每点20条独立请求轨迹，置信区间另存数据表。" if zh else
                           "Frozen policies, deterministic actions; 20 request traces per point; intervals saved in data.")
        figure.text(.52, .065, note, ha="center", fontsize=8.5)
        figure.text(.52, .03, uncertainty, ha="center", fontsize=8)
        files = []
        for extension in ("png", "pdf"):
            path = output / f"load_comparison_{language}.{extension}"
            figure.savefig(path, dpi=settings["dpi"], facecolor="white")
            files.append(path)
        plt.close(figure)
    write_csv(output / "plot_data.csv", points)
    write_json(output / "plot_config.json", {"base_config": str(config), "base_sha256": digest(config),
        "styles": settings["curve_styles"], "language": language, "errorbars": errorbars,
        "width_mm": 180, "height_mm": 142, "font_size": 10, "dpi": settings["dpi"], "fonts": fonts,
        "decode": "deterministic", "smoothing": None, "load_stats": load_stats})
    write_json(output / "provenance.json", {"source_summary": str(directory / "summary.csv"),
        "summary_sha256": digest(directory / "summary.csv"), "evaluation": source,
        "plot_script_sha256": digest(__file__), "data_sha256": digest(output / "plot_data.csv"),
        "plot_config_sha256": digest(output / "plot_config.json"),
        "software": {name: metadata.version(name) for name in ("matplotlib", "numpy", "SciencePlots")},
        "output_sha256": {path.name: digest(path) for path in files},
        "caption": note + " " + uncertainty,
        "interpretation": "Three observed load conditions, straight connecting segments only. No training-seed confidence intervals."})
    print("\n".join(str(path) for path in files))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "results/comparisons/paper-evaluation-v1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/plotting/convergence.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--no-errorbars", action="store_true")
    args = parser.parse_args()
    plot(args.summary.resolve(), args.config.resolve(),
         (args.output or args.summary / "figures/load").resolve(), args.language, not args.no_errorbars)


if __name__ == "__main__":
    main()
