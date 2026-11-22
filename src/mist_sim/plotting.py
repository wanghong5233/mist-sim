"""Read-only training curves; SciencePlots styles, explicit seed statistics."""
import argparse
import csv
import hashlib
import io
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np


METRICS = {
    "reward_per_slot": ("平均奖励 / 时隙", "Mean reward per slot"),
    "cost_per_slot": ("平均成本 / 时隙", "Mean cost per slot"),
    "service": ("按时完成率 / %", "On-time completion rate (%)"),
    "upper_loss": ("服务器策略 PPO 损失", "Server-policy PPO loss"),
    "lower_loss": ("调度策略 PPO 损失", "Scheduling-policy PPO loss"),
}
DATA_FIELDS = ["label", "run_id", "seed", "cycle_index", "total_slots", "metric", "value"]
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#333333"]


def moving_average(values, window):
    """Trailing mean with partial initial windows; never pad or invent points."""
    values = np.asarray(values, dtype=np.float64)
    if type(window) is not int or window < 1:
        raise ValueError("smoothing_window must be a positive integer")
    prefix = np.concatenate(([0.], np.cumsum(values)))
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    return (prefix[end] - prefix[start]) / (end - start)


def aggregate(curves, window, band, bootstrap_samples=5000, bootstrap_seed=0):
    """Align exact common x coordinates; do not extend early-stopped seeds."""
    if band not in ("std", "ci95", "none"):
        raise ValueError("band must be std, ci95, or none")
    common = sorted(set.intersection(*(set(x.tolist()) for x, _ in curves)))
    if not common:
        raise ValueError("Seeds have no common x coordinates; choose matching sampling budgets")
    seed_values = []
    for x, y in curves:
        lookup = dict(zip(x.tolist(), moving_average(y, window)))
        seed_values.append([lookup[point] for point in common])
    values = np.asarray(seed_values)
    mean = values.mean(axis=0)
    low = high = None
    if len(values) > 1 and band == "std":
        spread = values.std(axis=0, ddof=1)
        low, high = mean - spread, mean + spread
    elif len(values) > 1 and band == "ci95":
        if bootstrap_samples < 100:
            raise ValueError("Use at least 100 seed bootstrap samples")
        rng = np.random.default_rng(bootstrap_seed)
        indices = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
        boot = values[indices].mean(axis=1)
        low, high = np.quantile(boot, [.025, .975], axis=0)
    return np.asarray(common), mean, low, high


def read_cycles(path, label, metrics):
    """Snapshot complete lines only when a live CSV is currently being appended."""
    path = Path(path)
    raw = path.read_bytes()
    complete = raw if raw.endswith(b"\n") else raw[:raw.rfind(b"\n") + 1]
    reader = csv.DictReader(io.StringIO(complete.decode("utf-8-sig")))
    required = {"run_id", "seed", "cycle_index", "total_slots", *metrics}
    if not required <= set(reader.fieldnames or []):
        raise ValueError(f"Missing CSV columns in {path}: {sorted(required - set(reader.fieldnames or []))}")
    result, identities, previous = [], set(), (0, -1)
    rows = list(reader)
    for row in rows:
        if None in row:
            raise ValueError(f"Malformed CSV row in {path}")
        cycle, slots = int(row["cycle_index"]), int(row["total_slots"])
        if cycle <= previous[0] or slots <= previous[1]:
            raise ValueError(f"Non-increasing cycle/slot coordinates in {path}")
        previous = cycle, slots
        identities.add((row["run_id"], int(row["seed"])))
        for metric in metrics:
            value = float(row[metric])
            if not np.isfinite(value):
                raise ValueError(f"Nonfinite {metric} in {path}")
            result.append(dict(label=label, run_id=row["run_id"], seed=int(row["seed"]),
                               cycle_index=cycle, total_slots=slots, metric=metric, value=value))
    if len(identities) != 1:
        raise ValueError(f"Expected one nonempty run/seed in {path}")
    return result, dict(path=str(path.resolve()), sha256=hashlib.sha256(complete).hexdigest(),
                       completed_iterations=len(rows), ignored_incomplete_bytes=len(raw) - len(complete))


def load_data(settings, root):
    if settings.get("data_file"):
        path = resolve(root, settings["data_file"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if settings.get("data_sha256", digest) != digest:
            raise ValueError("Frozen plot data changed; use the recorded source snapshot")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            data = list(csv.DictReader(stream))
        for row in data:
            for key in ("seed", "cycle_index", "total_slots"):
                row[key] = int(row[key])
            row["value"] = float(row["value"])
        return data, settings.get("source_provenance", [dict(path=str(path), sha256=digest, frozen=True)])
    data, sources, used = [], [], set()
    for curve in settings["curves"]:
        for name in curve["runs"]:
            path = resolve(root, name)
            path = path / "cycles.csv" if path.is_dir() else path
            if path in used:
                raise ValueError(f"Duplicate run input: {path}")
            used.add(path)
            rows, source = read_cycles(path, curve["label"], settings["metrics"])
            data.extend(rows)
            # Result CSVs carry run/seed; the original manifest supplies method
            # and source/config identity when this is the normal project layout.
            run = root / "logs" / "runs" / rows[0]["run_id"]
            for filename in ("manifest.json", "resolved_config.json"):
                item = run / filename
                if item.is_file():
                    content = json.loads(item.read_text(encoding="utf-8"))
                    source[filename] = (dict(status=content.get("status"), source_commit=content.get("source", {}).get("commit"),
                                             config_sha256=content.get("config_sha256")) if filename == "manifest.json"
                                        else dict(method=content["method"], horizon=content["horizon"],
                                                  window=content["window"], pretrain_slots=content["training"]["pretrain_slots"]))
            sources.append(source)
    return data, sources


def group_data(data, metric, coordinate):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in data:
        if row["metric"] == metric:
            if not np.isfinite(row["value"]):
                raise ValueError("Nonfinite frozen plot value")
            grouped[row["label"]][(row["run_id"], row["seed"])].append(row)
    for label, runs in grouped.items():
        seeds = [seed for _, seed in runs]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Duplicate training seed within {label}; repeated runs are not independent seeds")
        for key, rows in runs.items():
            x = np.asarray([row[coordinate] for row in rows], dtype=np.int64)
            if np.any(np.diff(x) <= 0):
                raise ValueError(f"Non-increasing plot coordinates for {key}")
    if not grouped:
        raise ValueError(f"No data for {metric}")
    return grouped


def resolve(root, path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def portable_path(path, root):
    """Keep result snapshots relocatable with their project root."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(settings, root):
    # Plotting dependencies and backend are loaded only for this explicit action.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, ticker
    import scienceplots  # noqa: F401; registers the installed MIT-licensed styles

    every = settings.get("marker_every", 20)
    if type(every) is not int or every < 1:
        raise ValueError("marker_every must be a positive integer")
    data, sources = load_data(settings, root)
    coordinate = "cycle_index" if settings["x"] == "iteration" else "total_slots"
    summaries = []
    grouped_metrics = {metric: group_data(data, metric, coordinate) for metric in settings["metrics"]}
    font_names = {font.name for font in font_manager.fontManager.ttflist}
    chinese = next((name for name in settings["chinese_fonts"] if name in font_names), None)
    if "zh" in settings["languages"] and chinese is None:
        raise ValueError("No configured Chinese font installed; install Noto Serif CJK SC or select --language en")
    output = resolve(root, settings["output"])
    output.mkdir(parents=True, exist_ok=True)
    files, aggregate_rows = [], []
    for metric, groups in grouped_metrics.items():
        series = {}
        for label, runs in groups.items():
            curves = [(np.asarray([row[coordinate] for row in rows]), np.asarray([row["value"] for row in rows]))
                      for rows in runs.values()]
            x, mean, low, high = aggregate(curves, settings["smoothing_window"], settings["band"],
                                            settings["bootstrap_samples"], settings["bootstrap_seed"])
            series[label] = x, mean, low, high, curves
            summaries.append(dict(metric=metric, label=label, seeds=len(runs), common_points=len(x),
                                  per_seed_points=[len(item[0]) for item in curves],
                                  first_x=int(x[0]), last_x=int(x[-1])))
            for index, point in enumerate(x):
                aggregate_rows.append(dict(metric=metric, label=label, x=int(point), seeds=len(runs),
                    mean=float(mean[index]), lower=float(low[index]) if low is not None else "",
                    upper=float(high[index]) if high is not None else ""))
        for language in settings["languages"]:
            zh = language == "zh"
            fonts = [settings["latin_font"]] + ([chinese] if chinese else []) + ["DejaVu Serif"]
            rc = {"font.family": fonts, "font.size": settings["font_size"], "mathtext.fontset": "stix",
                  "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path", "axes.unicode_minus": True,
                  "axes.labelsize": settings["font_size"], "xtick.labelsize": settings["font_size"] - .5,
                  "ytick.labelsize": settings["font_size"] - .5, "legend.fontsize": settings["font_size"] - 1}
            with plt.style.context(["science", "ieee", "no-latex"]), plt.rc_context(rc):
                fig, ax = plt.subplots(figsize=(settings["width_mm"] / 25.4, settings["height_mm"] / 25.4))
                rendered_lines = []
                for index, (label, (x, mean, low, high, curves)) in enumerate(series.items()):
                    style = settings.get("curve_styles", {}).get(label, {})
                    color = style.get("color", PALETTE[index % len(PALETTE)])
                    linestyle = style.get("linestyle", ["-", "--", "-.", ":"][index % 4])
                    if isinstance(linestyle, list):
                        linestyle = (linestyle[0], tuple(linestyle[1]))
                    multiplier = 100 if metric == "service" else 1
                    if settings["show_raw"]:
                        for raw_x, raw_y in curves:
                            ax.plot(raw_x, multiplier * raw_y, color=color, alpha=.22, linewidth=.6, linestyle="-", zorder=1)
                    if low is not None:
                        ax.fill_between(x, multiplier * low, multiplier * high, color=color,
                                        alpha=settings.get("band_alpha", .16), linewidth=0, zorder=2)
                    suffix = f" (n={len(curves)})" if len(curves) > 1 else ""
                    line, = ax.plot(x, multiplier * mean, label=label + suffix, color=color,
                            linewidth=style.get("line_width", settings.get("line_width", 1.35)),
                            linestyle=linestyle, marker=style.get("marker", ""),
                            markersize=settings.get("marker_size", 3.2), markerfacecolor="white",
                            markeredgewidth=.8, markevery=(style.get("marker_offset", index * 3) % every, every), zorder=3)
                    rendered_lines.append(line)
                for reference in settings["references"]:
                    if reference["metric"] == metric:
                        ax.axhline(reference["value"] * (100 if metric == "service" else 1),
                                   color="#666666", linestyle="--", linewidth=1, label=reference["label"])
                ax.set_xlabel(("训练迭代次数" if zh else "Training iteration") if settings["x"] == "iteration" else
                              ("累计训练时隙数" if zh else "Cumulative training slots"))
                ax.set_ylabel(METRICS[metric][0 if zh else 1])
                ax.grid(axis="y", color=".86", linewidth=.4, linestyle="--")
                ax.margins(x=.01, y=.09)
                if settings["x"] == "iteration":
                    if settings.get("x_tick_step") is not None:
                        ax.xaxis.set_major_locator(ticker.MultipleLocator(settings["x_tick_step"]))
                    else:
                        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
                else:
                    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
                y_axis = settings.get("y_axes", {}).get(metric)
                if y_axis:
                    ax.set_yticks(y_axis["ticks"])
                    ax.set_ylim(y_axis["limits"])
                ax.legend(loc=settings.get("legend_location", "best"),
                          bbox_to_anchor=settings.get("legend_anchor"),
                          ncol=settings.get("legend_columns", 1), handlelength=3.4, columnspacing=1.2)
                # Copy the displayed lines so the inset retains the exact same
                # custom dash patterns and marker offsets, with no resampling.
                inset = settings.get("inset", {})
                if inset.get("metric") == metric:
                    from copy import copy
                    zoom = ax.inset_axes(inset["bounds"])
                    for line in rendered_lines:
                        detail = copy(line)
                        detail.axes = None
                        detail.set_transform(zoom.transData)
                        detail.set_clip_path(zoom.patch)
                        zoom.add_line(detail)
                    zoom.set_xlim(inset["xlim"])
                    zoom.set_ylim(inset["ylim"])
                    zoom.set_xticks(inset["xticks"])
                    zoom.set_yticks(inset["yticks"])
                    zoom.tick_params(labelsize=settings["font_size"] - 2, direction="in", which="both")
                    zoom.minorticks_off()
                    zoom.grid(axis="y", color=".88", linewidth=.35, linestyle="--")
                    zoom.set_title(inset.get("title_zh" if zh else "title_en", ""),
                                   fontsize=settings["font_size"] - 1.5, pad=4)
                window = settings["smoothing_window"]
                note = ((f"{window}轮滑动均值" if zh else f"{window}-iteration moving average") if window > 1 else
                        ("未平滑" if zh else "Unsmoothed"))
                if all(len(item[4]) == 1 for item in series.values()):
                    note += ("；单种子，淡线为原始值" if zh else "; single seed, faint line: raw") if settings["show_raw"] else ""
                elif settings["band"] != "none":
                    note += ("；阴影：种子间±1标准差" if zh else "; band: +/-1 seed SD") if settings["band"] == "std" else (
                        "；阴影：种子均值95%点置信区间" if zh else "; band: pointwise 95% seed-bootstrap CI")
                if len(note) > (39 if zh else 83):
                    note = note.replace("；", "\n", 1) if zh else note.replace(";", "\n", 1)
                fig.text(.5, .015, note, ha="center", va="bottom", fontsize=settings["font_size"] - 1.5)
                fig.tight_layout(pad=.7, rect=(0, .1 if "\n" in note else .065, 1, 1))
                for extension in settings["formats"]:
                    path = output / f"{metric}_{language}.{extension}"
                    fig.savefig(path, dpi=settings["dpi"], bbox_inches=None, facecolor="white")
                    files.append(path.name)
                plt.close(fig)
    # Freeze only the values used by these figures. This remains reproducible
    # even when a live source cycles.csv gains more iterations after plotting.
    write_csv(output / "source_data.csv", DATA_FIELDS, data)
    write_csv(output / "aggregate.csv", ["metric", "label", "x", "seeds", "mean", "lower", "upper"], aggregate_rows)
    data_digest = hashlib.sha256((output / "source_data.csv").read_bytes()).hexdigest()
    frozen = dict(settings, data_file=portable_path(output / "source_data.csv", root), data_sha256=data_digest,
                  source_provenance=sources, output=portable_path(output, root))
    (output / "plot_config.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    distribution = metadata.distribution("SciencePlots")
    style_files = [distribution.locate_file(item) for item in distribution.files
                   if Path(item).name in {"science.mplstyle", "ieee.mplstyle", "no-latex.mplstyle"}]
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(), sources=sources, summary=summaries,
        source_data_sha256=data_digest,
        plotting_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependencies={name: metadata.version(name) for name in ("matplotlib", "SciencePlots", "numpy")},
        command=[sys.executable, *sys.argv], root=str(root),
        style_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in style_files},
        style=["science", "ieee", "no-latex"], chinese_font=chinese, latin_font=settings["latin_font"],
        semantics="Per completed iteration, rollout raw physical reward per slot; no validation scores or synthetic iteration zero.",
        band=settings["band"], smoothing="Trailing per-seed mean, partial initial windows, then aggregate exact common x coordinates.",
        files=files)
    (output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dict(output=str(output), figures=files, summary=summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", default="configs/plotting/convergence.json")
    parser.add_argument("--curve", action="append", help="LABEL=run directory or cycles.csv; repeat for algorithms/seeds")
    parser.add_argument("--output", help="Result figure directory, relative to root or absolute")
    parser.add_argument("--x", choices=("iteration", "total_slots"))
    parser.add_argument("--metric", action="append", choices=tuple(METRICS))
    parser.add_argument("--language", action="append", choices=("zh", "en"))
    parser.add_argument("--window", type=int)
    parser.add_argument("--band", choices=("std", "ci95", "none"))
    args = parser.parse_args()
    root = args.root.resolve()
    settings = json.loads(resolve(root, args.config).read_text(encoding="utf-8"))
    if args.curve:
        groups = defaultdict(list)
        for text in args.curve:
            if "=" not in text:
                parser.error("--curve requires LABEL=PATH")
            label, name = text.split("=", 1)
            if not label.strip() or not name.strip():
                parser.error("--curve requires nonempty label and path")
            groups[label].append(name)
        settings["curves"] = [dict(label=label, runs=runs) for label, runs in groups.items()]
        for key in ("data_file", "data_sha256", "source_provenance"):
            settings.pop(key, None)
    for argument, key in (("output", "output"), ("x", "x"), ("metric", "metrics"), ("language", "languages"),
                          ("window", "smoothing_window"), ("band", "band")):
        if getattr(args, argument) is not None:
            settings[key] = getattr(args, argument)
    if not settings.get("curves") and not settings.get("data_file"):
        parser.error("Provide --curve or a plot_config.json with frozen data")
    if not settings.get("output"):
        paths = [name for curve in settings["curves"] for name in curve["runs"]]
        if len(paths) != 1:
            parser.error("--output is required when combining multiple runs")
        path = resolve(root, paths[0])
        settings["output"] = str((path if path.is_dir() else path.parent) / "figures" / "convergence")
    if settings["x"] not in ("iteration", "total_slots") or not set(settings["metrics"]) <= set(METRICS):
        parser.error("Invalid plot x coordinate or metric")
    if not set(settings["languages"]) <= {"zh", "en"} or not set(settings["formats"]) <= {"png", "pdf", "svg"}:
        parser.error("Unsupported language or output format")
    if min(settings["width_mm"], settings["height_mm"], settings["font_size"], settings["dpi"]) <= 0:
        parser.error("Figure dimensions, fonts and dpi must be positive")
    print(json.dumps(plot_curves(settings, root), ensure_ascii=False, indent=2))
