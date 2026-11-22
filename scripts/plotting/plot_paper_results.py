"""Build the paper's frozen-policy figures and tables from verified observations."""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
from html import escape
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from scripts.evaluation.evaluate_frozen import (
    ROOT, aggregate_group, anonymity_metrics, digest, extract_source, paired_intervals, read_json, source_signature,
    source_tree, write_csv, write_json,
)
from scripts.plotting.plot_frozen_load import METHODS, PANELS, pooled_interval


INFORMATION = {"hpgi": "HPGI", "nominal": "Nominal-Anonymity", "full_information": "Full-Information"}
MODEL_NAMES = {"resnet50": "ResNet50", "resnet152": "ResNet152", "vit_b_32": "ViT-B32"}
DEFAULT_SPEC = ROOT / "logs/launchers/paper-completion-v1/spec.json"
DEFAULT_OUTPUT = ROOT / "results/comparisons/paper-completion-v1"


def load_batch(spec_path, batch):
    specification = read_json(spec_path)
    status_path = ROOT / "logs/runs" / f"{spec_path.parent.name}-{batch}" / "status.json"
    status = read_json(status_path)
    if status["status"] != "completed" or status["spec_sha256"] != digest(spec_path):
        raise ValueError(f"Incomplete or changed batch: {batch}")
    path = ROOT / "results" / f"{spec_path.parent.name}-{batch}" / "observations.jsonl"
    if digest(path) != status["observations_sha256"]:
        raise ValueError("Observation checksum changed")
    expected = {task["id"]: task for task in specification["tasks"] if task["batch"] == batch}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != len(expected) or {row["id"] for row in rows} != set(expected):
        raise ValueError("Missing or duplicate cells")
    checked_files = {}
    for row in rows:
        task = expected[row["id"]]
        method = specification["methods"][task["method"]]
        for file_path, expected_hash in ((task["input"]["path"], task["input"]["sha256"]),
                                         (method["checkpoint"], method["checkpoint_sha256"])):
            if file_path:
                if file_path not in checked_files:
                    checked_files[file_path] = digest(file_path)
                if checked_files[file_path] != expected_hash:
                    raise ValueError("Frozen input or checkpoint changed")
        if (row["method"] != task["method"] or row["seed"] != task["input"]["seed"] or
                row["information"] != method["config"]["information"] or
                row["decode"] != ("deterministic" if task["deterministic"] else "stochastic") or
                row["slots"] != 128 or row["input_sha256"] != task["input"]["sha256"] or
                row["checkpoint_sha256"] != method["checkpoint_sha256"] or
                not row["frozen_state_verified"] or not row["physics"]["constraints_and_kkt"]):
            raise ValueError("Unverified physical result or source")
        counts = row["metrics"]
        if counts["arrivals"] != sum(counts[name] for name in
                ("completed", "suppressed", "rejected", "timeout", "resource_failed")):
            raise ValueError("Request conservation failed")
        if not np.isclose(counts["total_energy"], sum(counts["energy"].values())):
            raise ValueError("Energy accounting failed")
        row.setdefault("axis", "load")
        row.setdefault("axis_value", method["config"]["terminals"] * task["probability"])
        if (row["axis"] != task.get("axis", "load") or
                row["axis_value"] != task.get("axis_value", method["config"]["terminals"] * task["probability"]) or
                row.get("config_overrides", {}) != task.get("overrides", {})):
            raise ValueError("Observation belongs to another parameter condition")
    return rows, {str(path): digest(path), str(status_path): digest(status_path)}


def paired_groups(rows, methods, values, axis):
    groups = defaultdict(list)
    for row in rows:
        if row["decode"] == "deterministic" and row["method"] in methods:
            if row["axis"] == axis and row["axis_value"] in values:
                groups[row["axis_value"], row["method"]].append(row)
    for value in values:
        reference = sorted((row["seed"], row["input_sha256"]) for row in groups[value, "hpgi"])
        if len(set(reference)) != 20 or len({seed for seed, checksum in reference}) != 20:
            raise ValueError("Twenty distinct paired traces required")
        for method in methods:
            group = groups[value, method]
            if len(group) != 20 or sorted((row["seed"], row["input_sha256"]) for row in group) != reference:
                raise ValueError(f"Twenty paired traces required: {axis}, {value}, {method}")
    return groups


def interval(rows, metric, replicates=5000, seed=0):
    if metric in PANELS:
        return pooled_interval(rows, metric, replicates, seed)
    rows = sorted(rows, key=lambda row: row["seed"])
    if metric == "timeout_accepted":
        numerators = [row["metrics"]["timeout"] for row in rows]
        denominators = [row["metrics"]["arrivals"] - row["metrics"]["suppressed"] - row["metrics"]["rejected"] for row in rows]
    elif metric in ("edge_per_slot", "bandwidth_utilization"):
        numerators = [sum(row["mechanism"]["splits"].values()) for row in rows]
        denominators = [row["slots"] * (2 * row["axis_value"] / 10 if metric == "bandwidth_utilization" else 1) for row in rows]
    elif metric == "suppression_rate":
        numerators = [row["metrics"]["suppressed"] for row in rows]
        denominators = [row["metrics"]["arrivals"] for row in rows]
    elif metric == "information_width":
        numerators = [(row["anonymity"]["normalized_position_width_sum"] +
                       row["anonymity"]["normalized_deadline_width_sum"]) / 2 for row in rows]
        denominators = [row["anonymity"]["published"] for row in rows]
    else:
        raise ValueError(metric)
    numerators, denominators = np.array(numerators, float), np.array(denominators, float)
    if np.any(denominators <= 0):
        raise ValueError("Undefined trajectory ratio")
    indices = np.random.default_rng(seed).integers(0, len(rows), (replicates, len(rows)))
    samples = numerators[indices].sum(1) / denominators[indices].sum(1)
    return float(numerators.sum() / denominators.sum()), *map(float, np.quantile(samples, [.025, .975]))


def paired_cost_interval(reference, comparison, replicates=5000, seed=0):
    reference = sorted(reference, key=lambda row: row["seed"])
    comparison = sorted(comparison, key=lambda row: row["seed"])
    if [(row["seed"], row["input_sha256"], row["slots"]) for row in reference] != [
            (row["seed"], row["input_sha256"], row["slots"]) for row in comparison]:
        raise ValueError("Cost comparison requires paired episodes")
    differences = np.array([first["metrics"]["conservative_cost"] - second["metrics"]["conservative_cost"]
                            for first, second in zip(reference, comparison)])
    slots = np.array([row["slots"] for row in reference])
    indices = np.random.default_rng(seed).integers(0, len(reference), (replicates, len(reference)))
    samples = differences[indices].sum(1) / slots[indices].sum(1)
    return float(differences.sum() / slots.sum()), *map(float, np.quantile(samples, [.025, .975]))


def setup_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import scienceplots
    settings = read_json(ROOT / "configs/plotting/convergence.json")
    settings["curve_styles"]["Greedy"] = {"color": "#343434", "linestyle": [0, [4, 1.5, 1, 1.5]], "marker": "P"}
    information_styles = read_json(ROOT / "results/comparisons/information-baselines/figures/convergence/plot_config.json")["curve_styles"]
    settings["curve_styles"]["Nominal-Anonymity"] = information_styles["Nominal-Anonymity"]
    settings["curve_styles"]["Full-Information"] = information_styles["Full-Information（理想参照）"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    chinese = next((name for name in settings["chinese_fonts"] if name in available), None)
    if chinese is None:
        raise ValueError("A configured Chinese font is required")
    plt.style.use(["science", "ieee", "no-latex"])
    plt.rcParams.update({"font.family": [settings["latin_font"], chinese, "DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10, "legend.fontsize": 8.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "mathtext.fontset": "stix", "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.unicode_minus": True})
    return plt, settings


def line_style(settings, label):
    style = settings["curve_styles"][label]
    linestyle = style["linestyle"]
    if isinstance(linestyle, list):
        linestyle = (linestyle[0], tuple(linestyle[1]))
    return {"color": style["color"], "linestyle": linestyle, "marker": style["marker"],
            "linewidth": style.get("line_width", 1.25), "markersize": 4.2,
            "markerfacecolor": "white", "markeredgewidth": .95}


def save_figure(figure, output, name, points, settings, caption):
    directory = output / "figures" / name
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(directory / f"{name}_zh.{extension}", dpi=settings["dpi"], facecolor="white")
    write_csv(directory / "plot_data.csv", points)
    write_json(directory / "plot_config.json", {"styles": settings["curve_styles"], "dpi": settings["dpi"],
        "bar_fill_alpha": settings.get("bar_fill_alpha", 1.0),
        "caption": caption, "smoothing": None, "bootstrap_replicates": 5000, "bootstrap_seed": 0,
        "axes": [{"title": axis.get_title(loc="left"), "ylim": list(axis.get_ylim()),
                  "yscale": axis.get_yscale()} for axis in figure.axes],
        "uncertainty": "Paired request-trajectory bootstrap, not training-seed uncertainty"})


def metric_figure(plt, settings, groups, methods, values, axis_name, panels, output, name, context,
                  *, y_limits=None):
    from matplotlib import ticker
    figure, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 141 / 25.4))
    figure.subplots_adjust(left=.09, right=.975, top=.845, bottom=.15, wspace=.31, hspace=.43)
    points, handles = [], []
    for panel_index, (axis, (metric, title, ylabel, scale)) in enumerate(zip(axes.flat, panels)):
        plotted_methods = methods if metric != "gateway" else {
            "suppression_rate": "抑制率", "information_width": "泛化区间平均宽度"}
        for method, label in plotted_methods.items():
            source_method = method if metric != "gateway" else "hpgi"
            source_metric = metric if metric != "gateway" else method
            estimates = np.array([interval(groups[value, source_method], source_metric) for value in values]) * scale
            style_label = label if metric != "gateway" else ("Greedy" if method == "suppression_rate" else "Fixed-Window")
            style = line_style(settings, style_label)
            line, = axis.plot(values, estimates[:, 0], label=label, zorder=5 if method == "hpgi" else 3, **style)
            axis.errorbar(values, estimates[:, 0], yerr=[np.maximum(0, estimates[:, 0] - estimates[:, 1]),
                np.maximum(0, estimates[:, 2] - estimates[:, 0])], fmt="none", color=style["color"],
                elinewidth=.7, capsize=2, capthick=.7, zorder=2)
            if panel_index == 0:
                handles.append(line)
            for value, estimates_at_value in zip(values, estimates / scale):
                points.append(dict(axis=axis_name, value=value, method=source_method, metric=source_metric,
                    estimate=estimates_at_value[0], ci95_low=estimates_at_value[1], ci95_high=estimates_at_value[2]))
        axis.set_title(f"({chr(97 + panel_index)}) {title}", loc="left", pad=8)
        axis.set_xlabel(axis_name)
        axis.set_ylabel(ylabel)
        axis.set_xticks(values)
        axis.xaxis.set_minor_locator(ticker.NullLocator())
        axis.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        axis.grid(axis="y", color=".88", linewidth=.45, linestyle="--")
        if metric == "gateway":
            axis.legend(loc="best", frameon=False, fontsize=8)
            axis.set_ylim(0, 102)
        if metric in ("on_time_rate", "on_time_rate_published", "bandwidth_utilization"):
            axis.set_ylim(0, 102)
        elif metric == "timeout_accepted":
            axis.set_ylim(bottom=-.4)
            if all(interval(groups[value, method], metric)[0] == 0 for value in values
                   for method in ("hpgi", "full_information")):
                axis.text(.5, .09, "HPGI与Full-Information均为0", transform=axis.transAxes,
                          ha="center", fontsize=8)
        else:
            axis.set_ylim(bottom=0)
        if y_limits and metric in y_limits:
            axis.set_ylim(*y_limits[metric])
    figure.legend(handles, list(methods.values()), loc="upper center", bbox_to_anchor=(.52, .99),
                  ncol=4 if len(methods) > 4 else 3, frameon=False, handlelength=3, columnspacing=1.25)
    caption = "固定权重、确定性决策；每点20条独立请求轨迹，误差棒为轨迹bootstrap 95%区间。"
    figure.text(.52, .058, context, ha="center", fontsize=8.2)
    figure.text(.52, .022, caption, ha="center", fontsize=8)
    save_figure(figure, output, name, points, settings, context + caption)
    plt.close(figure)
    return points


def mechanism_figures(plt, settings, central, output):
    groups = paired_groups(central, METHODS, [20], "load")
    pooled = {method: aggregate_group(groups[20, method]) for method in METHODS}
    labels = list(METHODS.values())
    positions = np.arange(len(METHODS))
    figure = plt.figure(figsize=(180 / 25.4, 123 / 25.4))
    grid = figure.add_gridspec(2, 2, width_ratios=[1.6, 1], left=.22, right=.975, top=.86,
                             bottom=.17, wspace=.32, hspace=.6)
    energy_axis = figure.add_subplot(grid[:, 0])
    active_axis = figure.add_subplot(grid[0, 1])
    switch_axis = figure.add_subplot(grid[1, 1])
    bottom = np.zeros(len(METHODS))
    components = [("static", "服务器基础能耗", "#9BA7B0"), ("wake", "唤醒", "#C9946E"),
                  ("local", "端侧计算", "#5E86A6"), ("transmit", "传输", "#C0AD57"),
                  ("edge", "边缘计算", "#896D9C")]
    points = []
    for component, label, color in components:
        values = [pooled[method][f"energy_{component}_per_slot"] for method in METHODS]
        energy_axis.barh(positions, values, left=bottom, color=color, label=label, height=.65,
                         edgecolor="white", linewidth=.35)
        bottom += values
        points.extend(dict(method=method, metric=f"energy_{component}_per_slot", value=value)
                      for method, value in zip(METHODS, values))
    energy_axis.set_yticks(positions, labels)
    energy_axis.invert_yaxis()
    energy_axis.set_xlabel("能耗 / (J/时隙)")
    energy_axis.set_title("(a) 实际能耗分解", loc="left", pad=10)
    short_labels = ["HPGI", "FS", "ST", "AO", "FW", "UPG", "Greedy"]
    colors = [settings["curve_styles"][label]["color"] for label in labels]
    for axis, numerator, title, ylabel, scale in (
            (active_axis, "active_server_slots", "(b) 活跃服务器", "平均台数", 1),
            (switch_axis, "switches", "(c) 启停切换", "次数 / 128时隙", 128)):
        values = [pooled[method][numerator] / pooled[method]["slots"] * scale for method in METHODS]
        axis.bar(positions, values, color=colors, width=.7)
        axis.set_xticks(positions, short_labels, rotation=40, ha="right", fontsize=7.5)
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", pad=9)
        axis.grid(axis="y", color=".9", linewidth=.4, linestyle="--")
        axis.set_axisbelow(True)
        points.extend(dict(method=method, metric=numerator + "_per_slot", value=value / scale)
                      for method, value in zip(METHODS, values))
    figure.legend(*energy_axis.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.55, .97),
                  ncol=5, frameon=False, columnspacing=1)
    caption = "同一中负载测试集；各方法服务水平见表T2，能耗包含端侧、传输及服务器运行与唤醒开销。"
    figure.text(.52, .05, caption, ha="center", fontsize=8.2)
    save_figure(figure, output, "F6_energy_mechanism", points, settings, caption)
    plt.close(figure)
    models, splits = {}, {}
    for method in METHODS:
        models[method], splits[method] = defaultdict(Counter), Counter()
        for row in groups[20, method]:
            for model, counts in row["mechanism"]["models"].items():
                models[method][model].update(counts)
            splits[method].update(row["mechanism"]["splits"])
    figure = plt.figure(figsize=(180 / 25.4, 139 / 25.4))
    grid = figure.add_gridspec(2, 3, left=.085, right=.98, top=.84, bottom=.15, hspace=.6, wspace=.4)
    service_axis = figure.add_subplot(grid[0, :])
    points = []
    positions = np.arange(3)
    width = .105
    from matplotlib.colors import to_rgba
    model_settings = {**settings, "bar_fill_alpha": .72}
    for method_index, (method, label) in enumerate(METHODS.items()):
        values = [100 * models[method][model]["completed"] / models[method][model]["arrivals"] for model in MODEL_NAMES]
        color = settings["curve_styles"][label]["color"]
        service_axis.bar(positions + (method_index - 3) * width, values, width=width * .78, label=label,
                         color=to_rgba(color, model_settings["bar_fill_alpha"]), edgecolor=color, linewidth=.5)
        points.extend(dict(method=method, model=model, metric="on_time_rate", split="", value=value / 100)
                      for model, value in zip(MODEL_NAMES, values))
    service_axis.set_xticks(positions, list(MODEL_NAMES.values()))
    service_axis.set_ylim(0, 102)
    service_axis.set_ylabel("全部到达按时率 / %")
    service_axis.set_title("(a) 混合业务中的分模型服务", loc="left", pad=8)
    service_axis.grid(axis="y", color=".9", linewidth=.4, linestyle="--")
    service_axis.set_axisbelow(True)
    for model_index, (model, model_label) in enumerate(MODEL_NAMES.items()):
        axis = figure.add_subplot(grid[1, model_index])
        choices = sorted({int(key.split(":")[1]) for method in ("hpgi", "fixed_split")
                          for key in splits[method] if key.startswith(model + ":")})
        for method_index, method in enumerate(("hpgi", "fixed_split")):
            denominator = sum(value for key, value in splits[method].items() if key.startswith(model + ":"))
            values = [100 * splits[method][f"{model}:{split}"] / denominator for split in choices]
            label = METHODS[method]
            color = settings["curve_styles"][label]["color"]
            axis.bar(np.arange(len(choices)) + (method_index - .5) * .36, values, width=.28,
                     color=to_rgba(color, model_settings["bar_fill_alpha"]), edgecolor=color,
                     linewidth=.5, label=label)
            points.extend(dict(method=method, model=model, metric="edge_split_share", split=split, value=value / 100)
                          for split, value in zip(choices, values))
        axis.set_xticks(np.arange(len(choices)), choices)
        axis.set_ylim(0, 105)
        axis.set_xlabel("实际切分点")
        axis.set_title(f"({chr(98 + model_index)}) {model_label}", loc="left", pad=8)
        if model_index == 0:
            axis.set_ylabel("边缘任务中占比 / %")
    figure.legend(*service_axis.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.52, .98),
                  ncol=4, frameon=False, columnspacing=1.2)
    caption = "三模型均参与混合训练；下排为各模型边缘执行任务的切分比例。"
    figure.text(.52, .055, caption, ha="center", fontsize=8.2)
    save_figure(figure, output, "F7_model_split", points, model_settings, caption)
    plt.close(figure)


def markdown_table(path, title, columns, rows, note):
    text = f"# {title}\n\n|" + "|".join(columns) + "|\n|" + "|".join(["---"] * len(columns)) + "|\n"
    text += "\n".join("|" + "|".join(str(value) for value in row) + "|" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n\n" + note + "\n", encoding="utf-8")


def base_tables(specification, central, output):
    methods = {**METHODS, **INFORMATION}
    groups = paired_groups(central, methods, [20], "load")
    summaries = [aggregate_group(groups[20, method]) for method in methods]
    for summary in summaries:
        observations = groups[20, summary["method"]]
        summary.update(energy_per_completed=summary["total_energy"] / summary["completed"],
            active_servers_mean=summary["active_server_slots"] / summary["slots"],
            switches_per_episode=summary["switches"] / summary["episodes"],
            accepted_rate=(summary["arrivals"] - summary["suppressed"] - summary["rejected"]) / summary["arrivals"],
            ready_servers_mean=sum(row["mechanism"]["ready_server_slots"] for row in observations) / summary["slots"],
            allocated_compute_per_slot=sum(row["mechanism"]["allocated_compute"] for row in observations) / summary["slots"],
            reserved_compute_per_slot=sum(row["mechanism"]["reserved_compute"] for row in observations) / summary["slots"],
            reserved_bandwidth_per_slot=sum(row["mechanism"]["reserved_bandwidth"] for row in observations) / summary["slots"])
    write_csv(output / "tables/T2_main_metrics.csv", summaries)
    rows = []
    for row in summaries:
        rows.append([methods[row["method"]], "共同保守" if row["information"] == "robust" else row["information"],
            f"{row['cost_per_slot']:.3f}" if row["information"] == "robust" else "不跨口径排名",
            f"{row['energy_per_slot']:.3f}", f"{100 * row['on_time_rate']:.2f}",
            f"{100 * row['on_time_rate_published']:.2f}", f"{row['completion_delay_p95'] * 1000:.2f}",
            f"{100 * row['timeout_rate']:.3f}", f"{100 * row['resource_failed_rate']:.3f}"])
    markdown_table(output / "tables/T2_main_metrics.md", "T2 共同独立请求测试", ["方法", "信息", "成本/槽",
        "J/槽", "全部到达按时率%", "已发布按时率%", "成功任务P95/ms", "全部到达超期率%", "资源失败率%"], rows,
        "每方法20条相同的128槽请求；单训练种子，确定性推断。P95仅对按时完成任务定义，须与失败率同时读取。信息参照不按跨口径成本排名。")
    fixed_splits = specification["methods"]["fixed_split"]["config"]["fixed_split"]
    hpgi_config = specification["methods"]["hpgi"]["config"]
    ppo = hpgi_config["ppo"]
    exploration = hpgi_config["ppo_layers"]
    parameters = [
        ("模型", "ResNet50 / ResNet152 / ViT-B32；224×224 FP32 batch1", "真实结构剖面；不等于系统硬件测量"),
        ("规模", "40终端、8服务器、2 AP", "条件仿真假设"),
        ("计算容量", "终端20；服务器4×100＋4×200 reference_work/s", "条件假设；工作单位10^9计数FLOPs"),
        ("内存", "每台1/2 GiB可用任务预算", "条件预算；请求内存来自剖面"),
        ("时序", "1秒/槽、128槽/轨迹、Q=4；Single-Timescale Q=1", "无跨槽排队"),
        ("业务", "Bernoulli到达；三模型等概率；期限U[0.8,1.0]s", "条件业务；原训练p=0.5"),
        ("负载扫描", "10 / 15 / 20 / 25 / 30请求每槽", "期望值；40终端p=.25/.375/.5/.625/.75"),
        ("匿名扫描", "k=2 / 3 / 4 / 6 / 8", "中心k4；4→2→1分箱；不足k抑制"),
        ("带宽扫描", "每AP 20 / 30 / 40 / 50 / 60 MHz", "每请求预留10MHz不变；原训练40MHz"),
        ("功耗", "端满速5W；服务器单任务满速20/40W；静态10/20W；休眠1/2W", "条件立方功率模型；非实测"),
        ("唤醒", "10/20J；一槽准备；初始全休眠", "计入能耗和服务损失"),
        ("固定切分", json.dumps(fixed_splits, ensure_ascii=False), "静态计算＋传输时延解析规则；非测试集选点"),
        ("目标与预留", "lambda_time=1；拒绝20J；匿名1ms＋控制10ms＋回程1ms", "目标权重与模型预留；实测控制时间单列T3"),
        ("原始训练", "各300轮；训练seed4200001", "8种学习方法；Greedy无需训练"),
        ("HPGI PPO", f"隐藏宽度{ppo['hidden']}；学习率{ppo['learning_rate']:g}；裁剪{ppo['clip']}；gamma={ppo['gamma']}；GAE={ppo['gae_lambda']}", "读取最终权重对应配置；基线具体配置见冻结清单"),
        ("HPGI更新", f"最多{ppo['epochs']} epoch；minibatch={ppo['minibatch']}；梯度范数阈值{ppo['max_grad_norm']}；KL阈值{ppo['target_kl']}", "有限时域价值模型、采样回报目标；达到KL阈值可提前结束更新"),
        ("HPGI探索", f"上层温度{exploration['upper']['sampling_temperature']}→{exploration['upper']['final_sampling_temperature']}；下层{exploration['lower']['sampling_temperature']}→{exploration['lower']['final_sampling_temperature']}；{exploration['lower']['temperature_anneal_cycles']}轮退火", f"上层熵系数{ppo['entropy_coefficient']}，下层{exploration['lower']['entropy_coefficient']}；信息对照匹配，结构基线探索设置不同"),
        ("训练执行", f"CUDA minibatch={ppo['minibatch']}；{hpgi_config['runtime']['collection_workers']}个CPU采样进程；torch线程{hpgi_config['runtime']['torch_threads']}", "HPGI下层预训练8192槽；每完整轮两批各1024槽；实际累计预算单列"),
        ("冻结评价", "每条件每方法20×128槽；20条独立请求；确定性决策", "不更新网络、归一化或优化器；不是逐参数重新训练"),
        ("不确定性", "5000次轨迹bootstrap，seed0，95%区间", "表示请求波动；不表示训练随机种子方差"),
    ]
    write_csv(output / "tables/T1_setup.csv", [dict(parameter=name, value=value, basis=basis) for name, value, basis in parameters])
    markdown_table(output / "tables/T1_setup.md", "T1 参数、来源与预算", ["项目", "取值", "依据/边界"], parameters,
        "来源：冻结清单、data/SCENARIO.md及数据剖面manifest。完整训练槽数按方法另见T1_training_budget.csv；不把相同iteration冒充相同梯度步数。")
    write_csv(output / "tables/T1_training_budget.csv", [dict(method=name, run_id=method["run_id"],
        total_training_slots=method["total_training_slots"], checkpoint_sha256=method["checkpoint_sha256"],
        config_sha256=method["config_sha256"], revision=method["revision"]) for name, method in specification["methods"].items()])


def timing_table(rows, output, platform):
    table = []
    for row in sorted(rows, key=lambda item: (item["method"], item["axis_value"])):
        timing = row["timing"]
        values = np.array(timing["total_control_seconds"]) * 1000
        if (len(values) != 128 or timing["warmup_slots"] != 128 or timing["torch_threads"] != 1 or
                row["runtime"]["priority_class"] != "AboveNormal" or not row["runtime"]["high_qos"]):
            raise ValueError("Uncontrolled timing measurement")
        table.append(dict(method=row["method"], expected_arrivals=row["axis_value"], measured_slots=len(values),
            policy_p50_ms=float(np.quantile(timing["policy_seconds"], .5) * 1000),
            allocation_p50_ms=float(np.quantile(timing["allocation_seconds"], .5) * 1000),
            total_p50_ms=float(np.quantile(values, .5)), total_p95_ms=float(np.quantile(values, .95)),
            total_max_ms=float(values.max()), model_reserved_ms=timing["reserved_control_seconds"] * 1000,
            exceeds_reserved_fraction=float(np.mean(values > timing["reserved_control_seconds"] * 1000))))
    write_csv(output / "tables/T3_serial_timing.csv", table)
    write_json(output / "tables/T3_platform.json", platform)
    methods = {**METHODS, **INFORMATION}
    markdown_table(output / "tables/T3_serial_timing.md", "T3 串行CPU决策计时", ["方法", "请求/槽", "P50/ms", "P95/ms", "超过10ms比例%"],
        [[methods[row["method"]], row["expected_arrivals"], f"{row['total_p50_ms']:.2f}",
          f"{row['total_p95_ms']:.2f}", f"{100 * row['exceeds_reserved_fraction']:.1f}"] for row in table],
        f"平台：{platform['processor']}，{platform['os']}，PyTorch {platform['torch']}。各方法各负载先预热128槽，再串行测128槽，CPU单torch线程、AboveNormal/HighQoS；桌面未完全隔离。包括公开观测/候选构造、策略及投影/P3；不含物理执行模拟、磁盘及模型加载。总P50是逐槽总时间的分位数，不能由分项P50相加。效能测试使用固定10ms模型控制预留，本表的实测时间未注入该效能测试。")


def exact_table(specification, output):
    from mist_sim.preflight import small_optimality_audit
    from mist_sim.training import implementation_signature
    from mist_sim.environment import Environment
    from mist_sim.profiles import Profiles
    from mist_sim.diagnostics import check_physics
    method = specification["methods"]["hpgi"]
    if implementation_signature() != source_signature(source_tree(method["revision"])):
        raise ValueError("Exact audit requires the same physical source as HPGI")
    config = copy.deepcopy(method["config"])
    config["fixed_split"] = copy.deepcopy(specification["methods"]["fixed_split"]["config"]["fixed_split"])
    report = small_optimality_audit(config, max_nodes=20000, timeout_seconds=5)
    report["fixed_split"] = config["fixed_split"]
    report["source_revision"] = method["revision"]
    if not report["complete"]:
        raise ValueError("Small reference search did not finish; no optimality claim allowed")
    table = []
    representatives = sorted(range(len(config["servers"])), key=lambda index: config["servers"][index]["capacity"])
    small = copy.deepcopy(config)
    small.update(servers=[copy.deepcopy(config["servers"][index]) for index in (representatives[0], representatives[-1])],
                 horizon=1, k=1, terminals=2, initial_active=[1, 1], learned_rates=False, capacity_mask=True, information="robust")
    for case in report["cases"]:
        for mode in ("dynamic", "fixed_split", "always_on"):
            result = case["comparisons"][mode]
            scenario = copy.deepcopy(small)
            scenario["method"] = "fixed_split" if mode == "fixed_split" else "hpgi"
            batch = [dict(id=str(index), model=case["model"], **{"class": "standard"}, weight=1.,
                size=min(Profiles(scenario).sizes(case["model"])), position=case["position"],
                deadline=config["deadline_range"][0], fading=config["fading_range"][0], tie=float(index))
                for index in range(case["requests"])]
            environment = Environment(scenario, [batch])
            environment.begin_slot(result["active"])
            record = environment.execute(result["actions"])
            check_physics(scenario, [record])
            if not np.isclose(record["cost"], result["cost"], rtol=1e-12) or record["completed"] != result["completed"]:
                raise ValueError("Exact reference action replay differs")
            result["replayed_energy"] = record["energy"]
            table.append(dict(model=case["model"], position=case["position"], requests=case["requests"],
                constraint=mode, optimal_cost=result["cost"], completed=result["completed"],
                active=json.dumps(result["active"]), splits=json.dumps(result["splits"]),
                total_energy=sum(record["energy"].values()), energy_components=json.dumps(record["energy"]),
                resources_verified=True, resource_usage=json.dumps(record["usage"]),
                excess_cost=result["cost"] - case["comparisons"]["dynamic"]["cost"]))
    write_json(output / "tables/T4_exact_audit.json", report)
    write_csv(output / "tables/T4_exact_cases.csv", table)
    markdown_table(output / "tables/T4_exact_reference.md", "T4 两服务器小例精确参照", ["模型", "位置/m", "请求数", "动态最优成本", "固定切分受限最优", "全开受限最优"],
        [[MODEL_NAMES[case["model"]], case["position"], case["requests"],
          *[f"{case['comparisons'][mode]['cost']:.6f}" for mode in ("dynamic", "fixed_split", "always_on")]] for case in report["cases"]],
        "12个预先规定小例全部搜索完整；2台代表服务器、k1、初始已就绪、单槽1/2请求。此表隔离动作约束的潜在优化空间，不是已训练HPGI结果，不是八服务器全局最优差距。固定点复用当前3/2/4；所有合法固定点的结果及零差距反例保留在JSON。")
    print(json.dumps({"exact_complete": report["complete"], "strict_improvement_cases": report["strict_improvement_cases"]}))


def provenance(spec_path, output, sources):
    specification = read_json(spec_path)
    for path, expected_hash in specification["figure_provenance"].items():
        if digest(path) != expected_hash:
            raise ValueError("Confirmed training-figure provenance changed")
        sources[path] = expected_hash
    for directory, basename in (("learning-baselines", "reward_per_slot_zh"),
                                 ("information-baselines", "service_zh")):
        for extension in ("png", "pdf"):
            path = output.parent / directory / "figures/convergence" / f"{basename}.{extension}"
            sources[str(path)] = digest(path)
    assets = {str(path.relative_to(output)): digest(path) for path in output.rglob("*")
              if path.is_file() and path.name != "provenance.json"}
    write_json(output / "provenance.json", {"spec": str(spec_path), "spec_sha256": digest(spec_path),
        "script_sha256": digest(__file__), "sources": sources, "outputs": assets, "training_updates": 0,
        "base_spec_sha256": specification["base_spec_sha256"],
        "scope": "Frozen-policy transfer; one trained seed; request-trajectory uncertainty only"})


def gallery(output):
    assets = [
        ("F1 六方法学习曲线", output.parent / "learning-baselines/figures/convergence/reward_per_slot_zh.png",
         "单训练种子，300轮，10轮后向均值；展示学习改善，不代替独立测试。"),
        ("F2 信息机制训练对照", output.parent / "information-baselines/figures/convergence/service_zh.png",
         "共同物理服务指标；Full-Information为理想信息参照。"),
        ("F3 五点请求负载", output / "figures/F3_load/F3_load_zh.png",
         "冻结既有策略，测试期望10至30请求/槽；两种服务分母与能耗同时读取。"),
        ("F4 五点匿名等级", output / "figures/F4_anonymity/F4_anonymity_zh.png",
         "k=2/3/4/6/8；展示真实服务、超期和共同网关信息损失。"),
        ("F5 五点AP带宽容量", output / "figures/F5_bandwidth/F5_bandwidth_zh.png",
         "每AP20至60MHz；每请求10MHz不变，考察共享容量变化。"),
        ("F6 能耗与启停机制", output / "figures/F6_energy_mechanism/F6_energy_mechanism_zh.png",
         "复用共同中负载测试，拆分完整能耗及活跃服务器、启停次数。"),
        ("F7 分模型服务与切分", output / "figures/F7_model_split/F7_model_split_zh.png",
         "三模型已参与混合训练；用于解释收益来源，不是未见模型泛化。"),
    ]
    sections = []
    for title, path, caption in assets:
        if not path.exists() or not path.with_suffix(".pdf").exists():
            raise ValueError(f"Missing final figure: {path}")
        relative = Path(os.path.relpath(path, output)).as_posix()
        image_url = relative + "?v=" + digest(path)[:12]
        sections.append(f'<section><h2>{escape(title)}</h2><p>{escape(caption)}</p>'
            f'<a href="{escape(image_url)}"><img loading="lazy" src="{escape(image_url)}" alt="{escape(title)}"></a>'
            f'<p><a href="{escape(relative)}">PNG原图</a> · <a href="{escape(str(Path(relative).with_suffix(".pdf")).replace(chr(92), "/"))}">矢量PDF</a></p></section>')
    for name in ("T1_setup", "T2_main_metrics", "T3_serial_timing", "T4_exact_reference"):
        path = output / f"tables/{name}.md"
        lines = path.read_text(encoding="utf-8").splitlines()
        table_lines = [line for line in lines if line.startswith("|") and not line.startswith("|---")]
        rendered_rows = []
        for index, line in enumerate(table_lines):
            tag = "th" if index == 0 else "td"
            rendered_rows.append("<tr>" + "".join(f"<{tag}>{escape(cell)}</{tag}>" for cell in line.strip("|").split("|")) + "</tr>")
        note = " ".join(line for line in lines if line and not line.startswith(("|", "#")))
        sections.append(f'<section><h2>{escape(lines[0].lstrip("# "))}</h2><div class="table"><table>' +
                        "".join(rendered_rows) + f'</table></div><p>{escape(note)}</p></section>')
    html = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MIST论文实验图表</title><style>body{font-family:Georgia,'Microsoft YaHei',serif;background:#f3f4f6;color:#20242a;margin:0 auto;max-width:1120px;padding:30px}h1{font-size:28px}h2{font-size:22px}p{line-height:1.7;color:#4c5561}section{background:white;padding:25px;margin:24px 0;border:1px solid #dce0e6;border-radius:8px}img{width:100%;height:auto}a{color:#a42e2e}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #dce0e6;padding:10px;text-align:left}th{background:#f4f5f7}</style>
<h1>MIST论文实验图表：7图＋4表</h1><p>原始学习结果复用；新增结果为冻结策略测试。每点20条请求轨迹，误差棒反映请求波动。图片可点开查看原尺寸；各图提供PNG及矢量PDF。写作应区分单训练实例的条件仿真证据、机制小例和真实部署能力。</p>"""
    (output / "index.html").write_text(html + "".join(sections) + "</html>", encoding="utf-8")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("base", "exact", "exact-worker", "all"))
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    spec_path, output = args.spec.resolve(), args.output.resolve()
    specification = read_json(spec_path)
    base_path = Path(specification["base_spec"])
    if digest(base_path) != specification["base_spec_sha256"]:
        raise ValueError("Original specification changed")
    if args.command == "exact":
        files = source_tree(specification["methods"]["hpgi"]["revision"])
        with tempfile.TemporaryDirectory(prefix="mist-paper-exact-") as temporary:
            extract_source(files, temporary)
            subprocess.run([sys.executable, "-m", "scripts.plotting.plot_paper_results", "exact-worker",
                "--spec", str(spec_path), "--output", str(output)], cwd=ROOT,
                env=dict(os.environ, PYTHONPATH=str(Path(temporary) / "src")), check=True)
        return
    if args.command == "exact-worker":
        exact_table(specification, output)
        return
    central, sources = load_batch(base_path, "A")
    plt, settings = setup_plotting()
    mechanism_figures(plt, settings, central, output)
    base_tables(specification, central, output)
    if args.command == "all":
        rows = central.copy()
        for path, batch in ((base_path, "B"), (spec_path, "E1"), (spec_path, "E2")):
            additional, source = load_batch(path, batch)
            rows.extend(additional)
            sources.update(source)
        for axis, value in (("anonymity", 4), ("bandwidth", 40)):
            rows.extend({**row, "axis": axis, "axis_value": value} for row in central if row["decode"] == "deterministic")
        from mist_sim.anonymity import publish
        import mist_sim.anonymity
        gateway_source = Path(mist_sim.anonymity.__file__)
        if hashlib.sha256(gateway_source.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != specification["shared_physics_sha256"]["anonymity.py"]:
            raise ValueError("Central anonymity reconstruction uses different physics")
        base = read_json(base_path)
        inputs = {task["input"]["seed"]: task["input"] for task in base["tasks"]
                  if task["batch"] == "A" and task["method"] == "hpgi" and task["deterministic"]}
        gateway_cache = {}
        config = specification["methods"]["hpgi"]["config"]
        for row in rows:
            if row["axis"] == "anonymity" and row["axis_value"] == 4:
                if row["seed"] not in gateway_cache:
                    entry = inputs[row["seed"]]
                    if digest(entry["path"]) != entry["sha256"]:
                        raise ValueError("Central gateway input changed")
                    records = [{"raw": batch, "public": publish(batch, config, "robust")}
                               for batch in read_json(entry["path"])["trace"]]
                    gateway_cache[row["seed"]] = anonymity_metrics(records, config)
                row["anonymity"] = gateway_cache[row["seed"]]
        summaries, pairs = [], []
        definitions = [
            ("load", [10, 15, 20, 25, 30], METHODS, "期望到达数 / (请求/时隙)", "F3_load",
             [(key, labels[0], labels[2], labels[4]) for key, labels in PANELS.items()], "匿名k=4；每AP容量40MHz；仅改变请求到达概率。"),
            ("anonymity", [2, 3, 4, 6, 8], INFORMATION, "匿名等级 k", "F4_anonymity",
             [("on_time_rate", "全部到达请求的服务", "按时完成率 / %", 100),
              ("on_time_rate_published", "已发布请求的服务", "按时完成率 / %", 100),
              ("timeout_accepted", "接纳后的实际超期", "接纳请求中超期率 / %", 100),
              ("gateway", "共同匿名网关", "比例 / %", 100)], "区间宽度按位置与期限各自全范围归一化后平均；固定尺寸不计。"),
            ("bandwidth", [20, 30, 40, 50, 60], METHODS, "每AP总带宽 / MHz", "F5_bandwidth",
             [("cost_per_slot", "综合成本", "成本 / 时隙", 1),
              ("on_time_rate", "全部到达请求的服务", "按时完成率 / %", 100),
              ("edge_per_slot", "边缘执行任务", "任务数 / 时隙", 1),
              ("bandwidth_utilization", "预留带宽利用率", "平均利用率 / %", 100)], "每请求预留10MHz不变；每AP可并发2/3/4/5/6项边缘任务。"),
        ]
        for axis, values, methods, xlabel, name, panels, context in definitions:
            groups = paired_groups(rows, methods, values, axis)
            metric_figure(plt, settings, groups, methods, values, xlabel, panels, output, name, context,
                          y_limits={"cost_per_slot": (125, 220), "on_time_rate": (50, 100)}
                          if axis == "bandwidth" else None)
            summaries.extend({"axis": axis, "value": value, **aggregate_group(group)}
                             for (value, method), group in groups.items())
            for value in values:
                reference = groups[value, "hpgi"]
                for method in methods:
                    if method == "hpgi":
                        continue
                    comparison = groups[value, method]
                    estimates = paired_intervals(reference, comparison)
                    if axis != "anonymity":
                        estimate, lower, upper = paired_cost_interval(reference, comparison)
                        estimates.append({"load": reference[0]["load"], "decode": "deterministic",
                            "comparison": method, "metric": "cost_per_slot", "hpgi_minus_comparison": estimate,
                            "ci95_low": lower, "ci95_high": upper, "paired_trajectories": 20,
                            "bootstrap_replicates": 5000, "bootstrap_seed": 0})
                    pairs.extend({"axis": axis, "value": value, **estimate} for estimate in estimates)
        write_csv(output / "summary.csv", summaries)
        write_csv(output / "paired_differences.csv", pairs)
        gateway_groups = paired_groups(rows, INFORMATION, [2, 3, 4, 6, 8], "anonymity")
        gateway_table = []
        for value in (2, 3, 4, 6, 8):
            group = gateway_groups[value, "hpgi"]
            count = Counter()
            for row in group:
                count.update(row["anonymity"]["group_sizes_request_weighted"])
            gateway_table.append(dict(k=value, minimum_observed_group=min(map(int, count)),
                covered=sum(row["anonymity"]["covered"] for row in group),
                published=sum(row["anonymity"]["published"] for row in group),
                suppression_rate=interval(group, "suppression_rate")[0],
                normalized_information_width=interval(group, "information_width")[0]))
        write_csv(output / "anonymity_checks.csv", gateway_table)
        timing, source = load_batch(spec_path, "T")
        sources.update(source)
        platform_path = spec_path.parent / "timing_platform.json"
        timing_table(timing, output, read_json(platform_path))
        sources[str(platform_path)] = digest(platform_path)
        gallery(output)
    provenance(spec_path, output, sources)
    print(str(output), flush=True)


if __name__ == "__main__":
    main()
