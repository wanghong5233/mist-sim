"""Verify completed evidence, run the declared cross-controller diagnosis, report.

No candidate is selected using test inputs. Diagnostic interventions are frozen
and are not new trained policies. Existing figures are never regenerated here.
"""
import argparse
from collections import defaultdict
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from scripts.evaluation import strengthen_evidence as s

f = s.f
OUT = f.ROOT / "results/comparisons/paper-completion-v1/strengthening"


def read_rows(path, count=None):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if count is not None:
        assert len(rows) == count, (path, len(rows), count)
    assert len({r["id"] for r in rows}) == len(rows), path
    assert all(r["slots"] == 128 and r["physics"]["constraints_and_kkt"]
               and r["frozen_state_verified"] for r in rows), path
    return rows


def aggregate(rows):
    slots = sum(r["slots"] for r in rows)
    total = lambda k: sum(r["metrics"][k] for r in rows)
    return dict(episodes=len(rows), slots=slots,
        cost=total("conservative_cost") / slots, energy=total("total_energy") / slots,
        on_time=100 * total("completed") / total("arrivals"),
        published_on_time=100 * total("completed") / (total("arrivals") - total("suppressed")),
        rejected_per_slot=total("rejected") / slots,
        active=total("active_server_slots") / slots,
        switches=total("switches") / len(rows), timeout=total("timeout"),
        resource_failed=total("resource_failed"), injected_seconds=total("controller_seconds"))


def grouped(rows):
    result = defaultdict(list)
    for r in rows:
        result[(r["method"], r["load"], r.get("config_overrides", {}).get("control_seconds", .01))].append(r)
    return result


def paired(a, b):
    import numpy as np
    av = {r["input_sha256"]: r for r in a}
    bv = {r["input_sha256"]: r for r in b}
    assert av.keys() == bv.keys() and len(av) == 20
    differences = []
    for key in sorted(av):
        x, y = av[key], bv[key]
        assert x["slots"] == y["slots"]
        for k in ("arrivals", "suppressed"):
            assert x["metrics"][k] == y["metrics"][k]
        differences.append((x["metrics"]["conservative_cost"] - y["metrics"]["conservative_cost"]) / x["slots"])
    values = np.asarray(differences)
    rng = np.random.default_rng(5610001)
    boot = values[rng.integers(0, len(values), (5000, len(values)))].mean(axis=1)
    return dict(mean_difference=float(values.mean()),
                ci95=list(map(float, np.quantile(boot, [.025, .975]))),
                definition="first minus second, J/slot; paired trajectory bootstrap, not training seed uncertainty")


def verify_completed():
    spec = f.read_json(s.SPEC)
    assert f.digest(s.__file__) == spec["source_script_sha256"]
    assert f.digest(f.__file__) == spec["frozen_evaluator_sha256"]
    for mode in ("static", "latency"):
        status = f.read_json(s.LOG / (mode + "_status.json"))
        assert status["status"] == "completed" and status["spec_sha256"] == f.digest(s.SPEC)
    selection = f.read_json(s.LOG / "selection.json")
    for stage, n, score_key in (("dev_servers", 324, "server_scores"),
                              ("dev_cuts", 256, "cut_scores"),
                              ("dev_fixed_servers", 324, "fixed_server_scores")):
        rows = read_rows(s.RESULT / stage / "observations.jsonl", n)
        assert {r["seed"] for r in rows} == {x["seed"] for x in spec["inputs"]["development"]}
        assert s.select(rows)[1] == selection[score_key]
    assert selection["adaptive_active"] == spec["server_vectors"][int(min(selection["server_scores"], key=selection["server_scores"].get)[6:])]
    assert selection["fixed_cuts"] == spec["split_combinations"][int(min(selection["cut_scores"], key=selection["cut_scores"].get)[3:])]
    assert selection["fixed_active"] == spec["server_vectors"][int(min(selection["fixed_server_scores"], key=selection["fixed_server_scores"].get)[11:])]
    all_inputs = [e for entries in spec["inputs"].values() for e in entries]
    assert len({e["seed"] for e in all_inputs}) == len(all_inputs)
    for e in all_inputs + sum(spec["timing_inputs"].values(), []):
        assert f.digest(e["path"]) == e["sha256"]
    for method in spec["methods"].values():
        assert f.digest(method["checkpoint"]) == method["checkpoint_sha256"]
        assert f.source_signature(f.source_tree(method["revision"])) == method["implementation_signature"]
    tests = read_rows(s.RESULT / "test/observations.jsonl", 180)
    groups = grouped(tests)
    for load in ("low", "middle", "high"):
        expected = {e["sha256"] for e in spec["inputs"][load]}
        for method in ("hpgi", "static_adaptive", "static_fixed"):
            rows = groups[(method, load, .01)]
            assert len(rows) == 20 and {r["input_sha256"] for r in rows} == expected
            paired(rows, groups[("hpgi", load, .01)])
    return spec, selection, tests


def diagnose(worker=False):
    spec, selection, tests = verify_completed()
    groups = grouped(tests)
    needed = any(aggregate(groups[(m, load, .01)])["cost"] < aggregate(groups[("hpgi", load, .01)])["cost"]
                 for m in ("static_adaptive", "static_fixed") for load in ("low", "middle", "high"))
    if not needed:
        print("Cross-controller diagnosis not triggered")
        return
    path = s.LOG / "diagnostic_spec.json"
    base = spec["methods"]["hpgi"]
    low = [int(server["capacity"] == min(x["capacity"] for x in base["config"]["servers"]))
           for server in base["config"]["servers"]]
    design = dict(parent_spec_sha256=f.digest(s.SPEC), selection_sha256=f.digest(s.LOG / "selection.json"),
                  script_sha256=f.digest(__file__), active=selection["adaptive_active"],
                  four_low=low, inputs=spec["inputs"]["middle"], budget_seconds=1800,
                  purpose="Declared frozen cross-controller mechanism check; no selection or training")
    if path.exists():
        assert f.read_json(path) == design
    else:
        f.write_json(path, design)
    if worker:
        methods = {"selected_servers_hpgi_lower": s.static_method(base, design["active"], learned_lower=True),
                   "four_low_greedy": s.static_method(base, low)}
        jobs = [(m, [s.task(name, e) for e in design["inputs"]]) for name, m in methods.items()]
        s.stage("diagnostic", jobs, time.time() + design["budget_seconds"])
        return
    started = time.time()
    status_path = s.LOG / "diagnostic_status.json"
    status = dict(status="running", pid=os.getpid(), spec_sha256=f.digest(path))
    f.write_json(status_path, status)
    try:
        with tempfile.TemporaryDirectory(prefix="mist-evidence-diagnostic-") as directory:
            f.extract_source(f.source_tree(base["revision"]), directory)
            env = dict(os.environ, PYTHONPATH=str(Path(directory) / "src"))
            command = [sys.executable, "-u", "-m", "scripts.evaluation.report_strengthening", "diagnostic-worker"]
            with (s.LOG / "diagnostic.log").open("a", encoding="utf-8") as log:
                subprocess.run(command, cwd=f.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        read_rows(s.RESULT / "diagnostic/observations.jsonl", 40)
        status["status"] = "completed"
    except BaseException:
        status["status"] = "failed"
        raise
    finally:
        status["wall_seconds"] = time.time() - started
        f.write_json(status_path, status)
    print(json.dumps(status))


def report():
    spec, selection, tests = verify_completed()
    latency = []
    paths = list(s.RESULT.glob("*/observations.jsonl"))
    for method in spec["methods"]:
        latency.extend(read_rows(s.RESULT / ("latency_" + method) / "observations.jsonl", 80))
    for batch in ("A", "B"):
        path = f.ROOT / ("results/paper-evaluation-v1-" + batch) / "observations.jsonl"
        paths.append(path)
        latency.extend(r for r in read_rows(path) if r["method"] in spec["methods"]
                       and r["load"] in ("middle", "high") and r["decode"] == "deterministic")
    lg = grouped(latency)
    assert len(lg) == 18
    comparisons = {}
    tg = grouped(tests)
    for load in ("low", "middle", "high"):
        for m in ("static_adaptive", "static_fixed"):
            comparisons[f"hpgi_minus_{m}_{load}"] = paired(tg[("hpgi", load, .01)], tg[(m, load, .01)])
    for (m, load, reserve), rows in lg.items():
        assert len(rows) == 20
        assert {r["input_sha256"] for r in rows} == {e["sha256"] for e in spec["timing_inputs"][load]}
        assert all(r["checkpoint_sha256"] == spec["methods"][m]["checkpoint_sha256"] for r in rows)
        comparisons[f"{m}_{load}_{reserve}_minus_0.01"] = paired(rows, lg[(m, load, .01)])
    diagnostic = read_rows(s.RESULT / "diagnostic/observations.jsonl", 40)
    assert f.read_json(s.LOG / "diagnostic_status.json")["status"] == "completed"
    dg = grouped(diagnostic)
    for key, rows in dg.items():
        comparisons[f"{key[0]}_minus_hpgi_middle"] = paired(rows, tg[("hpgi", "middle", .01)])
    summary_rows = [dict(suite=suite, method=k[0], load=k[1], reserve_ms=1000*k[2], **aggregate(v))
                    for suite, groups in (("static", tg), ("reserve", lg), ("diagnostic", dg))
                    for k, v in sorted(groups.items())]
    config_keys = ("window", "learning_rate", "ppo_epochs", "minibatch_size", "entropy_coef",
                   "upper_entropy_coef", "lower_entropy_coef", "upper_temperature", "lower_temperature",
                   "temperature_anneal_cycles", "gamma", "gae_lambda")
    summary = dict(spec_sha256=f.digest(s.SPEC), selection_sha256=f.digest(s.LOG / "selection.json"),
        counts=dict(development=904, independent_test=180, reserve_new=240, diagnostic=40, preflight=6),
        selection={k: selection[k] for k in ("adaptive_active", "fixed_active", "fixed_cuts")},
        rows=summary_rows, paired_comparisons=comparisons,
        provenance={str(p.relative_to(f.ROOT)): f.digest(p) for p in paths},
        comparison_configs={m:{k:v for k,v in method["config"].items() if k in config_keys or "temperature" in k}
                            for m,method in spec["methods"].items()})
    OUT.mkdir(parents=True, exist_ok=True)
    f.write_json(OUT / "summary.json", summary)
    import csv
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader(); writer.writerows(summary_rows)
    labels = {"hpgi":"HPGI", "static_adaptive":"开发选定服务器＋逐请求贪心切分",
              "static_fixed":"开发选定服务器＋固定切分", "single_timescale":"Single-Timescale",
              "fixed_split":"Fixed-Split", "selected_servers_hpgi_lower":"选定服务器＋冻结 HPGI 下层",
              "four_low_greedy":"四台低功率服务器＋贪心下层"}
    loads = {"low":"10", "middle":"20", "high":"30"}
    parts = ['<!-- strengthening:start --><section id="strengthening"><h2>补充证据：静态参照与控制预留</h2>',
             '<p>静态方案仅用 4 条开发轨迹选定；独立测试每档 20 条新轨迹。保留原七图。以下新增表格来自冻结权重补评，不新增训练种子。</p>']
    for suite, title in (("static","T5 有界静态参照"), ("reserve","T6 控制预留敏感性"), ("diagnostic","T7 固定服务器交叉诊断")):
        parts.append(f'<h3>{title}</h3><table><thead><tr><th>方法</th><th>请求/槽</th><th>预留/ms</th><th>成本/(J/槽)</th><th>实际能耗/(J/槽)</th><th>全到达按时率/%</th></tr></thead><tbody>')
        for r in summary_rows:
            if r["suite"] != suite: continue
            parts.append(f'<tr><td>{html.escape(labels[r["method"]])}</td><td>{loads[r["load"]]}</td><td>{r["reserve_ms"]:.0f}</td><td>{r["cost"]:.2f}</td><td>{r["energy"]:.2f}</td><td>{r["on_time"]:.2f}</td></tr>')
        parts.append('</tbody></table>')
    parts.append('<p>10/60/80 ms 为候选及执行模型中的固定预留，没有注入机器实测耗时；P95 不是硬上界。交叉诊断仅替换固定服务器或下层，不是重训结果。静态搜索非联合全枚举最优。</p><p><a href="strengthening/summary.csv">逐项数值</a> · <a href="strengthening/summary.json">配对区间、选择结果及来源校验</a></p></section><!-- strengthening:end -->')
    gallery = OUT.parent / "index.html"
    page = gallery.read_text(encoding="utf-8")
    import re
    block = '\n'.join(parts)
    if '<!-- strengthening:start -->' in page:
        page = re.sub(r'<!-- strengthening:start -->.*?<!-- strengthening:end -->', lambda _: block, page, flags=re.S)
    else:
        assert '</body>' in page
        page = page.replace('</body>', block + '\n</body>')
    gallery.write_text(page, encoding="utf-8")
    print(json.dumps({"rows": summary_rows, "paired": comparisons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("diagnose", "diagnostic-worker", "report"))
    args = parser.parse_args()
    if args.command == "report": report()
    else: diagnose(args.command == "diagnostic-worker")
