# MIST 仿真实验工程

`mist-sim/`是独立仿真工程与Git仓库根，`src/mist_sim/`是Python导入包，HPGI是提出的方法。仓库实现环境、HPGI、全部基线及训练/评价流程。

## 仓库与隐私

远端为私有仓库`wanghong5233/mist-sim`。只提交源码、配置、测试及必要说明；results、runs、data中的实测资产、权重、原始图像、实验协议和本机执行记录均忽略，论文不在此仓库内。推送前执行`python scripts/maintenance/check_publish.py`，检查当前暂存内容与历史；首次建库不导入科研根仓库历史。Git提交使用GitHub noreply邮箱。

本机布局为`02_实验/mist-sim/`。科研父仓库以子模块记录这个独立仓库的提交，不重复管理源码；提交/推送仿真时进入mist-sim。克隆本仓库无需导师论文或MERA检出即可运行合成输入检查；数据恢复见[data说明](data/README.md)。父仓库只在本地更新引用，不随仿真push上传。

## 阅读入口

|文件|职责|
|---|---|
|[IMPLEMENTATION.md](IMPLEMENTATION.md)|唯一当前实现规格：模型、state/action张量、奖励、PPO、轨迹长度与更新节奏|
|[data/README.md](data/README.md)|本地剖面、原始输入和恢复边界|
|[scripts/README.md](scripts/README.md)|可执行脚本索引|
|[AGENTS.md](AGENTS.md)|执行与文档维护约束|

README管导航和命令；IMPLEMENTATION管当前语义；配置JSON管实际值。实验协议、SETUP、本地结果核验说明及父项目PLAN保留在本机，不上传。修改替换旧描述、链接复用，Git负责源码历史。

## 目录

```text
mist-sim/
├── configs/       环境/算法JSON及组合入口
├── src/mist_sim/  唯一可安装源码；环境、PPO、基线、数据与指标
├── tests/         手算、反例、训练及恢复回归测试
├── scripts/       data/数据处理，validation/核验，maintenance/维护
├── data/          raw/共享原始输入，processed/可复用剖面
├── runs/          正式执行的原始轨迹、配置、日志和检查点
├── results/       可追溯的轻量摘要、正式图表及汇总
└── references/    只读上游参考，不是运行依赖
```

采用[Python标准src布局](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)，不再外套mist_sim目录、不按版本复制源码。测试原始文件和权重使用系统临时目录，默认退出清理；不把工程短跑混入正式runs。真实训练成功/失败均保留，不按性能好坏删除。

## 环境与输入

本机唯一运行解释器：`D:/Anaconda/envs/pytorch/python.exe`（Conda pytorch）。不激活旧.venv、不重新安装CUDA。Python3.10中文editable路径按cp936读取，不设置`PYTHONUTF8`或`-X utf8`。首次执行：

```powershell
& D:/Anaconda/envs/pytorch/python.exe scripts/maintenance/check_environment.py
```

确认executable及package_path。后续命令同样使用该绝对解释器；其他机器先按pyproject安装依赖及`pip install --no-deps -e .`。PPO默认CPU；离线测量可用GPU。仿真只查数值剖面，不加载三个视觉模型，不需训练视觉网络。

普通安装同样支持：从克隆的mist-sim根目录运行，CLI优先使用当前工程配置；从其他目录运行时显式传`--root 工程路径`。验证套件从工程根运行，不假定机器上存在论文目录。

`configs/smoke.json`是随仓库提供的手算合成测试；`configs/conditional.json`引用本机ResNet50/ResNet152/ViT-B32结构剖面，8服务器、40终端等资源/业务/功率为条件假设。本地来源见data/SCENARIO.md和data/LOCAL_SOURCES.md；这些私有资产不随clone分发。

## 方法与模块

|方法键|实际差异|
|---|---|
|hpgi|双层PPO、容量掩码、P3|
|always_on|全开、独立训练快层|
|fixed_split|逐模型固定合法切分，保留双层训练|
|single_timescale|Q=1，PPO仍交替冻结|
|fixed_window|无联合容量掩码、学习速率、执行前投影|
|upper_ppo_greedy|慢层PPO、容量感知增量成本贪心快层|
|greedy_consolidation|上一完整区间驱动启停、当前公开候选贪心|
|nominal|区间中心构造候选，独立训练并报告真实违约|
|full_information|同发布集合的理想真实信息参照，独立训练|

`environment/anonymity/profiles/allocation`负责物理和信息边界；`policies、buffers、ppo、training`负责策略及学习；`evaluation/storage/diagnostics`负责核算、证据及复算。`reference_solvers`为有界离线/单隙枚举，完整搜索才称最优。MERA只作参考，详见[来源说明](references/README.md)。

## 执行命令

在本目录执行。纯代码克隆可直接跑pytest和smoke检查；缺少私有剖面时一项数据集成测试明确skip，其余不跳过。conditional相关命令需要先恢复本机输入。

```powershell
& D:/Anaconda/envs/pytorch/python.exe -m pytest tests -q
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim check --config configs/smoke.json
& D:/Anaconda/envs/pytorch/python.exe scripts/maintenance/check_publish.py
& D:/Anaconda/envs/pytorch/python.exe scripts/validation/check_rollouts.py
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim.verification --summary results/rl_chain_check.json
& D:/Anaconda/envs/pytorch/python.exe scripts/validation/check_profile_chain.py
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim train --config configs/conditional.json --method hpgi --seed 4200001
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim evaluate --checkpoint runs/<训练ID>/checkpoints/final.pt --seed 4400019
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim replay --checkpoint runs/<训练ID>/checkpoints/final.pt --records runs/<评价ID>/traces/evaluation.json
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim summarize --records runs/<评价ID>/traces/evaluation.json
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim train --checkpoint runs/<训练ID>/checkpoints/last.pt --extra-cycles 1
```

占位ID替换为实际运行路径；学习方法必须加载自己的权重。`--overlay 文件.json`和`--set ppo.learning_rate=0.0001`严格覆盖配置，未知键报错；上下层可通过ppo_layers独立配置。变更源码/配置/剖面后拒绝无说明续训。

## 日志与资产

每个正式run用用途-方法-seed命名，加后缀防覆盖；resolved_config及manifest记录实际配置、源码/输入哈希、种子、设备、耗时、退出状态。progress记录阶段与rollout摘要；metrics记录物理指标/PPO诊断；validation_history和selection记录验证及选模。`*_behavior/rollout/inputs/update.json`含可复算的关键链路数据，具体语义见IMPLEMENTATION。

检查点保存网络、优化器、统计、随机状态及完成阶段。默认initial/last/final/best/selected，阶段快照关闭；显式开启须设上限。best为开发选择，final为最终状态；没有服务阈值时不能称best满足论文业务要求。failed_last只诊断，正常恢复用last/initial。

完整验证套件默认临时空间，退出删除原始轨迹与权重；仅显式`--keep-artifacts tmp/debug-example`保留排障资料。汇总只保留必要证据，不复制全部原始日志到文档。未运行收敛实验，任何测试数值都不得进入论文性能表。
