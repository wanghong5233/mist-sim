# MIST 仿真实验工程

`mist-sim/`是独立仿真工程与Git仓库根，`src/mist_sim/`是Python导入包，HPGI是提出的方法。仓库实现环境、HPGI、全部基线及训练/评价流程。

## 仓库与隐私

远端为私有仓库`wanghong5233/mist-sim`。只提交源码、配置、测试及必要说明；results、logs、data中的实测资产、权重、原始图像、实验协议和本机执行记录均忽略，论文不在此仓库内。推送前执行`python scripts/maintenance/check_publish.py`，检查当前暂存内容与历史；首次建库不导入科研根仓库历史。Git提交使用GitHub noreply邮箱。

本机布局为`02_实验/mist-sim/`。科研父仓库以子模块记录这个独立仓库的提交，不重复管理源码；提交/推送仿真时进入mist-sim。克隆本仓库无需导师论文或MERA检出即可运行合成输入检查；数据恢复见[data说明](data/README.md)。父仓库只在本地更新引用，不随仿真push上传。

## 阅读入口

|文件|职责|
|---|---|
|[IMPLEMENTATION.md](IMPLEMENTATION.md)|唯一当前实现规格：模型、state/action张量、奖励、PPO、轨迹长度与更新节奏|
|[data/README.md](data/README.md)|本地剖面、原始输入和恢复边界|
|[scripts/README.md](scripts/README.md)|可执行脚本索引|
|[AGENTS.md](AGENTS.md)|执行与文档维护约束|

README管导航和命令；IMPLEMENTATION管当前语义；配置JSON管实际值。实验协议和SETUP保留在本机，不上传。修改替换旧描述、链接复用，Git负责源码历史。

## 目录

```text
mist-sim/
├── configs/       环境/算法JSON及组合入口
├── src/mist_sim/  唯一可安装源码；环境、PPO、基线、数据与指标
├── tests/         手算、反例、训练及恢复回归测试
├── scripts/       data/数据处理，validation/核验，maintenance/维护
├── data/          raw/原始输入，profiling/剖析及来源，processed/仿真数值输入
├── logs/runs/     正式训练/评价的运行记录，执行时创建
├── results/       每次运行的原始结果CSV，以及据此生成的统计表和论文图
└── references/    只读上游参考，不是运行依赖
```

采用[Python标准src布局](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)。日志采用“一个运行一个目录”：PyTorch的[SummaryWriter](https://docs.pytorch.org/docs/stable/tensorboard.html)支持指定log_dir，[CleanRL PPO](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo.py)按run_name隔离事件。本项目统一指定logs/runs，不依赖工具的默认顶层runs目录；该路径是项目约定，并非唯一行业标准。

测试文件和权重在系统临时目录，退出清理；检查/维护脚本默认只打印，确需排障才显式指定输出到tmp。数据剖析归data，不算强化学习实验结果。尚未正式训练时logs/runs与results不放占位文件，也不提前生成空分类。正式训练成功/失败均保留，不按性能好坏删除。

## 环境与输入

本机唯一运行解释器：`D:/Anaconda/envs/pytorch/python.exe`（Conda pytorch）。不激活旧.venv、不重新安装CUDA。Python3.10中文editable路径按cp936读取，不设置`PYTHONUTF8`或`-X utf8`。首次执行：

```powershell
& D:/Anaconda/envs/pytorch/python.exe scripts/maintenance/check_environment.py
```

确认executable及package_path。后续命令同样使用该绝对解释器；其他机器安装`pip install -e ".[monitoring]"`，开发测试用`".[test]"`。本机TensorBoard已安装。PPO默认CPU；离线测量可用GPU。仿真只查数值剖面，不加载三个视觉模型，不需训练视觉网络。

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
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim.verification
& D:/Anaconda/envs/pytorch/python.exe scripts/validation/check_profile_chain.py
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim train --config configs/conditional.json --method hpgi --dry-run
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim train --config configs/conditional.json --method hpgi --seed 4200001
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim evaluate --checkpoint logs/runs/<训练ID>/checkpoints/final.pt --seed 4400019
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim replay --checkpoint logs/runs/<训练ID>/checkpoints/final.pt --records logs/runs/<评价ID>/traces/evaluation.json
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim summarize --records logs/runs/<评价ID>/traces/evaluation.json
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim train --checkpoint logs/runs/<训练ID>/checkpoints/last.pt --extra-cycles 1
```

占位ID替换为实际运行路径；学习方法必须加载自己的权重。`--overlay 文件.json`和`--set ppo.learning_rate=0.0001`严格覆盖配置，未知键报错；上下层可通过ppo_layers独立配置。变更源码/配置/剖面后拒绝无说明续训。

## 日志与资产

每个正式运行位于logs/runs/<用途-方法-种子>/，同一次运行的材料放在一起，防止日志和权重错配；自动后缀仅防覆盖。目录固定为：

```text
logs/runs/<运行名>/
├── resolved_config.json、manifest.json   配置、输入/源码哈希、运行状态
├── stdout.log、stderr.log               可读进度、错误
├── progress.jsonl、metrics.jsonl        阶段事件、奖励/loss/物理指标
├── tensorboard/                        学习曲线事件
├── checkpoints/                        保存与恢复训练状态
├── traces/                             inputs/、phases/<阶段>/、validation/
└── summary.json、validation_history.json、selection.json
```

日志用于排障，traces用于详细复算，checkpoints用于恢复。正式数值结果在results/<同一运行名>/：iterations.csv每次新采样和PPO更新一行；episodes.csv每个完整episode一行；validation.csv每次固定开发集评价一行；evaluation.csv保存独立评价指标。每个目录带配置及来源/校验清单。仅在对应事件发生时创建文件，不预建空目录；中断前已完成的结果保留并标记运行状态。显式--output-dir用于临时验证时，结果在该运行包的results内，随临时包一起清理。

收敛图可用iteration作横轴；一个iteration是一个层的新rollout加一次PPO更新，stage/layer区分预训练、慢层和快层，不能混成同一条回报曲线。跨算法比较优先用total_slots（累计实际训练交互槽数），避免各方法每次迭代的数据量不同。原始CSV不平滑、不舍弃点；绘图时可额外展示滚动均值，并注明窗口。

记录接口分工：RunStore.event只写排障日志/TensorBoard，不创建结果目录；record_episode、record_validation、record_evaluation显式保存对应数值结果。验证同时输出validation_trajectories.csv，逐条保留轨迹种子、到达/完成数及全部评价指标，用于后续复算与配对统计，不必从日志提取。目录按首次实际写入创建，不初始化空CSV或示意曲线。

|结果表|一行对应什么|奖励与统计口径|
|---|---|---|
|iterations.csv|单层一次新rollout及随后的PPO更新|原始reward总和/物理槽数；loss按小批次样本数加权，epoch重复不增加奖励样本|
|episodes.csv|一个完整episode|raw_return为完整累计回报；不是单步均值，截断片段不当完整episode|
|validation.csv|固定开发集的一次评价|return_per_episode为轨迹回报均值；service为总完成/总到达|
|validation_trajectories.csv|某次开发评价的一条轨迹|保留轨迹种子和原始计数，支持独立复算与配对统计|
|evaluation.csv|一次独立评价|保留服务、时延、能耗分项及控制指标；不与训练loss混用|

奖励是稠密的逐物理槽负成本，慢层累计控制区间奖励；rollout负责收集，不另外制造奖励。PPO loss是训练损失，固定开发集成本/按时率/能耗才是协议学习曲线的主要性能指标。

iterations.csv保存原始reward总和/每物理槽均值、归一化训练reward、完整能耗、服务指标、实际epoch数及loss。loss是所有已执行小批次按样本数加权的均值；reward来自更新前新采样的环境，不是每个epoch产生一次。epoch重复优化同一批数据，不能将重复epoch当成独立奖励样本。validation.csv的cycle_index为完整交替轮次（预训练结束为0），iteration保留旧的组件更新计数用于兼容。确定性return_per_episode和sampled_return_per_episode分列，两者使用相同请求；采样评价使用独立且固定的动作随机种子并恢复训练RNG。best仍按确定性成本选择，未用随机评价替代部署口径。固定开发集的return_per_episode用于观察确定性策略性能；episode原始回报另存，截断片段不冒充完整episode。结果图放results/<运行名>/figures/；跨运行图表按实验主题组织并注明输入CSV，检查报告不进results。

当前HPGI：128槽一个episode；每次快层rollout为1024个transition，慢层为256个区间transition（Q4时同为1024槽）；采样后最多遍历5个PPO epoch。epoch不包含新的episode，不是重新采样。配置总量8192预训练槽＋28次慢/快交替＝65536训练槽，校准/验证另计；`--dry-run`零采样显示实际方法的用量。

每个rollout使用新建on-policy buffer，保存状态、原动作前缀/掩码、旧联合logprob、价值、奖励、后继状态和终止/截断标志；更新时复算原动作概率，GAE处理bootstrap边界，更新后清空。当前minibatch=64，快层每epoch16批、慢层每epoch4批；5个epoch是本次增加样本复用的试运行值，不是已调优结论。KL超过0.03提前结束本次更新，下一次仍重新采样。

学习率采用固定3e-4、无学习率warmup和余弦衰减，dry-run显式显示该选择。当前小型Tanh PPO先保持固定学习率以便判断采样及更新效果，不将大模型训练技巧作为必需项。已有256槽统计校准（不更新网络）及8192槽全开快层预训练；两者都不是学习率warmup。若后续实验引入衰减，须同时固定上下层进度定义、恢复状态和公平比较配置，不能在训练中途静默修改。

conditional配置默认启用TensorBoard，合成测试默认关闭。打开曲线：

```powershell
& D:/Anaconda/envs/pytorch/python.exe -m tensorboard.main --logdir logs/runs --host 127.0.0.1 --port 6006
```

浏览器访问http://127.0.0.1:6006。`ppo/upper|lower`横轴为该层累计epoch，`train/*`和`validation/*`横轴为累计训练物理槽。重点看固定验证输入的原始回报、成本与服务率，再用loss/KL/熵/梯度排障；PPO loss不要求单调下降。训练奖励尺度变化单独记录，不与原始回报混用。

training.cycles指定本次完整交替轮数，当前28轮。验证奖励或成本连续不改善不会停止训练，也不自动宣告收敛；每轮原始评价点持续保存，用于观察稳定后的表现。需要更多尾部轮次可通过同版本检查点和--extra-cycles追加，保留原有曲线，不补造数据。

仅保留显式--max-slots N/--max-seconds S预算限制、用户中断与错误退出。交互上限在完整rollout前检查，时间上限在槽/小批边界检查；中途打断保存interrupted.pt供诊断，续训使用完成阶段last.pt。KL>.03只结束本批PPO更新，不结束训练；非有限损失/梯度标记failed。

检查点保存网络、优化器、统计、随机状态、epoch计数及完成阶段。默认initial/last/final/best/selected，阶段快照关闭；显式开启须设上限。best为开发选择，final为最终状态；没有服务阈值时不能称best满足论文业务要求。正常恢复用last/initial。

完整验证套件默认系统临时空间，退出删除轨迹、报告与权重；仅显式`--keep-artifacts tmp/debug-example`保留排障现场，用完删除，不归档。检查结论必要时简短写入SETUP，详细证据可由tests和脚本重建。测试数值不进入论文性能表。
