# MIST 仿真实验工程

`mist-sim/`是独立仿真工程与Git仓库根，`src/mist_sim/`是Python导入包。MIST是系统框架，HPGI名称及双层PPO/P3设计来自导师V6初稿；joint_ppo为后续联合更新对照。另加动作搜索的改进试验使用历史代码名HARP（method=harp），现已搁置，仅保留可选实现与诊断来源。全部方法共用环境及训练/评价基础设施。初稿提供设计假设，不作为算法正确性标准；主算法效果由实际对照决定。

## 仓库与隐私

远端为私有仓库`wanghong5233/mist-sim`。只提交源码、配置、测试及必要说明；results、logs、data中的实测资产、权重、原始图像、实验协议和本机执行记录均忽略，论文不在此仓库内。推送前执行`python scripts/maintenance/check_publish.py`，检查当前暂存内容与历史；首次建库不导入科研根仓库历史。Git提交使用GitHub noreply邮箱。

本机布局为`02_实验/mist-sim/`。科研父仓库以子模块记录这个独立仓库的提交，不重复管理源码；提交/推送仿真时进入mist-sim。克隆本仓库无需导师论文或MERA检出即可运行合成输入检查；数据恢复见[data说明](data/README.md)。父仓库只在本地更新引用，不随仿真push上传。

## 阅读入口

|文件|职责|
|---|---|
|[IMPLEMENTATION.md](IMPLEMENTATION.md)|唯一当前实现规格：模型、state/action张量、奖励、PPO、轨迹长度与更新节奏|
|[强化学习排障与经验复用](docs/RL_TROUBLESHOOTING.md)|已经定位的问题、证据入口及向HPGI/学习基线迁移的检查顺序|
|[学习曲线与图表复现](docs/PLOTTING.md)|SciencePlots中文/英文样式、真实CSV绘图、多种子阴影与图表来源管理|
|[data/README.md](data/README.md)|本地剖面、原始输入和恢复边界|
|[scripts/README.md](scripts/README.md)|可执行脚本索引|
|[AGENTS.md](AGENTS.md)|执行与文档维护约束|

README管导航和命令；IMPLEMENTATION管当前语义；配置JSON管实际值。实验协议和SETUP保留在本机，不上传。修改替换旧描述、链接复用，Git负责源码历史。

论文补证据入口为 `python -m scripts.evaluation.strengthen_evidence prepare/preflight/static/latency`，依次冻结设计、预检、静态选择与独立测试、控制预留补评。已完成运行不得重新 prepare。结束后运行 `python -m scripts.evaluation.report_strengthening diagnose` 执行条件触发的交叉诊断，`report` 核验全部候选、输入、权重与指标并更新既有图表入口；这些命令不更新学习权重。运行设计和原始结果仅在本机保存，具体用途见[脚本索引](scripts/README.md)。

## 目录

```text
mist-sim/
├── configs/       环境/算法JSON及组合入口
├── src/mist_sim/  唯一可安装源码；环境、PPO、基线、数据与指标
├── tests/         手算、反例、训练及恢复回归测试
├── scripts/       data/数据处理，evaluation/评价，training/训练编排，validation/核验，maintenance/维护，plotting/绘图
├── docs/          强化学习排障、图表统计及复现说明
├── data/          raw/原始输入，profiling/剖析及来源，processed/仿真数值输入
├── logs/runs/     正式训练/评价的运行记录，执行时创建
├── results/       每次运行的原始结果CSV，以及据此生成的统计表和论文图
└── references/    只读上游参考，不是运行依赖
```

采用[Python标准src布局](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)。日志采用“一个运行一个目录”：PyTorch的[SummaryWriter](https://docs.pytorch.org/docs/stable/tensorboard.html)支持指定log_dir，[CleanRL PPO](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo.py)按run_name隔离事件。本项目统一指定logs/runs，不依赖工具的默认顶层runs目录；该路径是项目约定，并非唯一行业标准。

测试文件和权重在系统临时目录，退出清理；检查/维护脚本默认只打印，确需排障才显式指定输出到tmp。数据剖析归data，不算强化学习实验结果。尚未正式训练时logs/runs与results不放占位文件，也不提前生成空分类。正式训练成功/失败均保留，不按性能好坏删除。conditional正式训练默认关闭整批状态/动作/物理/更新调试JSON（diagnostics.phase_interval=0）；PPO buffer留在内存，绘图CSV、简要日志及低频检查点照常保存。显式排障才设置正整数间隔。

## 由结果CSV绘图

安装可选依赖`python -m pip install -e ".[plotting]"`，从工程根运行：

```powershell
& D:/Anaconda/envs/pytorch/python.exe scripts/plotting/plot_convergence.py --curve "Joint PPO=results/<运行名>"
```

图片、实际绘图数据快照、聚合CSV、配置及来源清单放`results/<运行名>/figures/convergence/`；一个完整iteration一个点，默认10轮后向均值并保留原始淡线。重复`--curve "算法名=运行目录"`可聚合同算法的独立种子；多算法比较可加`--x total_slots`及`--output results/comparisons/<比较名>/figures/convergence`。单种子不画统计阴影；多种子默认±1标准差，`--band ci95`为种子bootstrap逐点95%区间。中文/英文、指标、尺寸及格式由[绘图配置](configs/plotting/convergence.json)控制，详细语义与重画命令见[绘图说明](docs/PLOTTING.md)。

## 环境与输入

本机唯一运行解释器：`D:/Anaconda/envs/pytorch/python.exe`（Conda pytorch）。不激活旧.venv、不重新安装CUDA。Python3.10中文editable路径按cp936读取，不设置`PYTHONUTF8`或`-X utf8`。首次执行：

```powershell
& D:/Anaconda/envs/pytorch/python.exe scripts/maintenance/check_environment.py
```

确认executable及package_path。后续命令同样使用该绝对解释器；其他机器安装`pip install -e ".[monitoring]"`，开发测试用`".[test]"`。本机TensorBoard已安装。环境/采样固定CPU，PPO更新支持--device cpu或cuda；CPU仍为跨机器默认，CUDA不可用时报错。仿真只查数值剖面，不加载三个视觉模型，不需训练视觉网络。

普通安装同样支持：从克隆的mist-sim根目录运行，CLI优先使用当前工程配置；从其他目录运行时显式传`--root 工程路径`。验证套件从工程根运行，不假定机器上存在论文目录。

`configs/smoke.json`是随仓库提供的手算合成测试；`configs/conditional.json`引用本机ResNet50/ResNet152/ViT-B32结构剖面，8服务器、40终端等资源/业务/功率为条件假设。本地来源见data/SCENARIO.md和data/LOCAL_SOURCES.md；这些私有资产不随clone分发。

## 方法与模块

**搁置的HPGI改进试验（历史代码名HARP）**保留`--method harp`及`configs/algorithms/harp.json`入口以便复算，不作为推荐主方法。上层PPO控制启停，下层PPO在离线固定点集合上提出接纳/本地/服务器方案，联合求解器在完整动态集合上改进接纳、关联和切分，并调用P3比较成本。默认一次坐标扫描和接纳/释放交换；只接受联合可行、严格降成本且不减少总接纳数的操作。PPO使用原提案及概率，环境执行改进动作。它没有新增第三个网络，仍复用原双层网络、PPO更新与CPU采样。

代码通过`policies/registry.py:make_controller`分派：`harp.py:HARPController`负责改进试验，`hpgi.py:HPGIController`负责原HPGI，`base.py`仅共用网络、慢层控制和权重接口。Trainer、CLI、worker、评价与检查点均使用同一注册入口。方法名进入配置、manifest和自动运行ID；新运行分别写入`logs/runs/<运行ID>`及`results/<同一运行ID>`，不另建一套环境或日志系统。

历史`hpgi_split_refined.json`与`hpgi_assignment_refined.json`保留原内容，由`legacy_refined.py`明确兼容；它们是旧混合实验，不能当作原版HPGI或改名后的HARP正式结果。新HARP使用独立预设；`harp_finite_horizon.json`另选择共享finite_horizon/sampled_return价值估计。HARP本身不加载任何成熟权重，只有显式`--transfer-from`才迁移。

HARP依赖当前场景的离线选点：须叠加`select_fixed_split`导出的`selected_overlay.json`并保留其凭据；原演示固定点不用于科研实验。成熟网络开发继承父配置冻结的选点；算力、带宽、负载等改变后须重新选择。切分求解有额外计算成本，收益和耗时需同时报告。原HPGI已经包含P3；HARP增加的是离散联合改进。

显式迁移与严格恢复分开。`train --config <新配置> --transfer-from <父last.pt> --parent-source-revision <已审计完整提交> --seed <新种子> --run-id <新名称>`继承网络/归一化/价值单位，重置Adam、RNG及训练计数，跳过已继承预训练，并记录父哈希/配置及成本。它是新实验，不能把开发曲线当独立从头训练结果；物理/奖励/表示及折扣不兼容会拒绝。`--checkpoint`仍严格检查源码与配置，不能用来改参续训。限定5轮的配对开发入口见[scripts](scripts/README.md)。

|方法键|实际差异|训练组件|
|---|---|---|
|harp|固定点PPO提案、动态联合指派改进、容量掩码及P3|开关/调度交替PPO|
|hpgi|容量掩码、P3|开关/调度交替PPO|
|joint_ppo|同轨迹阶段条件联合PPO；保留Q槽开关和逐槽调度|开关/调度联合PPO|
|always_on|服务器全开|仅调度PPO|
|fixed_split|逐模型固定合法切分|开关/调度交替PPO|
|single_timescale|Q=1，每槽重选开关及调度，保留长期回报|开关/调度交替PPO|
|fixed_window|无联合容量掩码、学习速率、执行前投影|开关/调度交替PPO，含速率密度|
|upper_ppo_greedy|容量感知增量成本贪心调度|仅开关PPO，无调度预训练|
|greedy_consolidation|上一完整区间驱动启停、当前公开候选贪心|无需训练，直接evaluate|
|nominal|区间中心构造候选，报告真实违约|开关/调度交替PPO|
|full_information|同发布集合的理想真实信息参照|开关/调度交替PPO|

`environment/anonymity/profiles/allocation`负责物理和信息边界；`policies、buffers、ppo、training`负责策略及学习；`evaluation/storage/diagnostics`负责核算、证据及复算。`reference_solvers`为有界离线/单隙枚举，完整搜索才称最优。MERA只作参考，详见[来源说明](references/README.md)。

Greedy属于无需训练的因果启发式，仍逐槽与同一环境交互；相同输入不需要重复策略种子，不同请求轨迹用于衡量负载波动。Always-On、Fixed-Split、Single-Timescale等仍训练PPO，同时承担开关、切分、控制时间尺度的机制消融；Full-Information为理想信息参照，单列其信息优势。非学习方法报告评价指标，不伪造训练收敛曲线。

共享请求输入的贪心批次入口：`python scripts/evaluation/evaluate_greedy.py --config configs/conditional.json --suite data/processed/evaluation/conditional-main-comparison-v1 --run-id evaluate-greedy-shared20`。默认20条新128槽输入、4个CPU进程；输入种子与已记录训练输入隔离，首次生成后固定文件哈希。后续已训练方法应读取同一输入，完整负载与初始状态保持一致；这是共同比较集，尚非最终盲测。

Always-On公平物理预算入口：`python -m mist_sim train --config configs/conditional.json --overlay configs/algorithms/always_on_finite_horizon.json --overlay configs/algorithms/cuda_update.json --overlay configs/training/fixed_200.json --overlay configs/runtime/windows_performance.json --seed 4200001`。它只训练调度网络，2048槽/iteration、8192槽预训练、200轮共417792训练槽，与HPGI总槽数对齐；不代表相同梯度步数。独立评价关闭，校准256槽另计；--dry-run可只查看用量。

## 执行命令

在本目录执行。纯代码克隆可直接跑pytest和smoke检查；缺少私有剖面时一项数据集成测试明确skip，其余不跳过。conditional相关命令需要先恢复本机输入。

```powershell
& D:/Anaconda/envs/pytorch/python.exe -m pytest tests -q
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim check --config configs/smoke.json
& D:/Anaconda/envs/pytorch/python.exe scripts/maintenance/check_publish.py
& D:/Anaconda/envs/pytorch/python.exe scripts/validation/check_rollouts.py
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim.verification
& D:/Anaconda/envs/pytorch/python.exe scripts/validation/check_profile_chain.py
& D:/Anaconda/envs/pytorch/python.exe scripts/validation/check_training_scenario.py --witness --exact --seed 4100017
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim train --config configs/conditional.json --method hpgi --dry-run
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim train --config configs/conditional.json --method hpgi --seed 4200001
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim train --config configs/conditional.json --overlay configs/algorithms/cuda_update.json --method hpgi --seed 4200001
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim train --config configs/conditional.json --method joint_ppo --seed 4200001
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim train --config configs/conditional.json --overlay configs/algorithms/cuda_update.json --method joint_ppo --seed 4200001
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim train --config configs/conditional.json --overlay configs/algorithms/cuda_update.json --overlay configs/algorithms/joint_finite_horizon.json --seed 4200001
& D:/Anaconda/envs/pytorch/python.exe scripts/training/compare_mechanisms.py --config configs/conditional.json --slots 32768 --seed 4200001
& D:/Anaconda/envs/pytorch/python.exe scripts/training/compare_mechanisms.py --config configs/conditional.json --overlay configs/algorithms/cuda_update.json --slots 32768 --seed 4200001
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim evaluate --checkpoint logs/runs/<训练ID>/checkpoints/last.pt --seed 4400019
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim replay --checkpoint logs/runs/<训练ID>/checkpoints/last.pt --records logs/runs/<评价ID>/traces/evaluation.json
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim summarize --records logs/runs/<评价ID>/traces/evaluation.json
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim train --checkpoint logs/runs/<训练ID>/checkpoints/last.pt --extra-cycles 1
```

占位ID替换为实际运行路径；学习方法必须加载自己的权重。`--overlay 文件.json`和`--set ppo.learning_rate=0.0001`严格覆盖配置，未知键报错；上下层可通过`--set ppo_layers.upper.minibatch=64`独立配置。变更源码/配置/剖面后拒绝无说明续训。

HARP的无采样入口检查可用：

```powershell
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim train --config configs/smoke.json --method harp --dry-run
```

正式场景先确认选点结果，用实际选择ID替换下列占位，`--dry-run`只预览、不建运行目录：

```powershell
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim train --config configs/conditional.json --overlay results/<选择ID>/selected_overlay.json --overlay configs/algorithms/harp_finite_horizon.json --overlay configs/algorithms/cuda_update.json --overlay configs/training/fixed_300.json --overlay configs/runtime/windows_performance.json --set checkpoint_policy.cycle_interval=10 --seed 4200001 --dry-run
```

确认实验预算后移除`--dry-run`才正式启动；不传`--checkpoint/--transfer-from`即从头初始化。通用`--method harp`安装算法必需的求解设置；切换离开HARP会清除其求解标志，防止污染基线。旧配置不加新默认键，旧运行不得靠重标方法恢复；源码签名改变后，继续旧实验须使用其原版匹配checkout。

固定200轮的HPGI及Windows后台运行试验：

```powershell
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim train --config configs/conditional.json --overlay configs/algorithms/cuda_update.json --overlay configs/algorithms/hpgi_finite_horizon.json --overlay configs/training/fixed_200.json --overlay configs/runtime/windows_performance.json --seed 4200001
```

fixed_200将完整iteration预算设为200，并把早停资格起点设为200，确保不会在200之前因稳定退出；后续学习方法的收敛曲线对比统一叠加该预设。其他配置仍保留原稳定早停行为。200轮HPGI为8192预训练＋200×2048=417792训练时隙，校准另计；与Joint相同轮数不等于相同交互预算。旧199点结果保持真实记录，不复制终值补点。

需要夜间接续时使用[scripts/training/overnight_queue.py](scripts/training/overnight_queue.py)，参数见[脚本索引](scripts/README.md)。它按实际完成状态等待，冻结候选配置和源码，选法后两个可见终端独立跑新种子，各200轮；队列状态、配置及失败记录在logs/queues/<队列名>。两训练共用GPU，会竞争计算资源；尾20轮训练成本筛选仅为初步选法，不冒充共同独立测试结果。新种子从头训练，不能把旧权重换种子续训当作独立重复。运行期间保持插电、盖子打开；队列请求系统唤醒允许屏幕熄灭，不覆盖手动睡眠/合盖策略。两任务结束后自动生成实际三种子的曲线，异长不补点。

windows_performance仅在Windows显式启用：主进程及每个采样worker各自设置AboveNormal优先级，并通过SetProcessInformation关闭EXECUTION_SPEED节能（HighQoS），随后读取实际状态确认；无需管理员，不改全局电源计划。主进程状态在process_policy.json，worker状态在progress.jsonl的collection_timing事件jobs[*].process_policy；终端前后台均保持这些进程设置，但不能保证消除其他负载/温控造成的波动。平台不支持或设置失败时明确报错；不启用该overlay时不改变OS调度。机制依据：[微软QoS](https://learn.microsoft.com/en-us/windows/win32/procthread/quality-of-service)、[进程API](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessinformation)。

其余学习基线复用同一train入口，通过--method选择表中方法；CPU采样、进程池、CUDA更新、整批张量缓存及诊断间隔均共用。单层方法只打印实际更新的组件。示例：

Fixed-Split正式入口包含一次离线开发选点与独立PPO训练。先运行`python -m scripts.evaluation.select_fixed_split --suite data/processed/development/fixed-split-v1 --run-id select-fixed-split-v1`，穷举各模型合法点的全部组合，在相同开发输入与固定因果控制器下按原目标成本选择；不保证所有可能学习策略下全局最优。结果`results/<选择ID>/selected_overlay.json`作为该场景的固定配置，后续不同训练种子/轮数复用，不逐次重选。随后运行`python -m scripts.training.train_fixed_split --selection-run select-fixed-split-v1 --suite data/processed/development/fixed-split-v1 --run-id train-fixed-split-selected-v1`，等待选择成功、校验输入/选择哈希、冻结逐模型切分，从头训练300轮；再次训练使用同一选择ID、新训练ID及`--seed`即可，可重复传`--wait-run 运行ID`等待其中一项300轮训练退出后再开可见终端。旧3/3/7仅为入口示例，旧权重不能直接沿用为重新选择后的结果。完整候选/排序/overlay归选择运行results，后续训练另存同名results/logs。

固定点适用于选点时的模型剖面、物理场景和目标权重。后续改变设备算力、内存、带宽、负载分布或成本权重做敏感性实验时，应在新场景的开发输入上重新选点、另存配置，再训练和评价；`train_fixed_split.py`会拒用与开发场景不一致的选择。不根据最终测试结果调整切分点；若专门考察原固定点的跨场景泛化，则明确标为冻结迁移对照。

```powershell
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim train --config configs/conditional.json --overlay configs/algorithms/cuda_update.json --method fixed_split --set ppo_layers.upper.minibatch=64 --dry-run
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim evaluate --config configs/conditional.json --method greedy_consolidation --seed 4400019
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim evaluate --checkpoint logs/runs/<训练ID>/checkpoints/last.pt --trace logs/runs/<输入运行ID>/traces/inputs/<输入文件>.json
```

check/evaluate/reference的--trace接受逐槽请求列表或训练保存的`{seed, trace}`文件；记录原路径、SHA-256及文件内种子，不用命令默认种子冒充输入来源。裸列表的轨迹种子记为空。所有方法可用同一输入评价；续训不能通过--method将已有权重改称另一基线。Q=1、仅开关或仅调度的方法每轮物理槽数不同，比较预算看--dry-run及CSV中的实际槽数。

只切换更新设备可用--device cuda；cuda_update.json另外选择minibatch128及4个CPU采样进程，不改变1024物理槽rollout。进程数可通过--set runtime.collection_workers=8等覆盖，benchmark_collection.py可在同输入下重新测量；本机测量波动较大，4进程不是跨机器最优保证。大块输入按类型整批上传，在多个epoch与KL检查间复用；更新结束后策略及Adam状态回CPU以供采样/保存。相同源码检查点可通过--checkpoint 路径 --device cpu/cuda跨设备继续，其他超参数仍需一致；旧源码检查点不会自动冒充兼容续训。GPU运行记录device_metrics（含搬运/回传/峰值显存）与input_pack_transfer_seconds，整轮效率以实际phase_seconds判断。

`joint_ppo`采用公共ppo设置（不接受冲突的ppo_layers覆盖）：一个iteration只采样fast_slots个物理槽，两个组件共同更新；slow_intervals不控制该方法的采样预算。默认1024槽对应1280个事件transition，最多5epoch、每小批64个事件。hpgi仍分别采样两个1024槽。比较脚本先共同校准/预训练一次，再从相同权重、Adam状态、归一化和随机状态出发；--slots指每种方法预训练之后的物理交互预算，须整除两者iteration大小，32768对应HPGI16轮和joint_ppo32轮，不是收敛轮数。8worker取自conditional配置。默认没有独立评价，只有显式--evaluation-trajectories N才在每种方法结束后各做一次共同开发输入评价，不逐轮额外采样。

`ppo.advantage_normalization`默认global，保留全rollout共同标准化作为对照；`configs/algorithms/joint_phase_normalized.json`选择joint_ppo及phase，分别对开关/调度事件的actor优势减均值、除标准差，避免两个critic的不同误差尺度混入同一标准化。原始reward、跨phase GAE和critic目标不变；优势在更新前计算一次，所有epoch复用。此选项改变组件间的策略梯度尺度，不是已证明的收敛修复。GPU配置可组合预览：

```powershell
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim train --config configs/conditional.json --overlay configs/algorithms/cuda_update.json --overlay configs/algorithms/joint_phase_normalized.json --seed 4200001 --dry-run
```

HPGI与joint_ppo共用采样缓存、剖面/P3优化、CPU进程池和CUDA更新；cuda_update预设适用于两者。HPGI的两个buffer本来就分别标准化优势，global与phase对单层buffer等价，无需额外移植分组逻辑。CUDA128下HPGI每阶段仍采样256个开关或1024个调度transition，每epoch分别2/8小批、最多10/40次Adam更新；上层样本较少，128并非已验证的最佳学习batch，可通过ppo_layers.upper.minibatch独立调整。两方法比较入口支持相同--overlay，避免只给其中一方应用加速配置。

等物理槽不等于等优化计算量：默认每2048槽，HPGI最多100个Adam步骤，joint_ppo为200个。comparison.csv并列实际训练transition、样本使用次数、优化步数与分项耗时；两者统一冷启动采样池。不能由“只采样一次”推断联合方法整体更快，详细口径见IMPLEMENTATION。

共同预训练权重只在HPGI运行的checkpoints/common_pretrain.pt保存一份，两个manifest引用同一哈希，续训仍使用各自完整检查点。结果保留各自cycles.csv、iterations.csv和comparison.csv；joint_ppo额外components.csv记录两组件loss/KL/参数变化，系统reward不拆成两个回报。训练reward按实际物理槽平均；事件reward均值另记training_event_reward_mean，不能用事件数稀释系统reward。比较图同时提供物理交互量与耗时横轴。

场景检查命令不更新网络：先计算单请求可行上界，--witness构造并重放完整调度，--exact穷举预先定义的两服务器小网格及所有合法固定切分。默认仅打印，搜索不完整会返回失败且不报告最优差距。它检查“存在可服务和改善成本的选择”，不保证HPGI优于全部基线。当前场景期限[0.8,1.0]秒、每请求10MHz/每AP40MHz；推导与条件假设见data/SCENARIO.md。状态结构已变更，旧210维评分网络的检查点不能续训为当前270维网络。

## 日志与资产

`joint_finite_horizon.json`选择联合PPO、按组件标准化优势、有限时域critic和采样回报目标。critic由过去物理奖励均值乘剩余折扣槽数建立基线，再学习随剩余时域缩小的状态残差；最后一槽仍估计该槽奖励，只有剩余零槽时价值为零。critic目标用实际采样奖励累加，真实终止归零，采样截断使用末端bootstrap；actor继续使用跨组件GAE。它针对尾部价值偏差，不改变环境奖励、状态字段或actor容量。默认plain/gae及joint_phase_normalized对照仍可选；HPGI可通过`--set ppo.value_model=finite_horizon --set ppo.value_target=sampled_return`使用相同估计器。不同估计器须新建运行，不能直接续训旧源码快照。

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

conditional训练关闭独立评价（validation.enabled=false）：没有initial、逐轮或final额外轨迹。一个终端iteration是完整慢/快交替；cycles.csv每轮一行，平均奖励为两段实际采样reward之和/物理槽数之和，两层reward和loss分列。它描述更新过程中采样的行为策略，不冒充更新后固定策略的测试成绩。原始CSV不平滑；iterations.csv仍保留组件级更新计数。独立evaluate命令用于后续确有需要的测试。

记录接口分工：RunStore.event只写排障日志/TensorBoard，不创建结果目录；record_episode、record_validation、record_evaluation显式保存对应数值结果。验证同时输出validation_trajectories.csv，逐条保留轨迹种子、到达/完成数及全部评价指标，用于后续复算与配对统计，不必从日志提取。目录按首次实际写入创建，不初始化空CSV或示意曲线。

|结果表|一行对应什么|奖励与统计口径|
|---|---|---|
|iterations.csv|单层一次新rollout及随后的PPO更新|原始reward总和/物理槽数；loss按小批次样本数加权，epoch重复不增加奖励样本|
|cycles.csv|无独立评价时的一次完整交替|两层采样reward按实际物理槽数加权，分列两层reward/loss及采样/PPO耗时|
|episodes.csv|一个完整episode|raw_return为完整累计回报；不是单步均值，截断片段不当完整episode|
|validation.csv|固定开发集的一次评价|return_per_episode为轨迹回报均值；service为总完成/总到达|
|validation_trajectories.csv|某次开发评价的一条轨迹|保留轨迹种子和原始计数，支持独立复算与配对统计|
|evaluation.csv|一次独立评价|保留服务、时延、能耗分项及控制指标；不与训练loss混用|

奖励是稠密的逐物理槽负成本，慢层累计控制区间奖励；rollout负责收集，不另外制造奖励。PPO loss是训练损失，固定开发集成本/按时率/能耗才是协议学习曲线的主要性能指标。

iterations.csv保存原始reward总和/每物理槽均值、归一化训练reward、完整能耗、服务指标、实际epoch数及loss。loss是所有已执行小批次按样本数加权的均值；reward来自更新前新采样的环境，不是每个epoch产生一次。epoch重复优化同一批数据，不能将重复epoch当成独立奖励样本。validation.csv的cycle_index为完整交替轮次（预训练结束为0），iteration保留旧的组件更新计数用于兼容。确定性return_per_episode和sampled_return_per_episode分列，两者使用相同请求；采样评价使用独立且固定的动作随机种子并恢复训练RNG。best仍按确定性成本选择，未用随机评价替代部署口径。固定开发集的return_per_episode用于观察确定性策略性能；episode原始回报另存，截断片段不冒充完整episode。结果图放results/<运行名>/figures/；跨运行图表按实验主题组织并注明输入CSV，检查报告不进results。

当前HPGI：128槽一个episode；每次快层rollout为1024个transition，慢层为256个区间transition（Q4时同为1024槽）；采样后最多遍历5个PPO epoch。epoch不包含新的episode，不是重新采样。上限为8192预训练槽＋3000次慢/快交替＝6152192训练槽，校准/验证另计；奖励稳定可提前停止，`--dry-run`零采样显示实际方法的上限。

conditional基础配置的runtime.collection_workers=8，cuda_update覆盖为4；每批仍为8条完整系统轨迹。一条轨迹内两组件共同决策，只有一份逐槽系统奖励。HPGI交替更新：旧开关/旧调度采样后更新开关，新开关/旧调度重新采样后更新调度；joint_ppo从同一批轨迹联合更新两组件。两种机制均已实现，不由结构推断有效性。

采样进程跨批常驻，每进程PyTorch线程1，主进程收齐后才做PPO；0为旧串行路径，1为单worker的逐episode参考。请求种子和独立动作种子按episode编号分配，按提交顺序合并，不依赖完成快慢；改变worker数量不改变同一批结果，恢复沿trace_counter继续。该独立动作随机流与旧串行连续随机流不同，不宣称旧运行逐动作复现。每条episode内部保留时序和GAE边界，未缩短128槽目标。正常结束/异常关闭进程池；直接调用collect的工具须close。

每个rollout使用新建on-policy buffer，保存状态、原动作前缀/掩码、旧联合logprob、价值、奖励、后继状态和终止/截断标志；更新时复算原动作概率，GAE处理bootstrap边界，更新后清空。当前minibatch=64，快层每epoch16批、慢层每epoch4批；5个epoch是本次增加样本复用的试运行值，不是已调优结论。KL超过0.03提前结束本次更新，下一次仍重新采样。

学习率采用固定3e-4、无学习率warmup和余弦衰减，dry-run显式显示该选择。当前小型Tanh PPO先保持固定学习率以便判断采样及更新效果，不将大模型训练技巧作为必需项。已有256槽统计校准（不更新网络）及8192槽全开快层预训练；两者都不是学习率warmup。若后续实验引入衰减，须同时固定上下层进度定义、恢复状态和公平比较配置，不能在训练中途静默修改。

conditional配置默认启用TensorBoard，合成测试默认关闭。打开曲线：

```powershell
& D:/Anaconda/envs/pytorch/python.exe -m tensorboard.main --logdir logs/runs --host 127.0.0.1 --port 6006
```

浏览器访问http://127.0.0.1:6006。`ppo/upper|lower`横轴为该层累计epoch，`train/*`和`validation/*`横轴为累计训练物理槽。重点看固定验证输入的原始回报、成本与服务率，再用loss/KL/熵/梯度排障；PPO loss不要求单调下降。训练奖励尺度变化单独记录，不与原始回报混用。

training.cycles上限3000。conditional的convergence直接使用训练采样原始reward/slot：第100轮起比较相邻两个20轮窗口，均值相对差≤0.5%、趋势相对跨度≤1%、最近20轮相对标准差≤2%；总体与两层分别须满足，避免相反趋势抵消。连续5次满足后继续20轮，不稳定则重置，最早124轮停止。分母max(abs(前窗口均值),1)，预训练不参与；不以PPO loss或连续未刷新最佳值停止。随机采样曲线的平台不证明最优或成功学习。cycle_history及窗口计数进入检查点，convergence.csv保存判据；--extra-cycles不绕过已满足的停止条件。其他配置显式启用评价时仍可用评价回报判停。

此外保留显式--max-slots N/--max-seconds S预算限制、用户中断与错误退出。交互上限在完整rollout前检查，时间上限在槽/小批边界检查；中途打断保存interrupted.pt供诊断，续训使用最近已保存的last.pt。低频保存期间的更新可能需要重跑。KL>.03只结束本批PPO更新，不结束训练；非有限损失/梯度标记failed。

conditional使用checkpoint_policy.mode=minimal：只覆盖last.pt，在启动、预训练完成、每10个完整iteration和正常结束保存，不逐组件/逐轮保存，不保存best或selected副本；最佳评价仅记录数值，不能据此声称存在相应权重。检查点保留网络、优化器、统计及随机状态供中断恢复与后续评价；完整曲线CSV每轮照常保存。其他配置默认full模式保持initial/last/final/best/selected语义，阶段快照须显式开启并设保留上限。

更新前的行为概率检查先批量复算；误差超过1e−5时，仅对异常样本按采样路径逐条复算。逐条结果仍须满足1e−5，批量差额另受float32舍入界约束；非有限值或真实失配继续报错。此修复共用于HPGI、Joint PPO与学习基线，不改变PPO概率或梯度。源码签名不匹配仍禁止通用续训；只有明确证明未进入更新、且逐项审计学习状态和源码变更的故障，才可独立迁移恢复。新运行CSV保留原iteration编号，绘图合并须保留两段来源及未使用采样成本。

直接关闭终端可能终止训练，来不及保存当前状态。正常恢复用最近完整last.pt创建新运行；原运行在检查点之后的CSV保留作中断记录，合并曲线时以检查点为界选取前缀，再接新运行后缀，禁止重复计数或拼接不属于同一权重轨迹的旧尾部。恢复编排脚本的--restart-run支持本次Always-On接续，检查父检查点哈希、配置及种子，记录被替代轮次和重复交互成本；不修改原始CSV。

conditional每10次组件更新保存一次完整behavior/update/physical诊断；HPGI分别按开关/调度更新次数计数，避免固定间隔漏掉一侧。保留最近2组及最近2条输入；不再每阶段写巨大JSON再清理。input_seeds.jsonl保留全部种子，所有训练CSV/梯度/KL摘要每轮照常记录。未记录或已清理阶段不支持逐动作直接重放。独立评价关闭时不创建validation结果或选择最佳权重。

完整验证套件默认系统临时空间，退出删除轨迹、报告与权重；仅显式`--keep-artifacts tmp/debug-example`保留排障现场，用完删除，不归档。检查结论必要时简短写入SETUP，详细证据可由tests和脚本重建。测试数值不进入论文性能表。
