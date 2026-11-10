# 学习曲线与图表复现

绘图入口是`scripts/plotting/plot_convergence.py`，实现是`src/mist_sim/plotting.py`；只读取已完成iteration的`cycles.csv`，不加载权重、不采样环境、不更新网络。样式参数集中在`configs/plotting/convergence.json`，不为每次运行复制一份脚本。

## 样式来源与中文适配

直接调用[SciencePlots 2.2.2](https://pypi.org/project/SciencePlots/2.2.2/)（MIT，© John Garrett）的`science`、`ieee`、`no-latex`样式，没有复制其源码或样式文件。继承科研绘图的尺寸、内向刻度、细边框和无框图例；本项目覆盖色盲友好配色及线型、90×65 mm尺寸、9 pt字号。中文优先宋体，西文Times New Roman，PNG为450 dpi，PDF嵌入字体、SVG转字体轮廓；无需额外TeX编译器。中文字体缺失时明确报错，可安装Noto Serif CJK SC或切英文。

这是一套可复用的科研样式，不是某一本中文期刊的官方模板。期刊确定后按其投稿指南调宽度、字号及格式，无需改变数据处理。最终论文优先使用PDF矢量图，PNG用于检查。图中不内嵌“图1”等论文编号。

强化学习的“独立种子曲线＋均值/离散带＋显式平滑”参考[OpenAI Spinning Up plot.py](https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/utils/plot.py)（MIT；只参考流程，没有复制旧Seaborn tsplot调用）。本项目按实际CSV语义处理，并明确采用后向滑动窗口，而非把未来轮次混入当前点的中心窗口。

## 单次运行

从工程根运行；先安装可选绘图依赖`python -m pip install -e ".[plotting]"`。本机仍使用Conda pytorch解释器；当前环境已安装SciencePlots 2.2.2。

```powershell
& D:/Anaconda/envs/pytorch/python.exe scripts/plotting/plot_convergence.py --curve "Joint PPO=results/<运行名>"
```

默认输出中文/英文奖励图和按时率图，各有PNG/PDF/SVG，位于`results/<运行名>/figures/convergence/`。`--metric reward_per_slot`只画奖励；`--window 1`关闭平滑；`--language zh`只输出中文。参数默认来自绘图JSON，可用`--config`选择另一份配置。指标也支持cost_per_slot、upper_loss、lower_loss；loss用于诊断，不用于独立判断策略收敛。

目录内只保存这次图所需的产物：

|文件|用途|
|---|---|
|reward_per_slot_zh等PNG/PDF/SVG|图形，同名更新，不按时间戳堆副本|
|source_data.csv|本次图使用的原始数值快照，保留run、seed、iteration、物理槽和指标；原cycles.csv不修改|
|aggregate.csv|实际绘制的均线、上下界和种子数；没有阴影时上下界为空|
|plot_config.json|解析后的参数及快照哈希，可直接重画相同的数据范围|
|provenance.json|源CSV及代码哈希、依赖版本、样式哈希、字体、生成命令、实际点数及原运行来源|

训练仍在写CSV时只读取完整行，保存本次实际使用的数据快照。重新运行原命令会刷新到最新完成iteration；读取保存的配置会固定使用原快照。重画命令：

```powershell
& D:/Anaconda/envs/pytorch/python.exe scripts/plotting/plot_convergence.py --config results/<运行名>/figures/convergence/plot_config.json
```

快照哈希不符时报错；不要手工修改数据美化曲线。图是CSV的派生产物，不用TensorBoard截图代替原始结果。

项目内的快照及输出路径按工程根保存为相对路径；复制项目到其他机器后，从新工程根执行重画命令即可，原训练CSV无须继续存在。原机器绝对路径只作为来源记录保留。项目外的输出仍为绝对路径，搬移时需指定新的数据/输出配置。

## 多种子与多算法

同一label下的重复`--curve`代表独立训练种子，不同label代表不同算法。相同训练种子的重复运行会报错，不把它们充当独立重复：

```powershell
& D:/Anaconda/envs/pytorch/python.exe scripts/plotting/plot_convergence.py `
  --curve "Joint PPO=results/<联合种子1>" --curve "Joint PPO=results/<联合种子2>" `
  --curve "HPGI=results/<交替种子1>" --curve "HPGI=results/<交替种子2>" `
  --x total_slots --output results/comparisons/joint-vs-hpgi/figures/convergence
```

- 单种子：原始淡线与滑动均线，**没有统计阴影**。不能把同一运行在时间窗口内的波动称为种子间不确定性。
- 多种子：每个种子先用相同窗口求均值，再在共同横坐标上聚合；默认阴影为均值±1个种子间样本标准差（ddof=1），**不是95%置信区间**。
- `--band ci95`：按独立训练种子整条重采样，默认5000次、固定bootstrap_seed，得到均值的逐点95%百分位区间；不是整条曲线的同时置信带，少量种子时不夸大精度。`--band none`关闭阴影。
- 提前停止或异长曲线：同算法只在所有种子都有数据的精确横坐标上求均值和阴影，**不外推、不补齐尾部、不复制终值**。provenance列出共同及各种子实际点数；不同采样网格无交集则报错。
- 同算法排障用iteration最直观；HPGI每轮有两批独立交互、Joint每轮一批，比较样本效率用累计物理槽`total_slots`，包含真实预训练开销，不能把相同iteration数当相同预算。

调用前核对同一组种子的场景、算法版本和训练口径；来源哈希保存在provenance。算法标签可由实验目的区分，例如“Joint PPO / plain”与“Joint PPO / finite horizon”，不把不同实现版本混在同一条均线上。

非学习策略没有训练曲线。可以在相同输入与指标口径下用独立评价结果作为水平参照线：在绘图JSON的references中填`{"label":"Greedy","metric":"reward_per_slot","value":实际数值}`并保留评价来源；不要伪造每轮训练数据，也不把异分布的评价值直接与训练采样均值比较。

等预算尾窗比较须同时固定共同的累计物理槽终点和窗口槽数。例如HPGI每轮2048槽、Joint每轮1024槽时，同一40960槽窗口分别包含20轮和40轮，不能直接比较双方各自的“最近20轮”。完整训练终点的尾段成绩仍可并列，但须同时列出总交互量。`cycles.csv`的service是该轮完成数/到达数；对多轮service取均值称“轮次平均按时率”，若要总体按时率须从原始计数求总完成/总到达，不能将二者混称。

## 点数与训练是否继续

每个**完整iteration对应一个真实点**；PPO epoch、小批与同一批数据的反复优化不是新增横坐标。没有期刊通用的“至少多少点”规定。约100–300个点通常已足够呈现本项目这样的初期改善和后期平台，这是展示分辨率的经验范围，不是收敛阈值；保留全部点，不抽样到固定数量或插值增加点数。

不同算法的曲线允许在不同横坐标结束。某基线继续训练不要求其他算法增加同样的点数；没有真实观测就不延长水平线。固定预算比较使用共同物理交互预算，充分训练比较则允许不同预算并明确报告，是否续训由各方法自身进展决定。达到配置轮数上限只是预算结束，须核对stop_reason和稳定判据后才描述为平台或稳定早停。

是否继续训练看未平滑奖励的窗口均值、趋势与波动，以及服务/成本，沿用现有稳定判据和尾部预算；不因图看起来不够长就加训练。10轮平滑只是展示选择，可用1/5/10检查趋势是否依赖窗口。单次曲线出现平台后，多独立种子和共同测试输入通常比单纯延长一条平坦曲线更能补充证据；3种子可先看稳定性，5种子是后续正式比较的实用起点，数量仍取决于方差与预算。

当前cycles.csv是采样批次的原始物理reward/slot及按时率，**不是固定验证集或测试集成绩**。预训练与正式iteration分开，未记录的iteration=0不补造。平滑、归一化训练reward和原始物理reward也分别命名；论文图注明口径及种子数。
