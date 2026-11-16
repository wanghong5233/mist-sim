# 学习曲线与图表复现

绘图入口是`scripts/plotting/plot_convergence.py`，实现是`src/mist_sim/plotting.py`；只读取已完成iteration的`cycles.csv`，不加载权重、不采样环境、不更新网络。样式参数集中在`configs/plotting/convergence.json`，不为每次运行复制一份脚本。

## 样式来源与中文适配

直接调用[SciencePlots 2.2.2](https://pypi.org/project/SciencePlots/2.2.2/)（MIT，© John Garrett）的`science`、`ieee`、`no-latex`样式，没有复制其源码或样式文件。继承科研绘图的尺寸、内向刻度、细边框和无框图例；本项目覆盖离散配色及线型、90×65 mm尺寸、9 pt字号。中文优先宋体，西文Times New Roman，PNG为450 dpi，PDF嵌入字体、SVG转字体轮廓；无需额外TeX编译器。中文字体缺失时明确报错，可安装Noto Serif CJK SC或切英文。

这是一套可复用的科研样式，不是某一本中文期刊的官方模板。期刊确定后按其投稿指南调宽度、字号及格式，无需改变数据处理。最终论文优先使用PDF矢量图，PNG用于检查。图中不内嵌“图1”等论文编号。

`curve_styles`按算法label固定颜色、线型和稀疏标记，避免增加/移除曲线后颜色换位；颜色之外仍可依靠线型与标记辨认相近曲线。`marker_every`控制标记间隔，`band_alpha`控制统计阴影透明度，`legend_columns`控制图例列数；`x_tick_step=50`可让300轮图明确显示300刻度。重叠多时可设置`show_raw=false`，原始数值仍完整保留于source_data.csv。样式随plot_config保存；旧快照没有这些选项时仍按原来的颜色/线型规则重画。

六方法统一视觉映射如下，以configs中的实际值为准；后续服务率、成本及扩展实验图沿用，不按每图的曲线顺序重新分色。采用红/赭金/绿/灰/紫/蓝六类色相；主算法用砖红实线略加粗，对照使用饱和度较低、白底仍清楚的颜色，形成单一视觉重点。Fixed-Split的赭金与Upper-PPO+Greedy的蓝色相隔较远，便于辨认相近尾段；绿色偏草绿、蓝色避开青绿，紫色偏蓝紫。线型和标记同时区分，灰度打印及颜色识别差异下仍有独立辨认依据；不宣称任意色觉条件下仅凭颜色即可区分全部六线。

|算法|颜色|线型|空心标记|
|---|---|---|---|
|HPGI|砖红 #C43C39|实线，1.8 pt|圆形|
|Fixed-Split|赭金 #B58A2A|中虚线|方形|
|Single-Timescale|草绿 #4F8A5B|点划线|上三角|
|Always-On|中灰 #7A7A7A|点线|菱形|
|Fixed-Window|柔紫 #8A6BA8|短虚线|下三角|
|Upper-PPO+Greedy|蓝 #4C78A8|长虚线|叉形|

当前六线主图为150×96 mm、10 pt、对照线1.35 pt、标记3.7 pt；每25轮一个标记，各算法的`marker_offset`固定且错开，避免交叉处标记堆叠。图例放下方空白区，`legend_anchor`可微调位置。可选`inset`通过metric/bounds/xlim/ylim/xticks/yticks/title_zh/title_en指定局部放大：当前180–300轮、奖励−174至−152，展示相近的四条后期曲线；其余曲线在该纵轴范围外。放大框复制主图的相同均线、虚线和标记，不插值、不改变平滑窗口。没有共同空白区时另选位置，不遮盖实际曲线。当前单种子不画阴影，保留全部真实数据；后续确有多种子时仍沿用相同颜色，阴影alpha宜低（默认0.10），不以堆叠阴影代替线型区分。

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

## 冻结策略负载评价

入口`python -m scripts.plotting.plot_frozen_load`读取已完成A/B的原始评价与汇总，验证来源哈希、每格20条完整输入和方法间配对；生成成本/能耗/全到达按时率/已发布按时率四联预览。复用上述六方法配色与线型，Greedy增加深灰加号；不纳入不同信息口径的Nominal/Full成本。每点按整轨迹池化，5000次轨迹bootstrap误差棒不表示训练种子方差。默认中文PNG及PDF写入比较结果的figures/load，plot_data.csv和provenance保留精确点值、区间与来源；支持--language、--summary、--output和--no-errorbars。

此旧入口仅覆盖10/20/30请求每槽三个实测条件，作为粗粒度预览保留。正式完整套图使用`python -m scripts.plotting.plot_paper_results all --spec logs/launchers/paper-completion-v1/spec.json`，先核验A/B/E1/E2/T完整状态、数据/输入/权重来源及每点20条配对轨迹。负载补15/25点，匿名k2/3/4/6/8、每AP带宽20/30/40/50/60MHz均为五个实测条件；中心条件直接复用A，不重新评价。

统一入口的`base`只从A生成能耗组成、启停、分模型服务与切分图及设置/主指标表；`exact`在原源码下调用公共小例求解器及动作重放。`all`合并三组参数图、配对区间、匿名组规模/覆盖统计及串行计时表。图放比较目录的figures/F3_load至F7_model_split，表在tables；每图同目录保存PNG/PDF、plot_data.csv和plot_config.json，套图provenance记录所有源与输出哈希。

颜色沿用已确认学习曲线，Greedy使用深灰；按参数真实位置连线，不平滑、不插值。每个点5000次整轨迹bootstrap给95%区间，配对差值用于同输入比较，均不代表训练种子不确定性。信息模式不按不同估计口径的成本混排；完成时延P95仅覆盖按时任务，须同时报告失败。冻结测试证明已有策略的参数迁移表现，各条件重新训练后的性能是另一项实验。五点足以显示当前范围的粗趋势，七点不是期刊通用规定；是否加点取决于未识别阈值，而非美化曲线。
