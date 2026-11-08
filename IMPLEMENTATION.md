# MIST 仿真实现规格

本文件描述当前实现、接口和边界，历史由Git保存。模型对应导师V6小修工作稿；本地布局中的工作稿为`../../03_论文/MIST/main.tex`，不随代码分发。执行记录、实验协议和结果核验留在本机；本文件不依赖这些私有资产才能阅读。

## 环境与论文对应

|职责|实现|约束|
|---|---|---|
|匿名发布|anonymity.py|同模型/服务类/公开权重内逐级泛化，区间覆盖真实值，不足k抑制；短期标识不包含终端身份|
|物理推进|environment.py|慢层读历史→发布当前请求→逐隙ready=旧激活×新激活→快层→分配与执行；只有执行推进一隙|
|任务候选|environment.py、profiles.py|拒绝、本地、合法内部切分；最低边缘速率=边缘工作量/(期限下界−非边缘耗时)，分母非正则不可行|
|资源|PublicProblem、allocation.py|计算/显存/AP带宽独立约束；本地不占边缘资源；P3以双精度单调根和对偶二分求解|
|真实执行|environment.py|无跨槽队列；超期取消剩余阶段，只计已执行能耗；真实内存超限的服务器本批后端准入失败|
|评价|evaluation.py|完成+拒绝+抑制+超期+资源失败=全部到达；总体按时率=总完成/总到达，零分母返回空值|

控制器只接收公开对象及允许的历史反馈，不能访问当前未发布请求。真实属性由环境执行层持有。保守候选逐规格取上界，不假定剖面单调。Nominal使用区间中心；Full-Information只精确化同一已发布集合，保持共同抑制集合，单列为理想参照。

静态能耗每槽计一次、唤醒仅0→1计一次。保守调度成本不因实际超期取消而回写。默认控制耗时为条件假设，Python实测耗时另记；measured-latency模式显式计入超出预留的开销，不能作为设备部署测量。本地DNN剖面及系统参数来源见data/SCENARIO.md，获取边界见[data说明](data/README.md)。

## State、动作与网络

|张量（float32/CPU）|快层|慢层|
|---|---|---|
|items `[行数,19]`|全批候选：动作类型、剖面成本、期限/开销、资源及剩余比例|2H个开关条目及历史公开请求的需求特征|
|servers `[H,13]`|服务器资源/功耗、激活/就绪、历史利用等|同类公开服务器特征|
|context `[11]`|当前公开统计、历史反馈、控制相位与剩余时间|历史统计、控制相位与剩余时间|
|prefix `[7]`|处理进度、累计资源和接纳数|已选服务器开关的数量/比例|
|actor输出|每个请求K个条件候选分数，非法项掩码为零概率|每服务器2个条件开关分数|
|critic输出|每个物理时隙一个标量|每个控制区间一个标量|

各特征的精确定义以`policies/networks.py`的frame/features函数为准。输入采用训练校准统计：`z=(x−mean)/max(std,0.1)`，再`z/(1+abs(z))`映射至(-1,1)。校准执行真实采样动作；统计固定，评价不更新；超范围和Tanh饱和比例进入日志。有限历史/均值池化是一种部分可观测策略近似，不保证充分Markov状态。

每层actor与critic编码器独立：items `19→64→64→64`，servers `13→64→64→64`，分别均值池化。actor拼接当前候选64、两组池化128、context11、prefix7：`210→64→64→1`；critic为`139→64→64→1`。每个MLP有两个Tanh隐藏层和线性输出。线性权重按[out,in]存放。每层actor 36,545、critic 32,001参数；两层合计137,092，FP32权重约0.523 MiB，不含优化器、激活及轨迹。

当前H=8，上层联合开关空间2^8=256；快层每请求最多`2+8×4=34`候选，整批联合动作是条件选择的组合，容量掩码进一步缩小可行集合。共享编码器不枚举全部组合，参数量不随联合动作数指数增长。现有真实剖面运行最后一批中，任务层items行数4/68/340（最小/中位/最大），每槽请求4/19/29，每请求候选1/1/26；服务器层items为16/90/114。每个状态包含servers[8,13]、context[11]，每条件动作另含prefix[7]。64个任务层transition首批items实际合并为[4549,19]，候选评分输入[4549,210]、输出[4549,1]，critic输入[64,139]、输出[64,1]。这些是旧运行样本尺寸，非固定形状或总体最大值。状态包含连续资源/历史特征，不能把张量元素数当有限状态总数。64宽保留为未调优起点；是否扩网须看充分训练后的价值误差、策略熵/KL及验证表现，不能由输入维数直接推导。

## Reward、GAE与PPO

物理奖励为`r=−(保守能耗 + lambda_time×公开权重加权时延 + rejection_cost×被拒绝请求的公开权重之和)`。当前lambda_time=1、rejection_cost=20是目标偏好，非测量或论文唯一值；抑制单列，不伪装成策略拒绝。

训练采用**running return std**：按物理槽累计`R←gamma×R+r`（episode起点重置R），Welford更新R的运行方差，`r_train=clip(r/sqrt(max(var(R),1e−8)),−10,10)`。不平移奖励，不要求奖励同时为正负。两层共享统计，校准和每个PPO阶段结束时更新；采样至本阶段更新完成期间冻结，评价不更新。价值输出通过非训练value_scale换算单位，统计与该比例均保存到检查点。裁剪可能改变偏好，必须报告触发数；±10是单槽保护界，慢层区间累计奖励可超出该界。

对持续d槽的transition：`delta=r_interval+gamma^d×V(next)−V(current)`，终止时next项为0；`A=delta+gamma^d×lambda×A_next`，终止/截断/缓冲末端均停止递推。lambda按本层决策步衰减，gamma按物理槽衰减。快层d=1，慢层d≤Q；区间内部奖励逐槽折扣。

PPO的critic目标是**GAE lambda-return `A+V_old`**，不是完整Monte Carlo回报。日志另列采样片段折扣和（截断末端加bootstrap），两者不可混称。actor用整层rollout的`(A−mean(A))/max(std(A),1e−8)`；单样本不去均值。gamma=1保持有限时域未折扣目标；lambda=.95的TD权重约20个决策步，因此快层约20槽、Q4慢层约80槽，并不等于只需要20条训练数据。

联合logprob按原请求顺序/动作前缀重算并求和；容量掩码参与条件概率。PPO裁剪比率区间[.8,1.2]；loss=actor+.5×(.5×MSE)−.01×entropy。空动作时隙仍更新critic，actor项为零；上下层分别on-policy采样，另一层更新后不能复用旧轨迹。actor/critic各自裁剪梯度范数至.5，Adam lr=3e−4/eps=1e−5；全buffer更新后联合KL>.03提前结束epoch。详细逐批中间量可独立复算。

## Episode、rollout与更新节奏

当前条件配置为128槽episode、Q=4、8服务器/40终端、每终端每槽到达概率.5。一个请求内部选择不是一个transition；一整个批次动作才对应一个快层transition。episode是环境重置单位，rollout是本次采样集合，minibatch是一次梯度步的数据，epoch是遍历该rollout一次。

|阶段|每次rollout|物理槽/完整episodes|minibatch=64，epochs≤5|
|---|---|---|---|
|快层预训练|1024个快层transition，共8次新采样更新|每次1024槽/8条|每epoch16批，最多80步|
|交替快层|1024个快层transition|1024槽/8条|每epoch16批，最多80步|
|交替慢层|256个区间transition|1024槽/8条|每epoch4批，最多20步|

预训练总量8192按fast_slots分块，最后一块允许不足1024；块后更新再重新采样，恢复跳过完成块。之后28次慢/快交替，HPGI合计65536训练槽，校准/验证另计。保留128槽episode与每rollout八条完整轨迹：GAE有效记忆约20快槽/80慢层物理槽，128槽已覆盖多次Q4控制及唤醒；当前没有证据需缩短episode，缩短还会改变有限时域目标。只将PPO epoch由2增为5，上层每轮梯度步8→20、下层32→80，保留KL=.03保护、64宽、学习率和奖励。训练按配置完成全部交替轮次，无连续无改善早停；验证曲线仅监测，不自动判定收敛。需要稳定后的更多曲线点可追加实际训练轮次。相邻样本相关，不能当独立样本；预训练一大块8192只更新一次的旧节奏已替换。

17槽episode的慢层持续长度为4/4/4/4/1。真实终止不bootstrap；采样截断使用正确下一观测和价值，GAE不跨重置。新collect从新episode开始，截断后未采样的后缀不补造。若horizon/Q/方法改变，物理交互预算必须重新核算：Q1、Always-On、仅慢层学习不能凭相同cycles宣称公平。

## 方法与配置

所有方法共用环境、指标和PPO核心；[README](README.md)列全部方法入口。Fixed-Window输出无联合容量掩码的离散动作和学习速率，PPO联合密度含原始高斯潜变量；执行前固定投影不修改行为概率，保留投影前违约。Greedy快层比较完整P3增量成本；慢层用上一完整公开区间从全开逐台试关，仅预测总成本下降才接受，不看未来。有界枚举只有完整搜索才报告最优。

`configs/environments/`管场景，`configs/algorithms/`管采样、网络及优化，顶层JSON组合二者。overlay/--set严格校验键；`ppo_layers.upper/lower`可覆盖宽度、学习率等，gamma共用。12条validation是可改的开发样本量，不是batch或论文固定值；min_service_rate=null，选中最低验证成本不代表服务要求满足。

检查点保存优化器、统计、随机状态、阶段、累计物理槽/epoch/完成episode及源码/输入签名。完成阶段last可续训；interrupted/failed_last若包含未完成阶段则只供诊断。槽上限在完整rollout前检查；时间上限在物理槽/优化器小批次边界检查，可能包含当前操作耗时的超出量。stopped与数值错误failed分开；恢复重复的未完成阶段交互单列。新增源码不强行加载旧签名权重。

## 定位问题与外部参考

效率检查边界：候选生成只在单个请求调用内复用与服务器无关的切分剖面，不做跨配置持久缓存；执行只检查选中候选，按服务器一次分组，空组跳过P3；贪心复用已知当前成本，仍比较完整P3增量成本。文件哈希以1MiB分块读取，避免把整份轨迹装入内存。`tests/test_static_efficiency.py`以调用次数、动作一致性和哈希检查这些性质，不以墙钟阈值作为测试。

评价和冻结层使用`record=False`动作路径，跳过critic及训练记录；采样在同一rollout内复用已算出的后继frame/value，重置时清空缓存。PPO的`RolloutTensors`只缓存原始输入、掩码及已选动作张量，跨epoch复用；网络激活仍逐次重算，缓存随本次更新释放。环境时序、容量前缀、GAE递推和P3二分保留必要顺序。

behavior采用`mist.behavior.shared.v1`共享frame/特征组格式，避免重复展开相邻观测与上层前缀；统一通过`traces.read_behavior`还原，兼容旧JSON列表。JSON采用缓冲流式写入临时文件后原子替换，不构造完整输出字符串；失败保留上一个完整文件。检查点history仅存每阶段标量及短摘要，不再携带输入诊断、逐槽duration和transition例子，完整证据各存于原诊断文件。恢复仍保存完整优化器、归一化器、随机状态与选择历史，原始结果CSV不变。

先看resolved_config和progress阶段；`traces/phases/<阶段>/`内rollout.json给出样本/物理槽/episode/小批次数及代表transition，behavior.json保存完整观测/前缀/掩码，inputs.json给出输入与网络诊断，update.json保存TD/GAE/loss/梯度，physical.json保存执行。`inspect_logs.py`兼容新旧目录并独立复算；`check_rollouts.py`只采样核验。

PPO更新通过`policies/batching.py`执行真正的小批张量前向/反向：变长集合展平编码并按样本分段均值池化，候选分数加掩码后批量计算条件概率，再按原动作前缀汇总联合概率。空任务仅更新critic，Fixed-Window连续速率密度保留。逐样本`replay`仅作独立测试参照。采样仍按环境时间和资源前缀顺序执行，但直接保存当次动作概率，不再重放整批动作；上层前缀共享只读items，只复制发生变化的服务器行。

每个epoch的loss/熵/clip等按实际小批样本数加权，尾批不与满批等权；KL另在epoch后对全部rollout分块批量重算，不计算无用critic。更新前概率复核也走同一批量路径。`metrics.jsonl`单列diagnostics_seconds、trace_write_seconds、probability_check_seconds及update_timings（前向/loss、反向/step、KL），不把阶段剩余时间冒称磁盘写入。轨迹JSON不缩进，完整数值和独立复算接口保持不变。

`runtime.torch_threads`可配置CPU算子线程数，默认1；小网络串行采样与批量更新对线程数的收益不同，需按整条链路实测，不能用线程数推断并行效率。`scripts/validation/benchmark_batching.py`默认使用合成变长输入，也可传--behavior读取已有轨迹，只测量不写结果或更新权重。

终端按完整交替轮次显示Iteration及三个步骤：服务器PPO、调度PPO、联合评价。采样、epoch和评价轨迹进度使用tqdm临时单行，结束清除；非交互输出省略进度条。每轮摘要记录联合评价原始平均episode回报、成本/时隙、按时率及两个组件各自的样本加权平均loss，不把两层loss合成整体目标。详细逐epoch诊断仍写stdout.log、progress.jsonl和TensorBoard；关闭writer后才计算文件哈希。CSV的iteration字段保留组件更新计数，新validation.csv显式增加cycle_index表示完整轮次；确定性与固定种子概率采样reward分列，两者仅用于监测与分析，不触发停止，不改写历史结果。原始episode回报和固定验证回报不受归一化尺度影响；训练奖励/尺度/裁剪次数另列。具体命令、停止开关和目录说明只在README维护。

- [CleanRL PPO](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo.py)：fe8d8a03c41a7ef5b523e2e354bd01c363e786bb，MIT；正交初始化、Adam、GAE、clip/KL/梯度经验参考，没有复制源码文件。
- [SB3](https://github.com/DLR-RM/stable-baselines3/tree/0cdcd200deef06b9ad0af72f6ddfb07c06c6df39)：MIT；终止/截断、种子与评价核对。[MaskablePPO](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/tree/a8af63682070df0bda9707400b2e971055600910)：MIT；掩码语义核对，不调用其learn。
- [MERA来源及许可](references/README.md)：只读参考，未移植其DDPG、合成资源或奖励。
- 奖励缩放参考本机SB3 2.3.2的[VecNormalize](https://github.com/DLR-RM/stable-baselines3/blob/v2.3.2/stable_baselines3/common/vec_env/vec_normalize.py)，文件SHA-256 `dfe7723393b73e7bf57a414dd77d0265e8d46412a7b9475f5679d62132ea997b`；本实现改为phase边界更新无伪计数Welford统计，并显式换算critic单位，没有复制其文件。
