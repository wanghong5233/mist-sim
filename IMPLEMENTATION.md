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

|张量（float32，采样时CPU）|快层|慢层|
|---|---|---|
|items `[候选数,19]`|当前公开请求的物理候选；request_sizes保留每个请求的分组|历史公开请求的同语义候选；开关动作另由独立actor_actions编码|
|servers `[H,13]`|服务器资源/功耗、激活/就绪、历史利用等|同类公开服务器特征|
|item_servers `[候选数,13]`|逐候选绑定目标服务器；本地/拒绝为零|历史候选与目标服务器对应|
|context `[7]`|剩余时间、请求数、历史发布/成本/完成数、历史存在标记、控制相位|同义统计，当前请求尚未发布|
|prefix `[7]`|处理进度、累计资源和接纳数|已选服务器开关的数量/比例|
|前缀关系|remaining `[H,3]`、已选任务 `[已选边缘数,19]`、所属服务器索引、待处理请求掩码|逐次更新已决定服务器的激活/首槽就绪状态|
|actor输出|每个请求K个条件候选分数，非法项掩码为零概率|每服务器2个条件开关分数|
|critic输出|每个物理时隙一个标量|每个控制区间一个标量|

各特征定义以`policies/networks.py`为准。候选保留计算需求、传输量、期限余量、内存及成本系数，去掉用服务器/切分编号代替物理属性的输入；上层开关使用独立动作编码器与归一化器。输入按校准数据计算`z=(x−mean)/scale`，std>1e−8时scale=std，否则取max(abs(mean),1)，再用`z/(1+abs(z))`映射至(-1,1)。不再以0.1强行抹平小量纲变化。剩余资源比例另用同一有界映射，保留Fixed-Window投影前负余量的方向/大小关系。统计校准后冻结，评价不更新。有限历史/均值池化仍是部分可观测近似，不承诺保留任意复杂状态的全部信息。

每层actor与critic编码器独立。候选19、服务器13各经两层64宽Tanh编码，候选与其服务器拼接为128维关系输入。先在每个请求内部池化，再结合选项数量及可服务标记编码为64维；只汇总待处理请求。已选任务按所属服务器汇总，与服务器编码、剩余三类资源和任务数拼接为132维，再编码到64维。actor评分输入为当前动作64＋请求摘要64＋服务器摘要64＋明确目标服务器64＋context7＋prefix7，即`270→64→64→1`；critic为`135→64→64→1`。每个MLP有两个Tanh隐藏层和线性输出。每层actor96,001、critic77,761参数；双层合计347,524（不含Fixed-Window速率头）。

当前H=8，上层联合开关空间2^8=256；快层每请求最多`2+8×4=34`候选，整批联合动作是条件选择的组合。actor批量评分为`[所有条件选择的候选总数,270]→[同样行数,1]`，按请求或服务器分组后取Categorical；critic为`[transition数,135]→[transition数,1]`。行数不是单次输入特征宽度。`tests/test_state_relations.py`用同候选多重集、不同请求分组且最优动作相反的反例验证区分能力；另验证目标服务器、先前任务指派及真实采样概率/梯度。64宽仍为未调优起点，不能由维数断言容易收敛。

## 方法身份、改进试验与历史兼容

MIST是系统框架。HPGI改进试验的历史代码名HARP是本项目自起缩写，以下仅作实现简称，不是标准算法名称或已证实的贡献。方法键`harp`与原HPGI的`hpgi`分开，保留实验身份和恢复签名。`policies/registry.py`是统一工厂，Trainer/CLI/核验及spawn worker通过它创建控制器。`base.py`复用网络构造、慢层控制及`upper/lower`权重布局；`hpgi.py:HPGIController`只执行原策略动作，`harp.py:HARPController`独立实现提案、映射、改进和执行元数据，不在原HPGI路径中挂新搜索。

HARP配置必须为robust信息、dynamic_split、容量掩码、P3、`fixed_proposal`及`assignment`，扫描预算为1或2；不允许通过零遍或splits范围悄悄成为另一算法。`for_method(...,"harp")`安装该组合，离开HARP时清除专用求解标志；旧默认配置不增加这些键，避免改变其配置签名。网络结构、初始化顺序、优化器、状态/奖励、校准、采样与存储接口共用，不增加学习参数。

原HPGI默认`split_solver=policy`。历史`hpgi+fixed_proposal`经`legacy_refined.py`兼容，保留原方法键、配置及零遍/splits/assignment行为，不是新的正式HARP运行。HARP及此兼容路径将策略输入、候选和前缀资源统一限制到配置中的逐模型固定点，仍学习启停、接纳和服务器指派；历史上层编码、校准、当前/下一状态critic、worker与评价采用相同提案语义。旧零遍模式复现同权重Fixed-Split；固定点来自独立离线选点，作为提案可行集限制。新HARP由`harp.json`或带共享价值设置的`harp_finite_horizon.json`进入，不隐式加载父权重。

历史`split_refinement_scope=splits`在1或2遍内枚举已接纳edge任务的同服务器切分，检查联合内存/最低算力，只重算受影响服务器的P3，接受严格降成本的可行变更；此范围不增加接纳、不换服务器。HARP使用`assignment`同时搜索本地/接纳/服务器/切分；单项改善不将已接纳任务改为拒绝。随后对仍被拒的任务尝试一次交换：同AP/服务器的一个edge任务转至其最低成本本地或拒绝选项，目标任务利用释放资源；仅接受两项合并后可行且严格降成本的交换，总接纳数不减少，但不保证每个原任务都继续接纳。每遍次序固定，默认仅1遍；不搜索上层开关、不穷举全局组合。每服务器P3按该槽任务指派缓存，单请求无容量约束最优成本之和作为合法下界剪枝，缓存只在内存且随槽释放。公开求解器实现见policies/assignment_solver.py；无隐藏请求或未来信息。单槽成本/接纳性质不能推导历史反馈改变后的完整闭环支配。

PPO样本保留原提案动作、前缀掩码和联合log概率；映射/优化后的`executed_actions`独立保存于内存样本，环境按实际动态候选执行，真实奖励给原提案。优化器未把确定性求解结果当作从Categorical采样的动作。条件于公开状态的确定性求解器属于转移核，因此仍可对原提案分布作score-function更新；不需要穿过P3求导。该槽不增成本只对同状态、同启停和同输入提案成立，不能推出历史反馈改变后整episode不增，更不能推出优于所有基线。求解耗时另作端到端评价。

`transfer.initialize_transfer`允许HPGI/HARP/Fixed-Split父网络显式迁移到新HPGI或HARP开发实验，但仍只接受当前实现和两个明确审计的历史源码签名；历史源码须以Git对象重建签名，不能伪造版本或放宽Trainer.load。检查父配置签名/校准/完整阶段，检查物理、奖励、网络及折扣合同，严格加载权重；复制统计/价值单位、重置Adam及新种子流、清空阶段历史/计数，继承预训练只记来源及成本。manifest和新检查点保留迁移凭据。允许的训练控制项/学习率/熵/探索温度改动必须作为新配置记录；每次更新重新采样，旧buffer不迁移。严格恢复不允许方法重标；新增源码导致签名变化，旧实验续训须使用其原匹配源码，不经兼容工厂绕过来源检查。

可选分层探索通过`ppo_layers.upper/lower`的`sampling_temperature`、`final_sampling_temperature`及`temperature_anneal_cycles`配置。第c个完整iteration（从0计）使用线性插值T(c)，到指定周期后保持终值；两个阶段共享本轮温度，阶段内冻结。合法离散动作分布为softmax(logits/T)，默认T=1完全保留旧行为；正温度不改变确定性argmax，不改高斯速率分布、奖励、环境或网络维度。旧logprob来自实际温度分布，每条内存样本保存T；串行和批量PPO都用该样本T重算新logprob，GPU缓存一次打包温度，worker接收本轮两层温度。检查点保存当前T，恢复按原配置和真实iteration继续，旧权重迁移从新配置起点开始。每阶段标量增加温度、逐服务器活动占比及AP带宽满载率，不导出详细轨迹。预设`hpgi_exploration.json`只作为待检验工程候选：上层2→1.5（第10轮达到），下层0.7→0.5（第5轮达到）、下层熵系数0.001；它提供分层Boltzmann探索/退火，不保证跨越次优组合或优于固定切分。

## Reward、GAE与PPO

物理奖励为`r=−(保守能耗 + lambda_time×公开权重加权时延 + rejection_cost×被拒绝请求的公开权重之和)`。当前lambda_time=1、rejection_cost=20是目标偏好，非测量或论文唯一值；抑制单列，不伪装成策略拒绝。

训练采用**running return std**：按物理槽累计`R←gamma×R+r`（episode起点重置R），Welford更新R的运行方差，`r_train=clip(r/sqrt(max(var(R),1e−8)),−10,10)`。不平移奖励，不要求奖励同时为正负。两层共享统计，校准和每个PPO阶段结束时更新；采样至本阶段更新完成期间冻结，评价不更新。价值输出通过非训练value_scale换算单位，统计与该比例均保存到检查点。裁剪可能改变偏好，必须报告触发数；±10是单槽保护界，慢层区间累计奖励可超出该界。

对持续d槽的transition：`delta=r_interval+gamma^d×V(next)−V(current)`，终止时next项为0；`A=delta+gamma^d×lambda×A_next`，终止/截断/缓冲末端均停止递推。lambda按本层决策步衰减，gamma按物理槽衰减。快层d=1，慢层d≤Q；区间内部奖励逐槽折扣。

`ppo.value_target=gae`保留critic的GAE lambda-return `A+V_old`对照；`sampled_return`改用当前采样片段的实际折扣奖励和，真实终止不bootstrap、采样截断及buffer末端用已保存的后继value。两种模式都保持actor原始GAE不变。日志分别列gae_value_target、critic_target及segment_return_with_bootstrap，不能混称。actor默认整层rollout标准化`(A−mean(A))/max(std(A),1e−8)`，phase模式按组件计算；单样本不去均值。gamma=1保持有限时域未折扣目标；lambda=.95的TD权重约20个决策步，不等于只需20条训练数据。

`ppo.value_model=plain`保留原MLP价值函数。`finite_horizon`令n为公开context的剩余物理槽数，D(n)=sum(gamma^i,i=0..n−1)，价值为`V=s×[b×D(n)+f(state)×D(n)/D(H)]`；s为原value_scale，b为过去真实物理奖励均值转换到内部单位，H为episode长度。只在采样前用过去统计更新b，采样和PPO期间冻结；两组件使用同一均值，各自学习状态残差f。初始残差输出为0，不改变actor初始化随机流。剩余一槽的价值包含最后一次奖励；n=0才为0。该参数化防止整个时域的累积回报偏移挤入终止状态，不添加私有信息或新reward。均值、scale及运行统计保存恢复，串行、批量、CPU/CUDA和worker快照共用一条value计算路径。折扣时域使用log/expm1稳定计算，gamma=0和1单独处理，避免接近1时float32消减使价值与残差梯度归零。

联合logprob按原请求顺序/动作前缀重算并求和；容量掩码参与条件概率。PPO裁剪比率区间[.8,1.2]；loss=actor+.5×(.5×MSE)−.01×entropy。空动作时隙仍更新critic，actor项为零。HPGI及其交替变体分别on-policy采样，另一层更新后不复用上一阶段轨迹；joint_ppo使用同一快照的事件buffer联合更新，见方法章节。actor/critic各自裁剪梯度范数至.5，Adam lr=3e−4/eps=1e−5；全buffer更新后联合KL>.03提前结束epoch。详细逐批中间量可独立复算。

## Episode、rollout与更新节奏

当前条件配置为128槽episode、Q=4、8服务器/40终端、每终端每槽到达概率.5。一个请求内部选择不是一个transition；一整个批次动作才对应一个快层transition。episode是环境重置单位，rollout是本次采样集合，minibatch是一次梯度步的数据，epoch是遍历该rollout一次。

|阶段|每次rollout|物理槽/完整episodes|minibatch=64，epochs≤5|
|---|---|---|---|
|快层预训练|1024个快层transition，共8次新采样更新|每次1024槽/8条|每epoch16批，最多80步|
|交替快层|1024个快层transition|1024槽/8条|每epoch16批，最多80步|
|交替慢层|256个区间transition|1024槽/8条|每epoch4批，最多20步|

预训练8192槽按fast_slots分块，之后最多3000次慢/快交替，上限6152192训练槽，校准另计。conditional禁用额外独立评价；一个完整iteration只有两个组件的采样及PPO更新。保持128槽episode、rollout及最多5epoch/KL=.03；不通过缩短有限时域改变目标。cycles.csv汇总真实采样的按槽平均奖励、两层loss与耗时，cycle_history及稳定窗口保存恢复。稳定判据分别检查整体、慢层、快层原始采样奖励，不增加环境交互；这不是更新后冻结策略的测试分数。

configs/training/fixed_200.json将cycles与早停资格min_cycles均设为200；不改变窗口/阈值、网络或采样量，正常运行准确产出200个完整iteration点。hpgi_finite_horizon.json选择HPGI及共享finite_horizon/sampled_return，CPU/CUDA设置另由cuda_update覆盖。该组合总计417792训练槽，校准另计；不同方法仍按CSV实际物理槽比较交互效率。

runtime.windows_performance默认false，显式启用时由Trainer构造器对当前Windows进程设置AboveNormal和HighQoS。worker也构造自己的Trainer，因此逐worker独立设置，不假定Python spawn继承主进程优先级。Win32句柄与结构宽度显式声明，执行后读回priority和power masks，失败抛出原OS错误；不改线程数量、CPU亲和性、管理员权限或全局电源计划。进程PID/策略记录在process_policy.json及worker计时中，用于核对后台运行试验；收益依据真实墙钟/CPU时间判断，不由设置成功推断已加速。

17槽episode的慢层持续长度为4/4/4/4/1。真实终止不bootstrap；采样截断使用正确下一观测和价值，GAE不跨重置。新collect从新episode开始，截断后未采样的后缀不补造。若horizon/Q/方法改变，物理交互预算必须重新核算：Q1、Always-On、仅慢层学习不能凭相同cycles宣称公平。

## 方法与配置

`joint_ppo`是可切换的训练机制对照：不改变环境、状态字段、匿名边界、容量掩码或P3。`training_joint.py`按`upper`服务器事件（duration=0、即时奖励0）与`lower`物理槽事件（duration=1、完整负成本）构造同一个EventRolloutBuffer；开关读取当前请求发布前的信息，调度读取开关后发布的当前公开批次，时间仅由执行物理槽推进。每个episode的首事件是开关，随后每Q槽插入一次开关。历史不足或空请求不会删掉物理奖励/critic样本。

JointPolicy包装原有两套SetPolicy作为phase条件分支，不增加可见信息或参数数量；两类actor评分仍为270→64→64→1，critic135→64→64→1。批量重放按phase分组，复用RolloutTensors与原动作前缀/掩码，再把结果放回原batch顺序；不是对整条轨迹概率相乘。一个联合Adam更新两分支，首次联合更新显式复制预训练调度Adam状态。两个actor共同裁剪、两个critic共同裁剪。ppo.advantage_normalization默认为global；phase按开关/调度分别标准化，单元素组保留原值。优势和所选critic目标在更新前计算一次，各epoch复用。phase改变组件间策略梯度尺度，不是额外奖励或收敛保证。joint_phase_normalized.json保留plain/gae对照；joint_finite_horizon.json选择finite_horizon/sampled_return及phase标准化，dry-run和终端启动行显示实际价值模型与目标。

事件TD=`r+gamma^d*V(next_phase,next_obs)−V(old_phase,obs)`；GAE递推系数=`(gamma*lambda)^d`，开关零时长事件不增加折扣或lambda衰减，价值bootstrap跨phase。终止归零、截断bootstrap且不跨重置，行为快照包含两网络版本，使用后清空。采样至更新结束冻结归一化，每个真实物理reward只更新统计一次。PPO沿用公共clip损失/GAE/优势标准化/梯度裁剪，任一分支或整体KL超过阈值终止该批epoch，不结束整个训练。common ppo配置控制联合更新，冲突的层覆盖报错。

默认每iteration 1024物理槽=1024调度+256开关事件，1280事件在每epoch分20个64大小minibatch；最大5epoch，实际受KL限制。系统reward、成本、能耗与服务仅按物理记录统计一次；训练归一化reward均值按物理槽，training_event_reward_mean另列。components.csv保存组件loss/KL/参数变化，不假造两份独立系统回报。hpgi保留两个新采样批次的原机制及每组件决策步lambda语义；两方案比较的是完整训练机制，不能称只改变样本复用的一项消融。

`comparison.py`提供等交互预算的两方法运行：先在HPGI公共环境校准/预训练，再显式校验配置与源码签名，复用同一初始权重/优化器/归一化/随机状态。公共预训练后关闭进程池，两方法均从冷进程池开始计时。comparison.csv同时记录物理槽、训练transition、所有epoch样本使用总次数、Adam步数、原子决定数与采样/PPO/概率复核/轨迹写入耗时；sample_presentations按实际完成epoch计数，不包括KL复核前向。

相同2048物理槽下，HPGI使用256个开关+1024个调度训练样本，最多5epoch共100个Adam步骤；joint_ppo使用512个开关+2048个调度训练样本，共200步。因此等交互预算不等于等优化计算量，减少重采样不保证总耗时更短；需分别观察按物理槽和墙钟时间的学习曲线。各方法独立保存记录，不按iteration编号直接等同。原有PPO工程参考仍是下方固定版本CleanRL/SB3；阶段化事件实现是本项目适配，不宣称已证明最稳或优于HPGI。

所有方法共用环境、指标和PPO核心；[README](README.md)列全部方法入口。Single-Timescale通过`single_timescale`强制Q=1，每个物理槽重新选择服务器状态和任务调度；仍保留两套策略及交替PPO，按rollout更新参数，不是每槽做梯度更新，也不等同于Joint PPO。该对照改变控制周期，不自动关闭长期反馈；不能仅凭此对照归因长期回报的收益。

Upper-PPO+Greedy保留Q槽开关控制及上层PPO，下层直接调用`greedy_fast`，不做调度网络预训练、采样或参数更新。贪心从全部拒绝开始，按公开请求顺序处理；在算力、显存、AP带宽联合掩码允许的本地和边缘候选中，逐个试选并重新执行整个已选批次的P3分配，比较固定开关下的总调度成本（含拒绝代价）。只有成本下降超过1e−12才替换当前动作，平局保留原选择。静态及唤醒成本由共同环境另计，对固定开关下候选比较是常数；上层区间回报仍含这些成本。不读取当前未发布请求或真实私有属性；固定请求顺序是该启发式的定义，不声称全局最优。

Fixed-Window沿用HPGI的固定Q窗口、保守候选和两层交替PPO；当前正式配置Q=4，名称不表示另一种buffer或滑动窗口。区别是关闭请求间的联合容量掩码，并用学习速率替代P3；单请求合法性、服务器就绪和模型兼容筛选仍保留。下层对所选边缘动作采样高斯潜变量，经sigmoid映射为服务器容量比例。统一投影在环境实际执行前处理联合超配：按请求顺序将无法保留的指派改为拒绝，再将各服务器速率投影到最低速率及总容量约束；不对投影求环境梯度。

Fixed-Window的buffer保存原始离散动作及高斯潜变量，PPO使用二者的联合logprob，不能把投影后的拒绝/速率回写为行为动作。速率头及log_std包含在优化器、采样worker快照和检查点中；奖励来自实际执行结果，投影前违约另计。该对照同时改变容量接口与资源分配器，不能把结果差异仅归因于窗口长度、掩码或P3中的某一项。

Greedy慢层用上一完整公开区间从全开逐台试关，仅预测总成本下降才接受，不看未来。有界枚举只有完整搜索才报告最优，离线多槽枚举也必须在Q区间内保持开关不变。preflight的物理最坏边界、可执行调度及小网格穷举分别检查单请求、联合可行性和受限动作集差距，不将其冒充HPGI训练结果。

Fixed-Split由离线选点和在线策略学习两部分组成。`select_fixed_split.py`遍历各模型合法切分集合的笛卡尔积；当前三模型各4点，共64组，而不是把有共享容量竞争的三个模型独立选点。固定开发控制器为公共因果Greedy-Consolidation＋容量掩码＋P3，仅限制切分不随请求改变；每组在完全相同的混合模型开发输入上执行，选择`z*=argmin_z sum(C_z)/sum(physical_slots)`，C沿用训练的能耗、加权时延及拒绝成本。按实际槽池化，不平均不同分母的均值；无新增服务门槛，所有得分/按时率/能耗保留，完全平局按模型名排序后的切分元组字典序。该选择是在指定开发控制器下的最优候选，不保证任意训练策略下的全局最优静态切分。

`train_fixed_split.py`作为完整流程入口，在选择不存在时启动选择，已有选择运行则等待其完成；选择已完成时直接复用，不重复搜索。校验开发场景、候选集合及输入/选择哈希，导出冻结配置，再经公共train从头独立训练Fixed-Split两层PPO。固定配置以模型剖面、物理场景和目标权重为适用范围，同场景跨训练种子/轮数复用；算力、内存、带宽、负载分布或目标改变的敏感性实验须在各自开发输入上另行选点，场景不匹配会拒用旧选择，不能将模型相同理解为旧固定点始终最优。开发输入不作为最终测试；选点成本与训练交互预算分开登记。训练及测试期间每个模型始终用冻结切分，开关和请求指派仍学习，P3仍执行；旧3/3/7不是有依据的选优结果，不能将其权重直接沿用为新流程结果。可等指定训练退出后打开新可见终端，队列保留配置、状态及退出回执，不产生第三路并行PPO。

另设朴素Fixed-Split (Static-Latency)，不调用上述业务轨迹选优流程。逐模型在合法点z中最小化静态单请求时延`C_local(z)/f_local + mean_j[B(z)/R_ref,j + C_edge(z)/F_j] + overhead`，j遍历支持该模型的服务器且等权，平局取较小点号。顺序执行的两段计算相加，不能用均分工作量或均衡两段耗时替代该时延目标。R_ref复用环境radio_rate，参考位置/衰落为配置范围中点，F_j为空载服务器容量；同请求各切点相同的匿名/控制/回传开销计入但不影响排序。当前参考位置50、衰落1、终端20、服务器100/200，三个模型独立推导3/2/4。它使用标称条件、忽略共享负载/启停/能耗目标，不是开发轨迹期望最优或原全成本选优的替代；在线仍用公共匿名信息/容量掩码/P3和相同完整成本奖励。12候选×8服务器仅作标量计算，无环境采样或PPO选点；推导及配置哈希保存在各次launcher，覆盖配置见configs/algorithms/fixed_split_static_latency.json。剖面、输入尺寸、算力或链路配置变化须重新推导。

`configs/environments/`管场景，`configs/algorithms/`管采样、网络及优化，顶层JSON组合二者。overlay和--set均可向`ppo_layers.upper/lower`添加合法覆盖（宽度、学习率、minibatch等），未知字段/非法取值报错，gamma共用。12条validation是可改的开发样本量，不是batch或论文固定值；min_service_rate=null，选中最低验证成本不代表服务要求满足。

基线校准只调用实际使用的策略组件：Always-On始终全开，Upper-PPO-Greedy执行贪心调度，未使用的网络不推理或拟合归一化。Greedy慢层在同一次启停搜索内复用已验证剖面；不同场景不共享实例。单层进度和续训预算按实际方法/已完成阶段计算，禁用独立评价不影响续训轮数。check的Fixed-Window手写控制器兼容统一评价接口；它仅检查环境核算，evaluate仍要求该学习方法自己的检查点。启发式误用train在创建运行或权重之前报错。

Greedy批量入口`scripts/evaluation/evaluate_greedy.py`只编排公共greedy/evaluate，不构造网络或修改算法。每个worker独立执行一个完整输入，按共同环境核算；20条输入不是20个训练种子。共享输入存data/processed/evaluation，首次冻结场景/种子/哈希后不可覆盖；输入需与已有训练种子台账隔离。evaluation.csv保存逐轨迹及pooled行，aggregate.csv保存轨迹均值/样本标准差（成本、能量除以实际槽数）；pooled按时率使用总完成/总到达，P95由全部完成延迟重算。物理检查只核算记录，不额外执行贪心。推理实测耗时记录但不自动加入预留控制时延，部署超时实验须另开配置。

`always_on_finite_horizon.json`选择有限时域critic及sampled_return，单调度rollout为2048槽。与fixed_200组合产生4次预训练更新及200轮，每轮16个128槽episode；8192+200×2048=417792训练槽、256校准槽另计。保持原环境和目标，采样CPU4worker、更新CUDA/minibatch128复用公共overlay；相同物理预算不等于相同optimizer步数或已充分调优。

检查点保存优化器、统计、随机状态、阶段、累计物理槽/epoch/完成episode及源码/输入签名。conditional每10个完整iteration原子覆盖唯一last；完成阶段last可续训，interrupted/failed_last若包含未完成阶段默认只供诊断。槽上限在完整rollout前检查；时间上限在物理槽/优化器小批次边界检查，可能包含当前操作耗时的超出量。stopped与数值错误failed分开；恢复重复的未完成阶段交互单列。新增源码不强行加载旧签名权重。独立故障迁移必须核对原快照哈希、原/新源码逐文件白名单、失败前无update事件、Adam步数等于完整history、状态有限性和完整CSV；仅修改已审计保护元数据，保留学习状态/RNG和已消耗采样计数，不能将此入口泛化为忽略签名。

更新前行为概率检查由batching.check_behavior_probability共用。float32批量矩阵计算可能与采样时逐条计算有微差；超过绝对1e−5的行逐条复算，逐条与原概率仍须相差≤1e−5，批量与逐条差额另须≤8×float32_eps×max(1, |原logprob|, 条件决策数)。仅异常行增加逐条计算，不重写行为概率或更新路径；metrics记录batch最大差、回退行数及逐条误差，非有限值/真实策略失配仍失败。

## 定位问题与外部参考

collection.py通过spawn常驻runtime.collection_workers个进程采集独立episode，复用_collect_serial物理循环；conditional基础配置为8，cuda_update覆盖为4，HPGI与joint_ppo共享。每个任务加载同一组网络/归一化/版本快照，固定请求和动作种子；worker不更新参数或统计、不写正式日志。主进程按episode顺序合并，终止/截断边界保持，完成采样后才更新PPO和运行统计；下一阶段发送新快照。池不进入检查点，run的finally关闭，直接collect调用者需close。交替设计将当前被更新组件视为策略、冻结组件视为环境的一部分；冻结组件改变后必须重新采样。此依据是on-policy条件，不是初稿的权威性，也不证明该设计优于联合更新。

效率检查边界：候选生成只在单个请求调用内复用与服务器无关的切分剖面，不做跨配置持久缓存；执行只检查选中候选，按服务器一次分组，空组跳过P3；贪心复用已知当前成本，仍比较完整P3增量成本。文件哈希以1MiB分块读取，避免把整份轨迹装入内存。`tests/test_static_efficiency.py`以调用次数、动作一致性和哈希检查这些性质，不以墙钟阈值作为测试。

评价和冻结层使用`record=False`动作路径，跳过critic及训练记录；采样在同一rollout内复用已算出的后继frame/value，重置时清空缓存。PPO的`RolloutTensors`只缓存原始输入、掩码及已选动作张量，跨epoch复用；网络激活仍逐次重算，缓存随本次更新释放。环境时序、容量前缀、GAE递推和P3二分保留必要顺序。

behavior采用`mist.behavior.shared.v1`共享frame/特征组格式，避免重复展开相邻观测与上层前缀；统一通过`traces.read_behavior`还原，兼容旧JSON列表。JSON采用缓冲流式写入临时文件后原子替换。检查点history仅存每阶段标量及短摘要，恢复仍保存优化器、归一化器、随机状态与评价历史。conditional按完整迭代低频覆盖一个last.pt；诊断按trace_retention有界保留，输入种子与所有原始结果CSV不删。细节与恢复边界见README。

先看resolved_config和progress阶段；`traces/phases/<阶段>/`内rollout.json给出样本/物理槽/episode/小批次数及代表transition，behavior.json保存完整观测/前缀/掩码，inputs.json给出输入与网络诊断，update.json保存TD/GAE/loss/梯度，physical.json保存执行。`inspect_logs.py`兼容新旧目录并独立复算；`check_rollouts.py`只采样核验。

PPO更新通过`policies/batching.py`执行小批张量前向/反向：按候选→请求→状态分段池化、按指派绑定服务器，随后批量计算掩码概率并沿原前缀汇总。空任务只更新critic，Fixed-Window连续速率密度保留。逐样本replay作独立测试参照。采样保留物理时序与资源前缀依赖，单次no_grad动作内复用不变的请求编码，跨参数更新不缓存激活。上层前缀共享只读请求/候选，改变服务器行；校准统计按共享输入去重，避免多前缀重复加权。

每个epoch的loss/熵/clip等按实际小批样本数加权，尾批不与满批等权；gradient_after_clip_max始终取本次更新全部小批的最大联合梯度范数，不依赖详细审计开关。actor/critic分别裁剪，联合范数上界为sqrt(2)×max_grad_norm。KL另在epoch后对全部rollout分块批量重算，不计算无用critic。更新前概率复核也走同一批量路径。`metrics.jsonl`单列diagnostics_seconds、trace_write_seconds、probability_check_seconds及update_timings（前向/loss、反向/step、KL），不把阶段剩余时间冒称磁盘写入。rollout.learning_signal分组件记录更新前优势均值/标准差/正值比例、与时隙位置的相关性和价值解释方差；quality另记终止处value/reward/TD、bootstrap相对reward的波动，以及跨episode重复时间位置的组均值方差占比（描述性统计，不自动判错）。训练后的critic拟合不能证明策略收益改善。inspect_learning_quality.py从保存的公开请求/动作重新求P3成本，量化可直接替换为本地执行的逐槽节省下界；完整未裁剪gamma=1联合轨迹可离线重算留一episode的时间基线GAE，不将其用于训练或更改奖励。正式conditional训练关闭整批调试JSON导出（diagnostics.phase_interval=0）；正整数仅用于显式排障。PPO buffer始终在内存中正常采样、更新、消费，不因关闭文件导出而裁减。调试导出包括输入轨迹、行为状态/动作/掩码、物理明细和逐批更新中间量；这些不是论文结果CSV。关闭时保留轻量输入种子账本、每轮原始结果、标量诊断及低频恢复检查点；输入可由配置/版本/种子重建。

采样请求编码命中缓存时不再重建整组候选张量；梯度更新仍重算激活。环境逐槽复用已验证Profiles，区间剖面界使用对象内有界缓存并返回独立字典。主进程为溯源生成的轨迹传给worker，避免重复生成；无store的诊断仍由worker生成。P3内层为单调区间保护的Newton迭代，外层维持对偶二分，只在可行侧达到容量容差时提前结束。collection_timing区分准备、进程池等待、每条轨迹CPU/墙钟及返回等待；allocation_seconds只累加Environment.execute的最终分配，不包含贪心试选候选时调用的P3，不能据此低估贪心计算。return_seconds含序列化、传输、反序列化和主进程调度，不称作纯通信时间；trace_write_seconds含详细轨迹打包、JSON编码和写入，不是纯磁盘耗时。相同物理槽数下，Q1增加开关决策及训练样本，贪心试选仅重算目标服务器P3，复用其他请求成本，保留原请求顺序求和、容量掩码和1e−12严格改进规则；本地试选不调用P3。贪心仍增加候选资源重分配次数，均不能假定与HPGI每轮计算量相同。进程数通过同快照、同轨迹的重复暖池测量选择，不自动修改正在运行的配置。

device=cpu/cuda只选择PPO更新设备。环境、P3、匿名发布、采样worker、GAE及优势统计均在CPU；worker强制device=cpu，不创建CUDA更新器。devices.update_device在每次完整update入口移动策略/Adam矩，在finally回CPU，参数对象身份不变；Adam step计数保持CPU。保存和恢复均以CPU权重/优化器为边界，同源码允许只覆盖device，仍严格检查其余配置及输入。CUDA缺失报错，不静默退回CPU；OOM/异常也恢复CPU后由原失败路径保存诊断快照。

Windows同会话的CUDA更新通过内核命名互斥量轮流执行；CPU采样仍可并行，等待时间单列gpu_slot_wait_seconds。锁覆盖模型搬入、PPO及回CPU后的缓存清理；进程意外退出后内核释放所有权，后续可接管。每次完整更新只在末尾empty_cache一次，释放闲置allocator缓存，不逐minibatch清理；idle_reserved_mib记录清理后预留量。此措施降低多训练资源峰值，不能据此宣称修复Windows蓝屏。

RolloutTensors按float32/int64/bool打包整批输入，各一块连续存储；所有视图共享底层张量。PPO旧概率/优势/目标/有效标记另外合成一块float32上传，完整epoch和KL复核复用这些原始张量，梯度所需激活逐步重算。目标服务器、已指派任务和剩余资源也复用GPU缓存；小批分组/索引元数据仍由CPU构建并传入，不把rollout_tensor_transfers=4误称全部CUDA传输仅4次。不会逐时隙、逐epoch搬运大块状态，也不在正常更新中逐批empty_cache。device_metrics报告模型搬入/回传、完整更新耗时与峰值allocated/reserved显存；input_pack_transfer_seconds包含CPU整理和上传，计时不是纯PCIe带宽。

`runtime.torch_threads`可配置CPU算子线程数，默认1；小网络串行采样与批量更新对线程数的收益不同，需按整条链路实测。`scripts/validation/benchmark_batching.py`测CPU串行/批量路径；benchmark_update_device.py的--full-rollout在同一真实buffer及权重/Adam快照下比较CPU64和CUDA64/128/256，含搬运/KL/回传，为等5epoch计时仅在探针中放宽KL阈值。--snapshot-only显式导入旧完成快照仅作独立性能诊断，不恢复运行或写结果。CPU64/CUDA64为设备对照；128/256同时减少梯度步数，不能将全部收益归于硬件或宣称更易收敛。

conditional终端按Iteration及两个步骤显示服务器PPO、调度PPO；进度条单行更新，末尾汇总采样平均reward/slot、两层loss、采样/PPO耗时。TensorBoard的iteration/train与cycles.csv采用完整交替编号。HPGI和joint_ppo均仅在diagnostics.phase_interval为正且命中时执行输入归一化/激活饱和度的额外前向和详细行为/更新审计；HPGI按各组件更新次数独立计数，避免偶数间隔永远只覆盖交替中的一侧。优势、KL、梯度、概率复算与CSV仍每轮保留。显式启用validation时才增加第三步和对应评价结果，打印同样使用单槽均值；episode累计回报单列，不与归一化训练reward混用。

## 结果绘图

plotting.py只读取完成的cycles.csv；scripts/plotting/plot_convergence.py是统一入口，configs/plotting/convergence.json管理样式/统计。直接复用SciencePlots 2.2.2的science/ieee/no-latex，中文宋体、西文Times，PNG及嵌入字体PDF/轮廓SVG。iteration用于同方法训练趋势，total_slots包含预训练开销、用于不同机制交互效率比较；不补造iteration=0或预训练期间不存在的cycle点。

原始指标保留不变；展示时每种子先做同窗口后向均值，初始窗口只用已发生的点，再按精确共同横坐标求种子均值。单种子无统计带；多种子默认±1样本标准差，可切换整条种子bootstrap的逐点95%区间。不把时间窗口当独立种子，不外推短曲线或复制终值。图片、source_data.csv快照、aggregate.csv、plot_config和provenance写入同一结果目录的figures/convergence；快照及原CSV/代码/样式哈希使运行中绘图也可精确重现。重画已保存配置时检查快照哈希，不重新读取不断追加的原CSV。采样均值不称作固定验证/测试成绩，详见docs/PLOTTING.md。

## 实现来源

- [CleanRL PPO](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo.py)：fe8d8a03c41a7ef5b523e2e354bd01c363e786bb，MIT；正交初始化、Adam、GAE、clip/KL/梯度经验参考，没有复制源码文件。
- [SB3](https://github.com/DLR-RM/stable-baselines3/tree/0cdcd200deef06b9ad0af72f6ddfb07c06c6df39)：MIT；终止/截断、种子与评价核对。[MaskablePPO](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/tree/a8af63682070df0bda9707400b2e971055600910)：MIT；掩码语义核对，不调用其learn。
- [MERA来源及许可](references/README.md)：只读参考，未移植其DDPG、合成资源或奖励。
- 奖励缩放参考本机SB3 2.3.2的[VecNormalize](https://github.com/DLR-RM/stable-baselines3/blob/v2.3.2/stable_baselines3/common/vec_env/vec_normalize.py)，文件SHA-256 `dfe7723393b73e7bf57a414dd77d0265e8d46412a7b9475f5679d62132ea997b`；本实现改为phase边界更新无伪计数Welford统计，并显式换算critic单位，没有复制其文件。
