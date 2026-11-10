# 强化学习排障与经验复用

用于HPGI、Joint PPO及学习基线出现奖励停滞、异常更新或采样过慢时，复用已经定位的问题。当前实现以[IMPLEMENTATION](../IMPLEMENTATION.md)和实际配置为准；本机运行编号、原始数值和阶段结论留在SETUP、logs/runs及results，本指南不复制实验结果。

## 已遇到的问题与修正

|问题|这次得到的经验与已落地修正|可追溯依据|
|---|---|---|
|维度能对齐，状态却丢失决策关系|候选混合平均会丢请求分组，服务器平均会弱化目标关联；不同含义的列也不能混用。保留请求分组、目标服务器、已选任务和剩余资源，用“相同候选集合、不同分组、最优动作不同”的反例检查可区分性。张量行数是候选/样本数，不是特征宽度；不能只凭输入宽度判断是否容易学。|`debcea4`；[状态反例](../tests/test_state_relations.py)|
|参数有变化，但策略改善弱|非零梯度只证明能更新。还要查TD、优势及critic是否反映真实动作收益；固定公开状态上的少量学习可区分局部排序故障与长时域信用分配困难。局部探针成功不能替代完整轨迹训练。|`2a2b4de`、`4e49389`；[学习质量检查](../tests/test_learning_quality.py)|
|整体价值拟合指标高，末尾估值仍严重偏离|整体explained variance会被长时域回报变化主导，掩盖终止附近误差。增加有限时域价值参数化，以过去奖励均值和剩余时域建立基线、网络学习状态残差；critic可选实际采样回报目标，截断保留bootstrap。最后一槽仍有奖励，不能强制V=0。|`0772134`；[有限时域检查](../tests/test_finite_horizon_value.py)|
|折扣接近1时，价值及对应梯度意外归零|直接用float32计算`1−gamma^n`会因gamma舍入为1丢失整个剩余时域；改为log/expm1稳定计算，gamma=0/1单独处理，用双精度逐项求和核对值与梯度。当前gamma=1运行不受这个边界缺陷影响。|`9239b76`；[折扣边界检查](../tests/test_finite_horizon_value.py)|
|联合buffer中两个组件的优势尺度失衡|单组件预训练后，两个critic误差可能不同；Joint PPO增加phase优势标准化。它不能代替修好critic。HPGI本来就分buffer标准化，单层buffer的global与phase等价。|`2a2b4de`、`43b90c4`；[采样及优势检查](../tests/test_sampling_efficiency.py)|
|PPO与采样耗时过大|神经网络用批量张量前向/反向，缓存本批原始输入、复用公开剖面和P3计算；评价/冻结策略不计算无用critic和训练记录。保留物理时间、容量前缀等必要顺序。分别计时采样、更新、概率复核、诊断与写盘，再优化大项。|`368acdc`、`a1410d7`、`2a2b4de`；[批量检查](../tests/test_batching.py)|
|多进程/GPU使用方式与负载不匹配|独立episode可并行，环境和逐步采样在CPU；PPO整批输入上传后供多个epoch复用，更新结束回CPU。进程数和小批大小按本机计时选择；更多进程不保证更快，单进程内一线程也不等于所有计算串行。|`af2be26`、`2a2b4de`；[设备检查](../tests/test_update_devices.py)|
|换baseline后仍走了错误组件|校准必须调用该方法真实控制器：Always-On保持全开，Upper-PPO-Greedy用贪心调度；未使用网络不推理或拟合。方法注册、参数覆盖、评价、CSV和续训形成同一闭环。|`b7288b7`；[基线接口检查](../tests/test_baseline_interfaces.py)|
|训练正常更新，却在整理诊断文件时报错|轨迹清理曾漏掉`_joint`阶段名，属于存储故障，不能解释成梯度爆炸或算法发散。检查Traceback位置及已完成阶段，区分已更新、未完成阶段和可续训快照。|`4418ae0`；[联合流程检查](../tests/test_joint_ppo.py)|
|更新前概率检查偶发超阈值|float32批量与逐条GEMM/求和顺序可能不同。异常行必须按采样路径重算并符合原绝对阈值，批量差另受精度/决策数约束，不能直接放宽阈值掩盖真实策略变化。每10轮原子覆盖last保护恢复；故障快照仅在证明未更新且源码/Adam/CSV逐项匹配后专门迁移。|[概率反例](../tests/test_batching.py)、[周期恢复](../tests/test_training_only.py)；当前规格见IMPLEMENTATION|

这些修改按顺序实施，未逐项做完整因果消融。可以复用具体修复和检查方法，不能把全部收益归给一个改动；早期奖励改善是学习的好信号，也不是所有强化学习任务必须遵循的固定曲线形状。

## 再次学不动时，按这个顺序查

1. **先确认正在看哪种数据。** 固定运行、源码/输入版本和已完成iteration。训练图用cycles.csv的按物理时隙平均原始reward；一个iteration是一个点。预训练单列，PPO epoch重复使用同一批样本，不产生一批新的环境奖励；loss按实际样本数加权。同时看成本、服务和资源占用，不能只看负号或累计总量。
2. **确认有可行且成本不同的动作。** 用公开候选、掩码、P3及完整记账构造并重放见证；有必要才用小规模穷举。检查本地执行不消耗边缘资源，期限/单位/唤醒时序一致。拒绝可能合法，但存在更便宜合法动作时要量化机会成本。不要为制造可学习性暗改目标权重或匿名边界。
3. **确认state保留决策需要的信息。** 检查请求分组、目标资源、期限/传输/计算/显存需求和历史反馈的对应关系；对同样公开记录保持匿名观测一致。查看归一化后每列的分布、饱和比例及常量列，不能只查全张量最大值。新增字段必须有可区分反例或决策依据。
4. **沿一条transition手算。** 从公开state、原始采样动作/掩码、环境cost/reward，走到next state、终止/截断、旧value/logprob、TD、GAE和critic target。奖励组成/单位与归一化统计分开检查：running return std缩放奖励，优势再减均值除标准差；采样至更新结束统计冻结。物理奖励全负不妨碍学习，不为凑正负号平移奖励。
5. **先查价值误差，再解释优势。** 查看terminal value/target/MAE、分时间位置的估值和优势，以及bootstrap差与reward的波动。time相关不自动等于噪声，但若终止附近value明显不符合剩余回报，应先修估计器；高EV、低value loss和优势标准化都不能免去这一步。`gae_value_target`和实际`critic_target`须区分。
6. **再查策略更新。** 未更新时新旧概率比应接近1；非法动作概率/梯度为0；旧logprob、GAE、target跨epoch固定。检查两个actor/critic各自的参数变化、收益/熵梯度、裁剪前后范数、KL及clip比例。Fixed-Window须用投影前采样动作和速率潜变量计算概率，不能拿投影后的执行值重算旧动作密度。
7. **有界判断后再扩大预算。** 局部状态学习、短完整轨迹更新、同输入冻结策略比较分别回答不同问题；只做能区分当前假设的最小检查，通过后不反复重跑。保持原始CSV，不把十几轮波动直接判成永不收敛；不增加无改善早停。奖励稳定停止与PPO的KL停止属于不同层次，前者停止训练，后者只结束本批剩余epoch。

## 向HPGI和其他baseline迁移

公共Controller、SetPolicy、PPO、归一化、设备搬运和采样池已共享；共享代码修复不代表各方法的正式训练效果已验证。**有限时域估计器是显式选项，HPGI及其他基线默认仍为plain/gae。** 不要仅因为代码已合入就认为配置已启用。

HPGI的finite_horizon/sampled_return已通过合成输入的CPU串行与双worker闭环：区间内折扣、4+1槽尾区间、采样截断bootstrap、冻结组件、行为版本刷新、独立GAE/loss/物理复算及重载续训。关闭长期反馈时，Environment在两层状态入口都已过滤历史；这次检查未发现正常训练路径存在反馈泄漏。两层gamma覆盖也由配置校验禁止，不能把不存在的覆盖冲突当作缺陷。

|方法|可直接复用的部分|仍须按方法核对的区别|
|---|---|---|
|hpgi；fixed_split、single_timescale、nominal、full_information|状态关系、批量PPO、归一化、价值估计选项、采样/设备优化|交替更新中冻结另一组件，更新后重新采样；区间奖励逐槽折扣、duration用实际槽数。名义/理想信息与切分限制仍各自保留；Q=1和尾区间预算重算。|
|always_on|调度PPO及公共设施|只有调度网络训练，服务器始终全开，无开关预训练/更新。|
|upper_ppo_greedy|开关PPO及公共设施|只有开关网络训练，调度为贪心；不运行调度预训练，也不解释不存在的调度loss。|
|fixed_window|公共PPO、价值估计和设备路径|还含速率密度、原始动作与执行投影，单独检查概率和容量回算。|
|joint_ppo|本次修正版完整训练路径|同一快照事件链，开关duration=0/reward=0，调度duration=1/完整物理reward；跨phase bootstrap，物理成本只计一次。phase优势标准化适用于这个混合buffer。|
|greedy_consolidation|环境、剖面、P3、核算和输入复用|没有PPO/critic，直接评价，不套用奖励归一化或价值修复。|

HPGI启用本次价值估计经验的预览命令（从工程根运行）：

```powershell
& D:/Anaconda/envs/pytorch/python.exe -m mist_sim train --config configs/conditional.json --overlay configs/algorithms/cuda_update.json --overlay configs/algorithms/hpgi_finite_horizon.json --seed 4200001 --dry-run
```

核对后移除`--dry-run`才会训练。其他学习基线替换`--method`，先检查是否有ppo_layers覆盖。**不要把joint_finite_horizon.json直接当HPGI预设：该文件同时把method设为joint_ppo。** 如果显式再用`--method hpgi`覆盖，仍要检查最终配置，避免以为在比较两个方法，实际两次跑了同一个方法。

HPGI启用新估计器后的真实剖面短检查，可用`check_profile_chain.py --overlay configs/algorithms/cuda_update.json --set ppo.value_model=finite_horizon --set ppo.value_target=sampled_return`；这是后续使用入口，本指南整理时未新开训练或重跑该检查。HPGI每iteration有两次独立采样，Joint只有一次；不能直接用相同iteration数比较交互效率。当前预算和网络尺寸读取配置/IMPLEMENTATION，不把进程数、小批大小或预训练次数当作通用经验常数。

## 对应证据从哪里读

|要回答的问题|优先入口|
|---|---|
|奖励真的在改善吗，分母是否正确|results/同名运行/cycles.csv；metrics.jsonl中的raw_reward_per_slot、physical_cost_components、physical_slots，以及服务指标|
|参数是否更新，谁在更新|metrics.jsonl的parameter_delta_l2及components；progress.jsonl中的ppo_epoch、KL和阶段版本|
|critic/优势是否失真|rollout.learning_signal各组件的quality；critic_prediction的终止误差；TensorBoard的signal/与critic/|
|哪一步动作或loss算错|保留的traces/phases/阶段名/下behavior、rollout、physical、update四种JSON；inspect_logs.py独立复算|
|为何拒绝了更便宜动作，尾部是否有时间偏差|inspect_learning_quality.py --run 运行目录 --stage 已保留详细阶段；只读、无更新。它的留一episode时钟反事实仅适用完整、未裁剪、gamma=1的Joint轨迹；HPGI不套用该反事实|
|局部策略到底能不能学|diagnose_learning.py默认探针使用Joint事件链，仅用于兼容Joint快照；不直接拿HPGI或单层基线快照套这个脚本。先核对结构兼容，局部改善不外推全时域|
|慢在环境还是更新/整理数据|collection_timing、collection_seconds、update_timings、device_metrics、probability_check_seconds、diagnostics_seconds和trace_write_seconds；在不竞争正式训练资源时才做独立计时|

正式conditional训练默认diagnostics.phase_interval=0，只保留结果CSV、轻量metrics和输入种子；buffer正常更新，不导出整批训练数据。排障时在独立有界运行显式设置正整数间隔，再读取保留的详细阶段；不要为了潜在排障给每轮正式训练增加大JSON。查版本用`git show 提交号 -- 文件路径`，不要通过跳过签名校验把旧快照伪装成可续训。运行失败时保留Traceback、已完成更新和来源，临时探针在系统临时目录退出清理。

## 维护方式

新问题并入对应症状，替换已过期的处理方式；新增已证实的问题才补表格行，不按日期追加聊天沿革。实现变化只维护IMPLEMENTATION和配置，具体运行证据只维护SETUP/原始记录，本指南保留“怎么看、怎么区分、适用于谁”。
