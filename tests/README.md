# 测试范围

test_frozen_evaluation.py核对冻结策略的串行/并行一致、逐轨迹动作随机源重置、原权重与归一化/优化器步数不变、物理重放、源码/输入/权重篡改拒绝、仅到达概率的场景覆盖，以及按请求计数池化服务率和原始完成延迟P95。配对bootstrap必须使用相同输入哈希；全部测试资产在临时目录，不启动正式训练。

test_exploration.py覆盖分层温度的独立概率公式、确定性动作不变、采样/串行/缓存批量重放与梯度、配置拒绝、旧父权重显式迁移、CPU及CUDA双worker两轮更新、温度退火和严格恢复。恢复比较CPU逐位一致，CUDA允许浮点规约舍入差；不作为性能收益证据。

test_harp.py检查HARP/原HPGI/历史混合的独立控制器身份、切换方法不泄漏求解标志、配置不能悄悄变成零遍或仅切分版本、原HPGI不调用新搜索；同权重/RNG下HARP与历史assignment路径的物理执行和原提案概率一致。显式迁移与从头初始化分开，物理合同变化/重标恢复须失败。公共CLI闭环test_baseline_interfaces.py也覆盖HARP的CPU与CUDA双worker：校准/预训练、两层冻结更新、CSV、独立物理复算、评价/回放/汇总及重载继续。微型检查不冒充收敛或性能实验。

test_assignment_solver.py用手算AP瓶颈验证单点不可改善、成对释放/接纳才改善的反例；枚举小型全部动作核对缓存P3与公共分配器成本一致，检查下界、输入不变、联合容量和总接纳数保持。test_hpgi_split_transfer.py同时覆盖联合搜索原提案概率、CPU/CUDA更新及严格恢复；不由小型检查证明完整训练有效。

test_hpgi_split_transfer.py检查固定提案在完整微型闭环中逐动作/概率/价值/特征复现Fixed-Split，切分搜索的联合可行性/不增成本与接纳/服务器保持，原潜在提案的串行/批量概率，以及真实CPU/CUDA更新、并行bootstrap、迁移来源、Adam/新计数、独立物理回放和严格恢复拒用。它不证明新方法全局最优或多种子优越。

test_fixed_split_selection.py覆盖全部合法组合、配对开发输入完整性、按物理槽聚合原目标成本、非有限得分拒用、真实环境执行固定点、原输入不变，以及选择overlay真正进入独立300轮Fixed-Split配置；选择文件篡改须失败。只做微型环境执行和配置流转，不启动正式训练。

test_process_runtime.py检查默认不改OS、非Windows显式拒用、配置布尔校验、200轮预算不提前稳定退出和Trainer入口；Windows子进程通过实际API读回AboveNormal及显式关闭的EXECUTION_SPEED。只修改测试子进程自身，退出后释放，不更改系统电源设置。

test_plotting.py核对后向平滑初始窗口、按独立种子求标准差/可重复bootstrap、异长曲线只取共同坐标、单种子无统计带及重复种子拒用；检查训练中CSV只忽略未完成末行，图形数据快照可在原CSV追加后精确重画，篡改哈希时报错。实际渲染检查需安装可选依赖`.[test,plotting]`，缺少绘图依赖时仅该项跳过；图形产物在系统临时目录中清理，不启动训练或使用GPU。

test_finite_horizon_value.py手算实际采样回报、真实终止/截断/bootstrap、零时长事件、有限时域时钟与单位转换，包括gamma非常接近1时的回报和残差梯度；核对actor仍使用GAE、critic目标独立前向复算、CPU/CUDA串行/批量/整批缓存及梯度一致。并行短训练覆盖尾区间、检查点重载和继续更新。正确性检查不代替长期收益判断。

test_learning_quality.py用手算反例检查非线性时间偏差、终止处价值误差、留一episode时间基线、截断/裁剪拒用和原输入不变；公开动作诊断独立求P3成本，不相信task中现成的成本字段。这些只检查诊断语义，不作为新算法或收敛证据。

test_baseline_interfaces.py覆盖其余七个学习基线的CPU/CUDA训练→评价→CSV→重放/汇总→续训；GPU路径用2worker覆盖并行采样，5槽episode检查尾区间。独立复算GAE/loss/服务分母/能耗，检查冻结组件、原行为概率、方法约束及续训权重一致。另查纯贪心无需训练、层级--set校验、单层校准不调用无用网络、进度编号、输入文件来源、minimal检查点，以及Fixed-Window投影前动作/潜变量的CPU与CUDA批量梯度。产物在系统临时目录内自动清理。

test_hpgi_efficiency.py检查交替组件的重型输入诊断门控、global/phase标准化及参数更新等价、公共加速overlay、原HPGI及HARP的4worker/CUDA预训练/交替/恢复、冻结层不变、行为版本刷新与旧buffer拒用；保存记录由独立GAE/loss和物理检查复算。另覆盖finite_horizon/sampled_return的CPU串行/双worker交替、非单位gamma、尾区间、截断bootstrap、奖励单位冻结及重载续训。

test_sampling_efficiency.py检查采样缓存命中不重建候选、环境复用已验证剖面、输入轨迹只生成一次、缓存返回值隔离，以及宽数值尺度P3容量/KKT；还检查分组件优势标准化、单元素/常量组、配置与dry-run及独立loss复算。rollout诊断记录更新前分组件优势、时隙相关性和价值拟合；不以更新后的explained variance代替动作收益证据。CPU/CUDA等价更新同时覆盖global和phase标准化。

test_update_devices.py覆盖设备校验、CPU worker边界、整批缓存共享存储、CPU/CUDA完整更新数值一致、GPU Adam及回传、异常恢复CPU、跨设备检查点及预训练/联合更新/续训。无CUDA时仅GPU测试显式skip；不把CPU测试通过当CUDA已验收。

test_joint_ppo.py覆盖零时长开关事件与跨phase GAE、物理奖励只计一次、终止/截断、原概率/掩码、两组件参数变化、1/2worker一致性、检查点恢复、原始CSV与独立PPO复算、共同预训练及对照计算量。test_joint_chain.py额外比较混合事件批量/逐样本前向、裁剪loss及每个参数梯度，检查非法动作零梯度、重建物理轨迹/资源前缀、当前请求不泄漏给开关网络、两方法同动作同环境、空任务critic与无审计时的梯度最大值。测试使用系统临时目录；真实输入试跑与合成语义测试分开记录。

test_parallel_collection.py检查同种子串行episode参考与并行buffer/动作一致、GAE边界、尾段bootstrap、行为概率、恢复及短交替更新的快照刷新/冻结层。它不与旧串行全局随机流强行比较。benchmark_collection.py仅测真实训练采样时间和动作/物理结果哈希，不更新网络，不跑独立评价，不保存结果目录。

测试先覆盖 V6 的可手算关系：逐时隙就绪与唤醒、期限最低速率、资源守恒、P3 边界和本地/边缘候选掩码。训练前必须通过这些测试；它们不替代正式实验。

运行：python -m pytest tests -q。test_physics.py覆盖物理、匿名、资源与有界搜索；test_learning.py覆盖联合概率、掩码、GAE、冻结、归一化、尾区间和完整恢复。

test_state_relations.py检查请求分组反例、目标服务器/既有指派、细量纲归一化、真实采样与批量概率/梯度一致及观测存储恢复；test_feasible_scenario.py检查最坏公开区间、可执行调度与重放、小网格全部切分对照及无收益案例。这两组不更新参数；真实剖面未恢复时数据测试显式skip。完整pytest还包含短PPO更新，暂停训练时只选择无需更新的检查。

test_convergence.py覆盖奖励窗口稳定/趋势/波动、两种评价一致判定、尾部重置和恢复；短PPO集成核对实际保存次数、只保留last.pt、详细诊断清理及完整CSV，测试使用固定评价替身只验证停止控制，不作为性能结果。轨迹清理测试覆盖upper/lower/joint命名、重复清理和路径穿越拒绝；test_joint_ppo.py以8批预训练＋23轮联合微型训练跨越每10阶段详记/保留2组时的首次joint淘汰，检查全部CSV保留及重载第24轮。空请求诊断统计无分母时返回null。

python -m mist_sim.verification先执行这些测试，再完成全部方法16/17隙短验证及HPGI续训，并在同一个临时空间完成CLI/日志核验；退出后自动删除权重、轨迹和索引，不写正式logs/runs/results。pytest的tmp_path也随用随清。仅需保留排障现场时显式指定--keep-artifacts 新目录；后续处理完自行清理，不默认归档。无性能阈值，不要求收敛。独立SciPy对照可能报告SLSQP尝试点被裁到界内的警告，必须以收敛状态及最终目标差判定。
