# 测试范围

test_hpgi_efficiency.py检查交替组件的重型输入诊断门控、global/phase标准化及参数更新等价、公共加速overlay、4worker/CUDA的预训练/交替/恢复、冻结层不变、行为版本刷新与旧buffer拒用；保存记录由独立GAE/loss和物理检查复算。

test_sampling_efficiency.py检查采样缓存命中不重建候选、环境复用已验证剖面、输入轨迹只生成一次、缓存返回值隔离，以及宽数值尺度P3容量/KKT；还检查分组件优势标准化、单元素/常量组、配置与dry-run及独立loss复算。rollout诊断记录更新前分组件优势、时隙相关性和价值拟合；不以更新后的explained variance代替动作收益证据。CPU/CUDA等价更新同时覆盖global和phase标准化。

test_update_devices.py覆盖设备校验、CPU worker边界、整批缓存共享存储、CPU/CUDA完整更新数值一致、GPU Adam及回传、异常恢复CPU、跨设备检查点及预训练/联合更新/续训。无CUDA时仅GPU测试显式skip；不把CPU测试通过当CUDA已验收。

test_joint_ppo.py覆盖零时长开关事件与跨phase GAE、物理奖励只计一次、终止/截断、原概率/掩码、两组件参数变化、1/2worker一致性、检查点恢复、原始CSV与独立PPO复算、共同预训练及对照计算量。test_joint_chain.py额外比较混合事件批量/逐样本前向、裁剪loss及每个参数梯度，检查非法动作零梯度、重建物理轨迹/资源前缀、当前请求不泄漏给开关网络、两方法同动作同环境、空任务critic与无审计时的梯度最大值。测试使用系统临时目录；真实输入试跑与合成语义测试分开记录。

test_parallel_collection.py检查同种子串行episode参考与并行buffer/动作一致、GAE边界、尾段bootstrap、行为概率、恢复及短交替更新的快照刷新/冻结层。它不与旧串行全局随机流强行比较。benchmark_collection.py仅测真实训练采样时间和动作/物理结果哈希，不更新网络，不跑独立评价，不保存结果目录。

测试先覆盖 V6 的可手算关系：逐时隙就绪与唤醒、期限最低速率、资源守恒、P3 边界和本地/边缘候选掩码。训练前必须通过这些测试；它们不替代正式实验。

运行：python -m pytest tests -q。test_physics.py覆盖物理、匿名、资源与有界搜索；test_learning.py覆盖联合概率、掩码、GAE、冻结、归一化、尾区间和完整恢复。

test_state_relations.py检查请求分组反例、目标服务器/既有指派、细量纲归一化、真实采样与批量概率/梯度一致及观测存储恢复；test_feasible_scenario.py检查最坏公开区间、可执行调度与重放、小网格全部切分对照及无收益案例。这两组不更新参数；真实剖面未恢复时数据测试显式skip。完整pytest还包含短PPO更新，暂停训练时只选择无需更新的检查。

test_convergence.py覆盖奖励窗口稳定/趋势/波动、两种评价一致判定、尾部重置和恢复；短PPO集成核对实际保存次数、只保留last.pt、详细诊断清理及完整CSV，测试使用固定评价替身只验证停止控制，不作为性能结果。空请求诊断统计无分母时返回null。

python -m mist_sim.verification先执行这些测试，再完成全部方法16/17隙短验证及HPGI续训，并在同一个临时空间完成CLI/日志核验；退出后自动删除权重、轨迹和索引，不写正式logs/runs/results。pytest的tmp_path也随用随清。仅需保留排障现场时显式指定--keep-artifacts 新目录；后续处理完自行清理，不默认归档。无性能阈值，不要求收敛。独立SciPy对照可能报告SLSQP尝试点被裁到界内的警告，必须以收敛状态及最终目标差判定。
