# 测试范围

测试先覆盖 V6 的可手算关系：逐时隙就绪与唤醒、期限最低速率、资源守恒、P3 边界和本地/边缘候选掩码。训练前必须通过这些测试；它们不替代正式实验。

运行：python -m pytest tests -q。test_physics.py覆盖物理、匿名、资源与有界搜索；test_learning.py覆盖联合概率、掩码、GAE、冻结、归一化、尾区间和完整恢复。

test_state_relations.py检查请求分组反例、目标服务器/既有指派、细量纲归一化、真实采样与批量概率/梯度一致及观测存储恢复；test_feasible_scenario.py检查最坏公开区间、可执行调度与重放、小网格全部切分对照及无收益案例。这两组不更新参数；真实剖面未恢复时数据测试显式skip。完整pytest还包含短PPO更新，暂停训练时只选择无需更新的检查。

python -m mist_sim.verification先执行这些测试，再完成全部方法16/17隙短验证及HPGI续训，并在同一个临时空间完成CLI/日志核验；退出后自动删除权重、轨迹和索引，不写正式logs/runs/results。pytest的tmp_path也随用随清。仅需保留排障现场时显式指定--keep-artifacts 新目录；后续处理完自行清理，不默认归档。无性能阈值，不要求收敛。独立SciPy对照可能报告SLSQP尝试点被裁到界内的警告，必须以收敛状态及最终目标差判定。
