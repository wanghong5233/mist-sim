# 配置约定

配置区分模型假设、工作负载、算法超参数和运行设置。没有依据的数字不得写成测量值。解析后的配置在运行开始时复制到对应 `logs/runs/<run_id>/resolved_config.json`。

smoke.json组合environments/synthetic.json与algorithms/ppo.json；两者键不能重叠。原有平铺JSON仍可读取。fixture_kind和source明确标明资源/业务参数为assumption，DNN剖面为合成。PPO参数是工程默认值，非调优结果。顶层method由CLI --method解析并保存生效值；信息模式、Q和Fixed-Window速率分支不会通过改标签冒充另一方法。合法固定切分必须存在于每个模型。

用--overlay文件和--set key=JSON值覆盖，顺序是基础组合→overlay（按给定次序）→--set→方法必需设置→完整校验。不识别的顶层/PPO键拒绝；网络宽度、批大小、epoch必须为正整数，学习率/梯度阈值等必须正，种子不能重叠。上下层独立调参使用ppo_layers.upper/lower，禁止各自修改gamma。更新后配置是新的运行输入，不覆盖旧run。

示例overlay：{"ppo_layers":{"upper":{"hidden":32,"learning_rate":0.0001}},"training":{"cycles":2}}。无需新增配置框架或按方法复制源码。validation.min_service_rate可为null（仅工程检查）或预定服务阈值；没有达到阈值时selection记录no_eligible_checkpoint，不能自动降低要求。

device取cpu或cuda，仅控制PPO更新，采样/环境/GAE与推理保持CPU；可用CLI --device覆盖。CUDA不可用时报错，不静默回退。algorithms/cuda_update.json是叠加到conditional.json的overlay，仅选择CUDA和minibatch128，保留CPU配置用于对照；128是本机更新吞吐与梯度步数的折中，不是已验证的收敛最优值。相同源码检查点允许显式改device，其他配置仍严格匹配。

algorithms/always_on_finite_horizon.json只训练全开服务器下的调度PPO，继承已用于HPGI的有限时域critic和sampled_return，rollout改为2048槽；与training/fixed_200.json组合为200轮、417792训练槽，与HPGI物理总预算一致。CUDA、4worker和Windows进程调度分别由现有overlay选择；不复制环境配置、不将200个点视为等优化器预算。

训练/验证/测试基础种子分别1000/2000/3000；训练子轨迹用训练基础种子加10000倍计数器，实际值逐条存档。17隙短验证训练基础种子1001、测试3001，仍与验证2000分开。这些种子只用于正确性，不作性能证据。正式配置待真实剖面及参数来源确定后增加；只用JSON，不引入Hydra。
