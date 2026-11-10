# 脚本入口

从mist-sim根运行；本机解释器固定`D:/Anaconda/envs/pytorch/python.exe`。scripts只编排，公共实现属于src/mist_sim。训练/评价/重放使用`python -m mist_sim`，见[工程README](../README.md)。

|目录|入口|用途|
|---|---|---|
|evaluation/|evaluate_greedy.py --suite data/processed/evaluation/conditional-main-comparison-v1 --run-id evaluate-greedy-shared20|冻结20条128槽新请求输入，4进程执行同一贪心方法；无网络/训练。逐轨迹和池化CSV归results，物理记录及校验归logs/runs；逐轨迹样本标准差与池化指标分开，失败不填零。支持--trajectories/--first-seed/--workers/--overlay；叠加runtime/windows_performance.json时主进程和worker继承AboveNormal/HighQoS并保存读回值；既有输入校验哈希并检查训练种子交集|
|evaluation/|python -m scripts.evaluation.select_fixed_split --suite data/processed/development/fixed-split-v1 --run-id select-fixed-split-v1|穷举逐模型合法固定点的笛卡尔积；所有组合使用相同开发输入、公共因果贪心/容量约束/P3，按完整目标成本选择。默认2条128槽开发轨迹、4个CPU进程；保存全部候选CSV/排序/哈希及selected_overlay.json，之后作为固定点overlay传给fixed_split独立训练；不声称全局最优训练策略|
|training/|compare_mechanisms.py --slots 32768|共同预训练后，等物理槽预算比较HPGI与joint_ppo；--overlay将同一加速/学习配置应用到双方。默认无独立评价，--evaluation-trajectories N仅结束时评价一次|
|training/|python -m scripts.training.train_fixed_split --selection-run 选择运行ID --suite 开发输入目录 --run-id 训练ID|Fixed-Split正式工作流：等待选点完成、核对全部候选/物理场景/哈希、冻结选择，从头训练300轮；可重复--wait-run等任一既有300轮训练正常退出后开可见终端，避免第三路PPO竞争显存。低频last、运行/失败/队列状态分别遵循公共接口；不复用旧固定点权重|
|training/|python -m scripts.training.baseline_queue --spec 冻结任务JSON --queue-id 名称|最多两路正式训练的完成驱动队列：既有运行成功300轮并退出后，依次启动Fixed-Split/Fixed-Window；冻结配置、源码和选点来源，AboveNormal/HighQoS、4worker，保留已实测的后台性能设置。失败停止补位，不盲目重试；24小时监控上限不强杀训练，等待及执行期间请求保持唤醒|
|training/|overnight_queue.py --queue-id 名称 --wait-run 运行ID --candidate 运行ID --candidate 运行ID --seed 种子2 --seed 种子3|Windows一次性依赖队列；等指定运行成功200轮并退出后，按两候选尾20轮原始成本暂选，两个可见终端各跑一个新种子/200轮。冻结原配置和源码；失败/失配不抢跑、不重试；logs/queues保存状态/选法/退出码，结果仍归results各运行。等待及训练监控期间请求系统保持唤醒，不要求屏幕亮；完成后释放。24小时仅为监控上限，不强杀训练。两任务成功后调用已有绘图入口生成三种子均值与标准差阴影；旧199轮不补点|
|plotting/|plot_convergence.py --curve "Joint PPO=results/<运行名>"|只读cycles.csv生成中英文学习曲线，支持多算法/独立种子、iteration/物理槽、滑动均值和标准差/置信带；图与复现快照归对应results的figures/convergence，详见docs/PLOTTING.md|
|validation/|benchmark_collection.py --checkpoint 路径|同冻结策略和8条输入轨迹比较2/4/6/8worker；区分冷启动、重复暖池中位数、worker墙钟/CPU/P3及回传等待，核对动作/概率/物理结果哈希；仅打印，旧快照只作诊断|
|validation/|diagnose_learning.py --checkpoint 路径|默认检查策略/熵梯度和固定公开状态的8次局部PPO学习；--joint-updates N --lambdas .95 .8对比GAE lambda，--normalization global或phase控制优势标准化。仅在内存采样/更新，前后用相同两条诊断轨迹比较，不续训旧记录、不写正式结果|
|validation/|inspect_learning_quality.py --run 路径 --stage cycle_022_joint|只读复算奖励分项、可避免的高成本动作、终止处价值误差、按时间位置分组的优势方差；完整gamma=1联合轨迹另与留一episode的剩余时间基线对比。仅打印及输入哈希，不更新参数，不写结果|
|data/|profile_models.py|官方模型及输入准备、串行离线测量|
|data/|export_profiles.py --verify-saved|导出/复算已保存剖面|
|data/|check_inputs.py|输入、物理和重放检查；默认临时运行|
|validation/|check_rollouts.py|当前配置快/慢rollout及GAE/概率/物理复算；默认仅打印，--summary可显式导出排障摘要|
|validation/|check_profile_chain.py|真实剖面的48槽HPGI短更新、独立物理/loss复算、重载和续训一致性；支持--overlay/--set检查GPU及采样配置，兼容minimal/full检查点，退出关闭进程池并清理产物|
|validation/|inspect_logs.py|复算已有轨迹、PPO及物理中间量|
|validation/|benchmark_batching.py|串行参照与批量张量前向/反向等价及交错计时；默认合成输入，--behavior可读已有轨迹，仅打印|
|validation/|benchmark_update_device.py --checkpoint 路径|默认32/64事件CPU/CUDA小批前向/反向/Adam及loss/梯度一致性；--full-rollout比较整批更新CPU64/CUDA64/128/256（5epoch，探针放宽KL限制），包括搬运/KL/回传及峰值显存；--snapshot-only仅为诊断导入旧完成快照，不作续训。仅更新内存副本，不写正式结果|
|validation/|report_hpgi_probe.py --run 路径|检查既有训练资产、日志及耗时；不继续训练|
|validation/|verify_commands.py --workspace 路径|完整验证套件的CLI子检查|
|validation/|build_verification_index.py --workspace 路径|检查临时套件资产及哈希|
|validation/|check_training_scenario.py|单请求可行上界；--witness构造调度并重放，--exact有界穷举预定义小网格及最佳固定切分，--seed可重复指定；无网络更新，默认仅打印|
|validation/|inspect_training_design.py|历史固定奖励尺度探针，仅用于重建当时检查，不作为当前归一化规格|
|maintenance/|check_environment.py|Conda与导入路径核验|
|maintenance/|check_publish.py|推送前检查暂存文件和Git历史，不输出秘密内容，不访问远端|
|maintenance/|catalog_runs.py|只读运行资产盘点|
|maintenance/|optimize_training_runtime.py|限定2026-11-10已审计源码的4worker/关闭调试导出迁移，保留学习状态；正常结束后合并真实曲线前缀和续训段，旧文件不改|
|maintenance/|python -m scripts.maintenance.recover_watchdog --run 运行ID --earlier 上次恢复目录 --output 新恢复目录|限定2026-11-11的GPU互斥/缓存清理源码迁移，保留学习状态及已有后台性能设置；多次中断仍只拼接检查点之前的真实CSV，原始尾部保留。可生成Fixed-Split选点源码兼容凭据；不放宽Trainer.load|
|maintenance/|archive_runs.ps1、restore_run.ps1|既有历史资产工具，按明确任务使用，不是测试产物默认流程|

profile_models与export_profiles的持久输出在data/profiling，数值输入在data/processed。检查、诊断、维护脚本默认打印到终端；--output或--summary仅用于显式排障导出（放tmp，用完删除）。验证套件的报告、日志、CSV均位于系统临时空间，退出删除。正式训练/评价自动写results/<运行名>中的原始CSV；后续图表以CSV为输入，不能把results作为检查脚本默认落盘目录。

训练用量预览用`python -m mist_sim train --config configs/conditional.json --dry-run`；TensorBoard与停止/恢复命令在README维护。测试产物默认临时目录、退出删除；需要保留排障原始文件时显式指定新目录。正式结果保留成功和失败。plotting目录只放统一绘图入口，图片及统计数据在results对应运行或比较目录，不放脚本目录。
