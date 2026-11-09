# 脚本入口

从mist-sim根运行；本机解释器固定`D:/Anaconda/envs/pytorch/python.exe`。scripts只编排，公共实现属于src/mist_sim。训练/评价/重放使用`python -m mist_sim`，见[工程README](../README.md)。

|目录|入口|用途|
|---|---|---|
|training/|compare_mechanisms.py --slots 32768|共同预训练后，等物理槽预算比较HPGI与joint_ppo；默认无独立评价，--evaluation-trajectories N仅结束时评价一次|
|validation/|benchmark_collection.py --checkpoint 路径|同配置同策略比较1/8worker训练采样，默认8条完整轨迹；只打印耗时/哈希，无优化更新、无独立评价、无持久输出|
|data/|profile_models.py|官方模型及输入准备、串行离线测量|
|data/|export_profiles.py --verify-saved|导出/复算已保存剖面|
|data/|check_inputs.py|输入、物理和重放检查；默认临时运行|
|validation/|check_rollouts.py|当前配置快/慢rollout及GAE/概率/物理复算；默认仅打印，--summary可显式导出排障摘要|
|validation/|check_profile_chain.py|真实剖面的48槽正确性短更新、重载和续训一致性，退出清理原始产物|
|validation/|inspect_logs.py|复算已有轨迹、PPO及物理中间量|
|validation/|benchmark_batching.py|串行参照与批量张量前向/反向等价及交错计时；默认合成输入，--behavior可读已有轨迹，仅打印|
|validation/|benchmark_update_device.py --checkpoint 路径|独立读取兼容joint检查点，CPU采集最多64槽；32/64事件比较CPU/CUDA前向、反向、裁剪与Adam，单列输入整理/传输及权重回传，检查loss/梯度一致；仅更新内存副本，不写结果、不切换现有训练设备|
|validation/|report_hpgi_probe.py --run 路径|检查既有训练资产、日志及耗时；不继续训练|
|validation/|verify_commands.py --workspace 路径|完整验证套件的CLI子检查|
|validation/|build_verification_index.py --workspace 路径|检查临时套件资产及哈希|
|validation/|check_training_scenario.py|单请求可行上界；--witness构造调度并重放，--exact有界穷举预定义小网格及最佳固定切分，--seed可重复指定；无网络更新，默认仅打印|
|validation/|inspect_training_design.py|历史固定奖励尺度探针，仅用于重建当时检查，不作为当前归一化规格|
|maintenance/|check_environment.py|Conda与导入路径核验|
|maintenance/|check_publish.py|推送前检查暂存文件和Git历史，不输出秘密内容，不访问远端|
|maintenance/|catalog_runs.py|只读运行资产盘点|
|maintenance/|archive_runs.ps1、restore_run.ps1|既有历史资产工具，按明确任务使用，不是测试产物默认流程|

profile_models与export_profiles的持久输出在data/profiling，数值输入在data/processed。检查、诊断、维护脚本默认打印到终端；--output或--summary仅用于显式排障导出（放tmp，用完删除）。验证套件的报告、日志、CSV均位于系统临时空间，退出删除。正式训练/评价自动写results/<运行名>中的原始CSV；后续图表以CSV为输入，不能把results作为检查脚本默认落盘目录。

训练用量预览用`python -m mist_sim train --config configs/conditional.json --dry-run`；TensorBoard与停止/恢复命令在README维护。测试产物默认临时目录、退出删除；需要保留排障原始文件时显式指定新目录。正式结果保留成功和失败。不创建空plots目录；有绘图入口时再增加。
