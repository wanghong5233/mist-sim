# 脚本入口

从mist-sim根运行；本机解释器固定`D:/Anaconda/envs/pytorch/python.exe`。scripts只编排，公共实现属于src/mist_sim。训练/评价/重放使用`python -m mist_sim`，见[工程README](../README.md)。

|目录|入口|用途|
|---|---|---|
|data/|profile_models.py|官方模型及输入准备、串行离线测量|
|data/|export_profiles.py --verify-saved|导出/复算已保存剖面|
|data/|check_inputs.py|输入、物理和重放检查；默认临时运行|
|validation/|check_rollouts.py|当前配置完整快/慢rollout，打印transition、独立复算GAE/后继/概率/物理；不更新参数，仅保留轻量摘要|
|validation/|check_profile_chain.py|真实剖面的48槽正确性短更新、重载和续训一致性，退出清理原始产物|
|validation/|inspect_logs.py|复算已有轨迹、PPO及物理中间量|
|validation/|report_hpgi_probe.py --run 路径|检查既有训练资产、日志及耗时；不继续训练|
|validation/|verify_commands.py --workspace 路径|完整验证套件的CLI子检查|
|validation/|build_verification_index.py --workspace 路径|检查临时套件资产及哈希|
|validation/|check_training_scenario.py|开发请求可行性诊断，不搜索场景|
|validation/|inspect_training_design.py|历史固定奖励尺度探针，仅用于重建当时检查，不作为当前归一化规格|
|maintenance/|check_environment.py|Conda与导入路径核验|
|maintenance/|check_publish.py|推送前检查暂存文件和Git历史，不输出秘密内容，不访问远端|
|maintenance/|catalog_runs.py|只读运行资产盘点|
|maintenance/|archive_runs.ps1、restore_run.ps1|既有历史资产工具，按明确任务使用，不是测试产物默认流程|

测试产物默认临时目录、退出删除；需要保留排障原始文件时显式指定新目录。正式结果保留成功和失败。不创建空plots目录；有绘图入口时再增加。当前规格替换维护，脚本迁移历史和旧命令由Git保留。
