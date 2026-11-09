# 外部参考

## 科研绘图

使用[SciencePlots 2.2.2](https://pypi.org/project/SciencePlots/2.2.2/)（[上游](https://github.com/garrettj403/SciencePlots)，MIT，Copyright © 2018 John Garrett）作为可选绘图依赖，直接调用science/ieee/no-latex样式，未复制样式文件。实际安装包版本和三个样式文件SHA-256写入每张图所在目录的provenance.json；中文字体/配色/尺寸覆盖见docs/PLOTTING.md及configs/plotting/convergence.json。

强化学习绘图流程参考[Spinning Up plot.py](https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/utils/plot.py)，固定提交038665d62d569055401d91856abb287263096178，MIT；查阅了独立Unit、窗口平滑与sd阴影实现，没有复制其源码或过时的Seaborn tsplot。MIST明确采用后向平滑、种子间样本标准差或种子bootstrap，单种子不制造统计阴影。

## MERA

`MERA/` 来自官方仓库 https://github.com/ziqiwang20/MERA ，MERA原论文§VI直接给出此URL。2026-11-07固定提交为 `9a03f8d04b3fb40383af4460ffbef96e868ea01d`；父仓库忽略其完整检出，避免嵌套仓库误提交。

在02_实验目录重建：

```powershell
git clone https://github.com/ziqiwang20/MERA.git references/MERA
git -C references/MERA checkout --detach 9a03f8d04b3fb40383af4460ffbef96e868ea01d
git -C references/MERA status --short
```

只在目录不存在时clone；已有目录先核对origin、HEAD及未提交修改，不自动覆盖。

结果管理核查：上述提交的resource_rl.py通过TensorBoard记录逐步Reward和Energy等指标，未提供可直接复用的训练结果CSV导出链路。MIST独立保存iteration/episode/评价CSV，日志仅用于排障；不把参考仓库的记录方式当成强制目录标准。

|文件|可参考内容|不可直接迁移的内容|
|---|---|---|
|environment.py|状态、动作、step分工|旧Gym接口；导入即读CSV和随机改服务器资源；不是V6期限/切分/唤醒模型|
|kanonymity.py|匿名处理与控制环境分开|距离及均值处理不能直接当作V6保守区间覆盖保证|
|Objective.py|评价函数独立封装|能效比不能直接替代V6的焦耳能耗与时延成本|
|resource_rl.py|PyTorch、参数入口、TensorBoard日志|文件自述DDPG，不能作为本文双层PPO实现；不能仅凭论文标题认定代码算法等价|
|dataset/|CSV输入组织|README称合成数据；不是DNN计算量/张量/功耗剖面|

初步静态读取还发现environment.py与resource_rl.py导入 `test_kanonymity`，当前检出仅提供 `kanonymity.py`，因此没有宣称原仓库可直接运行。本轮没有运行上游代码。根目录未见LICENSE文件；克隆供本地阅读不等于已获代码或数据再分发许可，后续若移植须确认许可、注明来源并核对模型。不要原样安装带Linux构建包的MERA_env.yml到本机。
