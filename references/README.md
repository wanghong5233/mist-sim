# 外部参考

`MERA/` 来自官方仓库 https://github.com/ziqiwang20/MERA ，MERA原论文§VI直接给出此URL。2026-11-07固定提交为 `9a03f8d04b3fb40383af4460ffbef96e868ea01d`；父仓库忽略其完整检出，避免嵌套仓库误提交。

在02_实验目录重建：

```powershell
git clone https://github.com/ziqiwang20/MERA.git references/MERA
git -C references/MERA checkout --detach 9a03f8d04b3fb40383af4460ffbef96e868ea01d
git -C references/MERA status --short
```

只在目录不存在时clone；已有目录先核对origin、HEAD及未提交修改，不自动覆盖。

|文件|可参考内容|不可直接迁移的内容|
|---|---|---|
|environment.py|状态、动作、step分工|旧Gym接口；导入即读CSV和随机改服务器资源；不是V6期限/切分/唤醒模型|
|kanonymity.py|匿名处理与控制环境分开|距离及均值处理不能直接当作V6保守区间覆盖保证|
|Objective.py|评价函数独立封装|能效比不能直接替代V6的焦耳能耗与时延成本|
|resource_rl.py|PyTorch、参数入口、TensorBoard日志|文件自述DDPG，不能作为本文双层PPO实现；不能仅凭论文标题认定代码算法等价|
|dataset/|CSV输入组织|README称合成数据；不是DNN计算量/张量/功耗剖面|

初步静态读取还发现environment.py与resource_rl.py导入 `test_kanonymity`，当前检出仅提供 `kanonymity.py`，因此没有宣称原仓库可直接运行。本轮没有运行上游代码。根目录未见LICENSE文件；克隆供本地阅读不等于已获代码或数据再分发许可，后续若移植须确认许可、注明来源并核对模型。不要原样安装带Linux构建包的MERA_env.yml到本机。
