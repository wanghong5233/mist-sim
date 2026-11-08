# 本地输入

仓库只分发`configs/environments/synthetic.json`中的手算合成测试输入。离线测量剖面、图像、模型权重、参数来源记录以及尚未发表的结果留在本机，不进入远端Git历史。

|本机路径|用途|
|---|---|
|raw/|共享原始图像、权重和下载元数据|
|processed/|由剖析程序导出的数值组件|
|LOCAL_SOURCES.md|既有测量来源、版本、许可及校验记录|
|SCENARIO.md|条件场景参数及依据|

本机既有文件迁移时原样保留。新克隆只有代码，不自动附带这些私有资产。`configs/conditional.json`引用本地剖面并校验SHA-256；缺少文件时明确失败，不回退成合成数据。

基础检查：`python -m mist_sim check --config configs/smoke.json`。需真实剖面时，在获准的机器按`scripts/data/profile_models.py --help`准备官方模型/输入，再使用`export_profiles.py`导出；或通过单独获准的渠道恢复现有资产。新测量的哈希与原测量可能不同，应记录新来源并显式更新配置，不绕过校验。

`.gitignore`只允许本README进入data的远端内容。不要通过`git add -f`上传私有数据；正式结果是否发布须另行决定。
