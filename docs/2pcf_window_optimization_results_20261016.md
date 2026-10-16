# 第二轮2PCF窗口优化结果

科学实现提交726bf35，上一版对照e94f114；独立分支codex/2pcf-optimization，默认仍为原卷积路径。本轮源FFT次数不变：每rank一次、全8rank共8次，无逆FFT。

1. fused路径保留每alias的scaling权重，转置原八项折叠并直接作标量内积，免去octant和半谱窗口大数组；单run复用相同1D基函数数组。
2. shell_edges路径再对相邻球壳共享边界积分：31bin由原62次球top-hat评估降为全MPI作业32次/不同pair power。仅内置binned_shell/thick_shell使用此路径，自定义callback始终保持其语义。

39本地数值/回归测试、每rank4真实MPI测试，4个J5门槛、15个正式测量全部通过。31bin对原参考allclose(1e-8,1e-11)，最大绝对xi差5.4401e-15；输入、同布局投影场、输出、源SHA、32不同物理核及真实线程mask已核验。scaling与原fold/projection helper源SHA不变，未将scaling因子提到折叠求和之外。

作业126885 COMPLETED/0，cn1单次32物理核、固定8×4、96GiB/30min。J8三组各三次轮转，以J8总时间中位数选shell_edges后在J9与上一版各测三次。以下各列独立取三次最小值，全部单位秒，不是跨布局最优值。

|J|上一版总时间|共享边界总时间|上一版主循环|共享边界主循环|
|---:|---:|---:|---:|---:|
|8|2.234014|1.261885|1.764195|0.799216|
|9|15.858726|9.385318|12.923254|6.337322|

J8总时间中位：previous2.402590、fused1.876251、edges1.265914秒。J9总时间中位：previous16.392865、edges9.525087秒。fused第一条2.804852、edges第一条1.914231等慢样本全部保留，未当异常剔除。总时间与主循环改善同时存在，不是将准备工作前移造成。

进程树RSS采样范围：J8 previous6.230–6.315GiB→edges5.133–5.504GiB；J9 previous32.590–33.654→edges25.856–26.350GiB。0.1秒launcher+后代采样，共享页可能重复；整作业Slurm MaxRSS34911092K单独记录，不作为单case峰值。

总时间含投影、全部准备、初始FFT/功率谱、边界计算/通信、31bin结果与汇集；排除导入、小J预热、公共IO、SHA/线程审计。主循环沿用rank0日志计时，不含初始FFT；所有边界计算与collective都在主循环内。各rank累计分项不能将最大值相加。

新路径每rank累计窗口工作（取三次最大rank累计的中位）J8约0.763秒、J9约6.009秒；原octant生成分别约1.452/10.005秒，原fold约0.188/1.735秒。改善包含共享边界、减少大数组、基函数复用和任务重分配，不能把全部收益归于单一因素。

启用需要pair_reduction='spectral'、reuse_input_fields=True、spectral_window='shell_edges'，并提供内置球壳窗口；显式边界用type='binned_shell'和R_in/R_out参数及正常mapping。原图benchmark的WINDOW去掉自定义func即可使用等价内置核。通用实octant可选fused；复杂/不支持或磁盘缓存窗口保留materialized回退。所有缓存仅限单run，不跨生产样本复用全场结果。

全量原始测量、所有启动/预热/慢样本、来源与独立统计/服务器SHA核验：主工作目录tests/logs/fig13_2pcf_window_optimization_20261016。服务器独立归档deliveries/fig13_2pcf_window_optimization_20261016，未嵌入冻结输出。没有新自动跟进、无本轮活动作业；未触碰其他作业、main、文章或原图。J7及4×8本轮没有新测量，不用本轮8×4结果声称两布局全局最优。

后续可单独测试全任务共享源FFT/功率谱和边界任务负载分配；本轮未应用这些变更，也未做scaling吸收。
