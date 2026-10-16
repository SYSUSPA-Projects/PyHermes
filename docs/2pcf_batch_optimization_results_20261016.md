# 第三轮：批量边界与全任务共享源FFT

科学实现4fd7d41，独立分支codex/2pcf-optimization。新增可选shell_edges_batch和spectral_fft='shared'，默认仍旧路径；未改main、文章、原图或其他项目。

批量核一次读取频谱，累积每rank分配的4个或8个球边界（最多8个一组），临时空间O(L×batch)，不建立多个3D窗口。每个原alias内的频率长度和scaling乘积在半径间复用，仍逐alias保留权重，未把scaling因子提出折叠求和。外层浮点累计顺序略变，数值不要求逐位相同。

共享源FFT在root计算所需实功率谱，随后广播给其他rank；不是零拷贝共享内存。xi自相关从全8/4次源FFT降到1次，仍零逆FFT。所有输入先由既有流程广播，形状/配对结构经collective检查。广播成本计入准备与总时间，主循环不含初始谱准备；不能只看主循环缩短。

54本地数值/回归、每rank7真实4rankMPI测试、8个J5门槛、36项正式测量全部通过。全31bin对原参考allclose(1e-8,1e-11)，最大绝对xi差1.4877e-14；输入/同布局投影/输出SHA、32不同物理核和真实线程mask已核验。原fold/scaling/window/projection helper源码SHA不变。

126898（30项+6门槛）先按预定统一开关规则比较J8/J9。总时间中位J8：edges1.299500、batch1.482144、shared1.617661秒；J9：8.931729、8.217330、7.184237秒。统一候选没有通过预定改善阈值，故保留edges为通用方案；J8的批量最快1.149815秒伴随最慢1.860669秒，全部保留，未按快慢筛选。

J9显示明确的按分辨率收益，因此只补126900的J9共享4×8三次和同批核edges对照三次（2门槛），没有重跑其他有效项。J9两布局的全部新样本范围均低于各自对照范围：8×4总时间中位8.931729→7.184237（-19.6%）；4×8为9.032735→7.488779（-17.1%）。据此J7/J8继续edges，J9可选batch+shared。初始统一规则/结果保持原记录，J依赖补测及选择单独说明，未改冻结科学源。

两作业COMPLETED/0，均cn1且实际32核mask相同，96GiB/30min，无GPU。相同mask仍是两次时间不同的分配，性能归因使用各自作业内对照；原pycorr/TreeCorr历史时间复用，不能当本轮同分配测得。

最终18样本矩阵为每J两布局各3次，按三次最小再在两布局间独立择最小。单位秒：

|方法|总时间|总时间布局|主循环|主循环布局|
|---|---:|---|---:|---|
|pycorr/Corrfunc|0.373190|1×32|0.373190|1×32|
|TreeCorr|1.813970|1×32|1.813970|1×32|
|PyHermes J7|0.335256|8×4|0.190496|4×8|
|PyHermes J8|1.222889|8×4|0.799956|8×4|
|PyHermes J9|7.144238|8×4|4.229767|4×8|

总时间含投影、全部准备、首次FFT/谱及广播、31bin和结果汇集；排除导入、小J预热/磁盘JIT准备、公共IO、SHA/线程审计。参考软件沿用完整曲线时间，非Hermes主循环边界。进程树RSS为0.1秒launcher+后代采样，共享页可能重复；Slurm两整作业MaxRSS28529312K/15039748K另存，不作为单caseRSS。

逐rank分项显示J9批量路径降低边界累计工作，共享FFT还减少并行重复谱计算/分配，改变rank等待位置。collective分项包含等待，不是纯传输；不能将最大rank分项相加或把全部收益只归某一个数学操作。线程/NUMA页与系统运行时原因未在本轮唯一定位，未调整共享环境或系统设置。

使用建议（需内置binned_shell或thick_shell，保持原bin边界/归一）：

```python
options.update(pair_reduction='spectral', reuse_input_fields=True,
               spectral_window='shell_edges_batch' if J == 9 else 'shell_edges',
               spectral_fft='shared' if J == 9 else 'per_rank')
```

自定义callback不按名字推断为球壳；复杂/非octant/磁盘窗口缓存等保留既有回退。缓存仅限单run，每项新进程、新全分辨率场/谱。J8批量方案可继续作为实验选项，但尚无稳定总时间收益证据。局部4标量展开探针只在本机测过，未写入服务器科学源或用于正式表。

原始36项及8门槛、完整启动/预热/慢样本、源/输入/输出SHA与独立统计见主工作目录tests/logs/fig13_2pcf_batch_optimization_20261016；补项在fig13_2pcf_j9_shared_layout_20261016。最新两表为analysis/recommended_total_time.md与recommended_main_loop_time.md，初始统一策略表及selection.json保留。服务器deliveries/fig13_2pcf_batch_optimization_20261016独立归档。无遗留作业、未新建自动跟进、未合并/推送main。
