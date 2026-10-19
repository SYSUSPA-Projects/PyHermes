# 精确径向分组2PCF实测（2026-10-16）

科学提交356d4f6；基线4fd7d41。实验分支codex/2pcf-optimization，默认旧路径不变，未合并/推送main。

每项仍保留原alias scaling权重及折叠multiplicity，仅按整数a²+b²+c²相同项重排求和；无插值/截频/fastmath。单run构建MPI分工、线程私有径向直方图，全局SUM后分工计算球边界；完整准备/内存分配/归约均计时。自定义和复杂窗口沿用原回退，256MiB/rank保守scratch上限。

126924 COMPLETED/0（12:04），cn5同一32物理核96GiB/30min，无GPU；6小网格门槛、36正式（J7/8/9×8x4/4x8×两方法×三次），顺序轮转。基线J7/J8为shell_edges/per_rank，J9为shell_edges_batch/shared；新路径统一shell_edges_radial/shared。

|J|布局|基线总时间中位 / s|新路径中位 / s|变化|
|---:|---|---:|---:|---:|
|7|8×4|0.337142|0.197389|-41.5%|
|7|4×8|0.356451|0.197794|-44.5%|
|8|8×4|1.729180|0.534144|-69.1%|
|8|4×8|1.298701|0.609023|-53.1%|
|9|8×4|7.905808|3.013761|-61.9%|
|9|4×8|7.180856|3.283341|-54.3%|

六组均通过预定总时间中位下降3%的采用条件；全部快慢/启动/预热样本保留。J7/J8收益包含FFT从per_rank变为shared，不能全归径向分组；J9对照两侧均shared。不同分项最大rank累计可重叠，不能相加作为wall时间。

两布局各三次取最小，再按指标独立择小；本轮两指标均8×4最快：

|J|投影至结束总秒|主循环秒|
|---:|---:|---:|
|7|0.169722|0.029206|
|8|0.532926|0.070165|
|9|2.687215|0.292769|

总时间含投影、首次FFT/功率及广播和完整31bin汇集；主循环不含初始FFT/功率分发，包含所有径向工作。排除公共IO、导入、小J预热/JIT、SHA/线程审计。参考软件继续用旧完整曲线时间：pycorr0.373189695、TreeCorr1.813970067，两表参考列计时边界须区别解释。不同J不是等精度比较。

68本地检查、每rank8项真实4rankMPI检查通过；36曲线+6门槛、源110文件/输入/投影/输出SHA、实际物理核/线程mask、独立统计与10表行均核验。最大绝对xi差1.1972645e-12（rtol1e-8/atol1e-11）；重排求和不保证位级相同。

J9新增径向scratch保守上限约60MiB/rank（4线程）或84MiB/rank（8线程），本轮均未触发回退。RSS为0.1秒进程树采样且共享页可重复；完整范围见报告，不以整Slurm MaxRSS代替单case。

首次partial报告误将baseline batch当作已启用candidate-only外层计数器；仅本地报告修复并保存v1，内置sweep/count与数值/SHA正常，冻结源/有效测量不动且无重跑。

启用示例（内置球壳，保持默认卷积路径作为默认）：

```python
Corr_2PCF({"Corr_2PCF": dict(
    pair_reduction='spectral', reuse_input_fields=True,
    spectral_window='shell_edges_radial', spectral_fft='shared',
    memory_strategy='speed',
    # binning_window使用内置binned_shell/thick_shell及原采样配置
)})
```

完整报告和两表位于本地测量归档fig13_2pcf_radial_optimization_20261016/analysis；服务器独立归档deliveries/fig13_2pcf_radial_optimization_20261016，不嵌入冻结output。原图/文章、其他任务和用户GPU均不在本轮范围；无新自动跟进。
