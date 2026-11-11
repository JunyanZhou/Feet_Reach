# 0.77 m 桌面：训练策略扶桌脚拨物块

这是当前可复现的**自由基座 Isaac 物理结果**。机器人从靠近桌面的姿态开始，左手接触桌面并承受向上支撑力，右脚承重，左脚伸入桌下把物块向外拨出。右手没有目标点。控制回路中没有运动学搬运机器人根部或物块；MuJoCo EGL 只把已记录的 Isaac 状态渲染成视频。

与仓库中更早的 [`README.md`](README.md) 脚本/IK 演示，以及 [`SUCCESS.md`](SUCCESS.md) 的较高桌面 1/32 策略结果不同，本页记录的是正常高度桌面、经过三个独立种子检验的组合控制器。完整的逐项尝试和失败原因见 [`LOW_TABLE_PROGRESS.md`](LOW_TABLE_PROGRESS.md)，全部 78 次批量评估及其原始 JSON 见 [`EVALUATION_LEDGER.md`](EVALUATION_LEDGER.md)。

## 场景、控制器与验收

| 项目 | 数值或方法 |
| --- | --- |
| 桌面上表面 / 前沿 | 0.77 m / x=0.280 m |
| 物块初始中心 | (0.315, 0.095, 0.061) m |
| 物块尺寸 / 质量 / 摩擦 | (0.06, 0.09, 0.12) m / 0.08 kg / 0.15 |
| 物块初始前沿 | x=0.285 m，仅在桌沿内侧 5 mm |
| 初态 | 左手已靠近桌面，随后立即建立接触；无需从远处走近 |
| 全身动作 | [`dagger_2.pt`](../../artifacts/table_policy/supported_bootstrap_dagger/dagger_2.pt) 神经网络输出全部 29 关节动作 |
| 脚部残差 | [`success_gen07.json`](../../artifacts/table_policy/late_two_phase_cem_y095/success_gen07.json) 对五个左腿关节叠加经 Isaac 物理搜索得到的七阶段参数 |

全身网络以仓库已有的 `artifacts/unified_support_reach/v2_goal_offset_x10.pt` 初始化，在 Isaac 自由基座物理中用关节参考做行为克隆，再在学生自己的状态上做两轮 DAgger；参考动作只用于训练标签，推理时不直接控制关节。左腿残差由 CEM 在 Isaac 接触动力学中优化，时间窗固定、幅值为搜索得到的参数。它尚未成功蒸馏进单一神经网络；运行时必须同时加载 `.pt` 和 `.json`。这些不是从零随机初始化训练得到的端到端策略。

“中心越沿”要求物块中心 x<0.280 m；“完整离桌”进一步要求按末帧物块四元数和三维尺寸计算的**最靠桌内角** x<0.280 m。两者都要求 root 全程高于 0.50 m、左手向上受力与右脚承重重叠超过 5 s，以及支撑时脚靠近物块并伴随持续外移。早期日志曾用不考虑物块旋转的固定半宽判断完整离桌；下表和当前代码均采用修正后的三维几何条件。

## 独立结果

| 评估 | 稳定 root | 支撑下中心越沿 | 严格完整离桌 |
| --- | ---: | ---: | ---: |
| seed 53，32 条 | 28 | 20 | 5 |
| seed 55，32 条 | 31 | 20 | 3 |
| seed 57，32 条 | 30 | 23 | 7 |
| 合计 | 89/96 | 63/96 | **15/96** |
| 去掉桌面上板，seed 53，16 条 | 0 | 0 | 0 |

最终展示的 [9 秒视频](../../artifacts/table_policy/table_supported_full_retrieval_final_policy_seed53_env04.mp4)取自**独立** seed53 的 env4：物块中心向外移动 10.95 cm，旋转后最靠桌内角 x=0.27614 m，比桌沿向外 3.86 mm；root 最低 0.543 m，左手向上支撑与右脚承重重叠 8.68 s。[逐帧证据图](../../artifacts/table_policy/final_policy_success_evidence_seed53_env04.png)和[该轨迹数据](../../artifacts/table_policy/late_two_phase_cem_y095_eval_success07_seed53/candidate_00_env_004/policy_rollout.npz)一同保存。去掉桌面上板后，16/16 条都跌倒；这证明**此控制器**依赖桌面支撑，但不能证明所有无支撑策略都够不到该物块。

修正验收代码为“支撑重叠超过 5 s”后，重新执行 seed57 的 32 条评估，结果仍为 7 条严格成功、23 条中心越沿；选中的 env13 与先前独立回放的 13 组状态数组逐元素相同。复现生成的[第二条视频](../../artifacts/table_policy/reproduced_final_policy_seed57_verified/policy_video.mp4)和[机读汇总](../../artifacts/table_policy/first_task_final_summary_20261111.json)也已保存。

## 一键复现

需要本项目的 Isaac Sim 4.5、Isaac Lab 2.2 环境和 `.venv-isaac`。在仓库根目录运行：

```bash
bash scripts/table_retrieval/reproduce_verified_task.sh
```

脚本加载仓库内的最终 `.pt` 与 `.json`，在 Isaac 中运行 seed57 的 32 条轨迹，选第一条严格成功轨迹，并使用 MuJoCo EGL 离线渲染。训练输入数据位于 `dataset/table_retrieval_supported/train`，生成脚本为 `make_supported_clip.py`。仓库中保存了全部批量评估 JSON、CEM 历史及候选参数，便于复核搜索与独立验收之间的差别。

## 目前做不到的事

- 严格成功率只有 15/96，且最好的结果仍只比桌沿多 3.86 mm，不能称为鲁棒。
- 物块初始前沿仅在桌沿内侧 5 mm。中心再向桌内移 5 mm 或 15 mm，独立 seed53 的 32 条都没有完整离桌；追加 10/12 代末段残差搜索也没有找到合格候选。
- 从自由站立到桌边、自主建立最初手部支撑、拨出后的手抓取与恢复行走尚未完成。
- 纯神经网络蒸馏在独立评估中失去脚物接触；当前交付是全身网络加训练得到的低维残差，而非单一权重文件。

这些限制、所有试过的场景参数、效果和失败原因都保留在详细记录及每批原始 JSON 中。
