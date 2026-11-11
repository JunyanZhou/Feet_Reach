# 正常高度桌面：扶桌脚拨物块进展

**当前可复现结果：**0.77 m 桌面、靠桌初态、物块前沿在桌沿内侧 5 mm。全身行为克隆/DAgger 网络加物理搜索得到的左腿残差，在三个未参与搜索的种子中完整离桌 **5/32、3/32、7/32，合计 15/96**。这只是场景特定的演示，不是通用脚部取物能力；把物块再向桌内移 5 mm 的独立评估即为 0/32 完整离桌。运行方法及视频见 [TRAINED_POLICY.md](TRAINED_POLICY.md)。本文件记录过程与失败原因；[EVALUATION_LEDGER.md](EVALUATION_LEDGER.md) 列出全部 78 次批量评估，原始 JSON 与 CEM 历史一同保存。

## 本轮目标与验收

本记录覆盖 2026-11-11 16:36 UTC 起的前期试验，以及 18:37 UTC 起追加的两小时训练与验收。G1 从左手已在桌面上方、左脚开始抬起的姿态起步；桌面高 0.77 m，物块在桌沿内侧。右手没有 EE target。目标是在自由基座 Isaac 物理里由训练后的策略输出全部关节动作，左手持续获得向上接触力、右脚承重，左脚绕到物块后方并向外拨到桌沿外。MuJoCo 仅用于渲染 Isaac 轨迹。

评估记录全部 rollout，不以倒地后物块碰巧移动作为成功。`eval_batch_policy.py` 记录最低 root 高度、手向上支撑时间、右脚承重时间、受支撑的脚物块接近和向外位移。

桌沿前缘 x=0.280 m。“中心越沿”指物块中心 x<0.280 m，物块至少半截露出桌沿；这只是可抓取区域的几何代理，尚未执行手抓取验证。更严格的“完全越沿”要求按物块最终姿态计算的三维最靠桌内角 x<0.280 m。两个指标分别为 `front_edge_crossing_success` 和 `task_success`，不可混用。越沿同时要求 root 始终高于 0.50 m、左手向上与右脚承重重叠超过 5 秒，且支撑状态下脚靠近物块时发生持续外移。

## 已保存资产

- `dataset/table_retrieval_supported/train`：从已有低桌轨迹的第 200 至 750 帧裁剪，从已扶桌姿态起步；生成脚本 `make_supported_clip.py`。
- `cfg/task/G1/G1_table_retrieval_supported_rl.yaml`：0.77 m 桌面、自由物块、实际接触奖励和跌倒终止。
- `artifacts/table_policy/supported_bootstrap/bc_expert.pt`：用自由基座物理中的直接关节参考行为克隆获得的 student。参考只用于训练标签，推理时所有关节由 student 输出。
- `artifacts/table_policy/supported_bootstrap_dagger/dagger_2.pt`：在 student 自己产生的状态上继续 DAgger；两轮各 250 控制步，最低 root 高度分别 0.534/0.536 m，物块尚未移动。对应 `report.json`。
- `artifacts/table_policy/supported_cem/best.json`：冻结上述 student，经过 Isaac 物理搜索得到的三阶段左腿残差参数。此参数是训练结果，不是人工指定的关节轨迹。

## 已测结果

1. 原扶墙权重在低桌 16 个确定性 rollout 中 0 成功，约 1 秒内失稳；`supported_baseline/batch_metrics.json`。
2. 旧低桌权重会在倒地后推动物块，但最低 root 仅约 0.10 m，不计成功；`supported_stage2_baseline/batch_metrics.json`。
3. 直接执行运动学关节参考能让机器人支撑约 10 秒，但左脚落回地面并未触物块；这只用于诊断，不计策略成功。`supported_direct_reference/metrics.json`。
4. 行为克隆后的 student 在 0.77 m 桌面上连续约 9 秒保持手部向上接触与右脚承重；最低 root 0.534 m，物块位移为零。`supported_bc_expert_eval/batch_metrics.json`，姿态图 `supported_bc_expert_t4.png`。
5. 常规 PPO 微调把稳定行为破坏，actor KL 达数千；已保存 `supported_train_bc_rl03` 等分支检查点和日志，未把它们当可用策略。
6. 左腿残差物理搜索第一轮让受支撑时的脚部最大 x 到约 0.42 m，物块仅向外约 4.5 mm；保留 `supported_cem/best.json` 和逐代 `history.json`。12 cm 高物块、y=0.12 m 时脚从近侧推入，故调至 y=0.08 m，让脚先从侧面绕过再拉回。
7. `supported_cem_pull_y08/candidate_gen7.json`：独立 16 条 Isaac 回放，7 条净外移超过 1 cm，最大 1.65 cm；桌面 0.77 m、物块 x=0.35/y=0.08 m、高 0.12 m、重 0.08 kg、摩擦 0.15。对应 `supported_cem_pull_y08_eval_gen7/batch_metrics.json`。
8. `supported_cem_pull_y08_wide/candidate_gen0.json`：扩大搜索后，独立 16 条回放中 7 条超过 2 cm、最大 4.77 cm；对应 `supported_cem_pull_y08_wide_eval_gen0`。此参数有物理接触与向外运动，但脚与块横向错位，接触提前中断。
9. `supported_cem_final_pull/candidate_gen0.json`：冻结前五阶段，额外搜索 6.1 秒开始的回拨。独立 16 条回放中 12 条超过 2 cm、6 条超过 4 cm、最大净外移 6.18 cm；最低 root 高度在最佳外移样本为 0.520 m，左手向上支撑和右脚承重贯穿大部分 9 秒。对应 `supported_cem_final_pull_eval_gen0/batch_metrics.json` 和 `candidate_01_env_009/policy_rollout.npz`。尚未让物块完全越过桌沿，不能宣称任务完成。
10. 环境变化消融：把物块宽度从 9 cm 加到 12 cm，16 条均未外移超过 1 cm（过早相撞）；把 y 从 0.08 调到 0.10 m，最大外移降至 4.2 cm；把初始 x 从 0.35 改到 0.33 m，最大外移约 3.1 cm；把摩擦从 0.15 改为 0.6，结果相近。分别见 `supported_cem_pull_y08_width12_eval`、`supported_cem_final_pull_y10_eval`、`supported_cem_pull_x33_eval`、`supported_cem_pull_friction06_eval`。强化 PPO 从 BC 权重出发时仍有极高 KL，动作发散，暂未得到更好的 checkpoint。
11. 移除桌面上板但保留其他条件时，同一控制器在 8 条回放中全部跌倒，最低 root 高度约 0.15 m、右脚承重不足 1 秒，物块未成功外拨；见 `supported_cem_no_table_eval/batch_metrics.json`。这是同一策略对桌面物理支撑的消融，并非所有可能无支撑策略的可达性证明。
12. `supported_cem_yaw_pull/candidate_gen3_fullout.json` 在 CEM 搜索的单个候选中曾使物块净外移 14.3 cm、完全越过桌沿且 root 最低 0.547 m；但独立 32 条回放中 **0 条完整越沿、仅 1 条物块中心越过桌沿**。这只是搜索命中，不作为任务完成证据；见 `supported_cem_yaw_fullout_eval/batch_metrics.json`。
13. 在同样 0.77 m 高桌面下，将物块初始 x 从 0.350 m 改到 0.345 m（仍在桌沿 x=0.280 m 的内侧），`supported_cem_yaw_pull/candidate_gen1.json` 独立 16 条回放中 **4 条物块中心越过桌沿、0 条完全越沿**。最佳净外移 7.13 cm；四条成功越中心线的 root 最低约 0.55 m，左手向上支撑约 8.8 s、右脚承重约 8.8 s。见 `supported_cem_yaw_x345_eval/batch_metrics.json`、`candidate_03_env_003/policy_rollout.npz` 和 `supported_center_crossing_env3.mp4`。视频从 Isaac 轨迹离线渲染，绿区在桌沿外；物块仅部分进入绿区，不应描述成已完全取出。
14. 针对 x=0.345 m 再做物理搜索，得到 `supported_cem_center_x345/candidate_gen0.json`。**独立 32 条回放中 30 条维持支撑稳定，6 条物块中心越过桌沿、23 条在稳定条件下净外移超过 2 cm、19 条超过 4 cm、0 条完全越沿**；平均净外移约 4.0 cm，最佳 7.19 cm。见 `supported_cem_center_x345_eval_gen0/batch_metrics.json`、`candidate_04_env_013/policy_rollout.npz` 和 `supported_center_crossing_32batch_env13.mp4`。这份 32 条评估是当时优先复现的阶段成果。
    - 该视频对应的 Isaac 轨迹中，控制步 280–340（约 5.6–6.8 s）物块向外 5.24 cm，左脚向机器人侧退 5.67 cm；同期左手平均向上接触力约 18.8 N、右脚平均约 183.7 N、root 高度始终不低于 0.568 m。随后物块继续滑出约 2.0 cm。运动由接触物理产生，而非移动物块坐标。
15. 对同一 x=0.345 m 策略移除桌面上板，独立 8 条全部跌倒，最低 root 高度约 0.15 m，0 条越沿；见 `supported_cem_yaw_x345_no_table_eval/batch_metrics.json`。
16. 按四个环境同时评分的进一步搜索得到 `supported_cem_robust_x345/candidate_gen0.json`，但独立 32 条只有 1 条中心越沿、20 条外移超过 2 cm；逊于第 14 项的 6/32，故未采用。见 `supported_cem_robust_x345_eval_gen0/batch_metrics.json`。
17. 对第 14 项原候选更换随机种子 55 后，独立 16 条中 15 条维持支撑稳定，只有 1 条中心越沿；8 条在稳定条件下外移超过 2 cm，7 条超过 4 cm。见 `supported_cem_center_x345_eval_seed55/batch_metrics.json`。结果说明动作有可复现的外拨能力，但“越沿”这一几毫米边界非常敏感，尚不能声称稳健完成。
18. 将物块再向桌沿移 5 mm 至 x=0.340 m，同一策略在独立 32 条中 4 条中心越沿、26 条外移超过 2 cm，平均外移约 4.4 cm；见 `supported_cem_center_x340_eval/batch_metrics.json`。物块提前进入脚的路径使净位移降低，未提升越沿率。
19. 将物块 y 从 0.080 m 微调到 0.085 m，独立 16 条中 4 条中心越沿、15 条保持支撑稳定，未显著优于原摆位；见 `supported_cem_center_y085_eval/batch_metrics.json`。

## 早期基线运行命令

评估环境为自由基座 Isaac，不对机器人或物块进行运动学搬运；MuJoCo 只读取 Isaac 逐帧关节/物块姿态渲染。以下命令复现早期基线；最终策略请使用文末的一键脚本：

```bash
cd /root/Feet_Reach
OMNI_KIT_ACCEPT_EULA=Y ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 MEMPATH=/root/Feet_Reach/dataset \
  .venv-isaac/bin/python scripts/table_retrieval/eval_batch_policy.py \
  --checkpoint artifacts/table_policy/supported_bootstrap_dagger/dagger_2.pt \
  --residual-json artifacts/table_policy/supported_cem_final_pull/candidate_gen0.json \
  --task G1/G1_table_retrieval_supported_rl --dataset table_retrieval_supported/train \
  --num-envs 16 --steps 450 --mode mean --object-y 0.08 --object-height 0.12 \
  --object-mass 0.08 --object-friction 0.15 --seed 48 \
  --output artifacts/table_policy/reproduced_final_pull
```

独立复现物块中心越过桌沿的 16 条评估：将上面命令的 `--residual-json` 改为 `artifacts/table_policy/supported_cem_yaw_pull/candidate_gen1.json`，增加 `--object-x 0.345`，将 `--seed` 改为 `52`，并选一个新 `--output` 目录。该候选额外训练了左髋 yaw 的最后一段残差；评估脚本自动从 JSON 读取五个关节名。

## 2026-11-11 18:37 UTC 起的追加实验

这轮把目标改为一个起始时**完整位于桌沿内**、更接近桌沿的小物块：初始中心 x=0.315 m，前缘 x=0.285 m，比桌沿 x=0.280 m 深 5 mm；尺寸 6×9×12 cm。桌高仍为 0.77 m，质量 0.08 kg，摩擦 0.15。仍由 `supported_bootstrap_dagger/dagger_2.pt` 输出全部关节动作，叠加在 Isaac 物理搜索得到的五个左腿关节残差 `supported_cem_center_x345/candidate_gen0.json`；左右手没有手工关节轨迹，右手没有 EE target。初始为已靠近桌面、即将建立手支撑的姿态。

- seed=53 的 32 条独立回放有 17 条满足 `front_edge_crossing_success`（物块中心越过桌沿，root>0.50 m、左手向上支撑与右脚承重重叠>5 s，且脚靠近物块时持续向外运动）；seed=55 的 32 条有 16 条。见 `small_block_baseline_seed53`、`small_block_baseline_seed55`。
- seed=53 最佳样本 env21 净外移 10.0 cm，root 最低 0.553 m，扶桌手/右脚同时承重 8.70 s，支撑状态下脚与物块接近且外移 1.48 s。视频 `small_block_center_crossing_seed53_env21.mp4` 只是 Isaac 原始物理轨迹的 MuJoCo 离线渲染。
- **更正严格验收**：早期 `batch_metrics.json` 用未旋转时的物块半宽判定完整离桌，给出 9/32 与 5/32；物块实际倾斜，不能按该数字宣称完整取出。env21 最终中心 x=0.215 m，但按末帧姿态计算的桌内最远角 x=0.289 m，仍比桌沿深约 9 mm。`eval_batch_policy.py` 已改为按四元数和三维尺寸计算最远角，之后所有 `task_success` 均按实际几何判断。11 秒回放显示物块在约 8.2 秒后不再继续移动。
- 移除桌面上板的 16 条同条件回放全部失稳、0 条中心越沿，最佳样本 root 最低约 0.146 m、右脚承重 0.86 s；见 `small_block_no_table_seed53`。该消融只说明这个控制器依赖桌面支撑，不证明任何无支撑策略都做不到。
- 6 cm 高方块会被脚跨过，32 条无外移；10 cm 高块有接触但最多约 3.3 cm 外移；轻量/低摩擦 12 cm 高块虽然中心仍越沿，真实最远角仍留在桌下。4 cm 深、起始前缘仍在桌沿内 5 mm 的薄块有 24/32 条中心越沿，但真实最远角至少仍在桌内约 8 mm。见相应 `cube6_baseline_seed53`、`block10_baseline_seed53`、`small_block_light_seed53`、`slim_block_seed53`。
- 接触反馈第一版只在搜索批次偶遇完整离桌，独立评估不复现；第二版在匹配训练场景的 32 条中仅 3 条中心越沿，弱于无反馈的 6 条。搜索器曾有物块重置问题：`env.reset()` 未恢复自定义动态物块，导致第二代起初始 x 不是指定值、虚增“成功”。`optimize_contact_feedback.py` 和 `optimize_supported_residual.py` 已在每代手动恢复物块位姿和零速度。历史上从第二代产生的 CEM 分数需要谨慎看待；本轮使用的原候选来自第零代，独立评估不受此问题影响。

### 前一阶段的有效策略与严格取出证据

修复重置并进一步搜索第七段左腿动作后，得到 `small_block_cem_final/candidate_gen10.json`。在物块 y=0.080 m 的两个独立 32 条回放中，物块中心越沿分别为 **22/32** 和 **20/32**，同时保持 root 高度和扶桌/右脚支撑；完整几何离桌仍为 0/64。对应 `small_block_cem_final_eval_seed53`、`small_block_cem_final_eval_seed55`。移除桌面上板后的同控制器 16 条全部跌倒，0 条满足支持条件下的中心越沿；见 `small_block_cem_final_no_table_seed55`。视频 `table_supported_foot_retrieval_policy_seed53_env27.mp4` 对应当前参数的成功“拨至可抓取区域”样本。

保持同一 0.77 m 桌面、同一物块尺寸/质量/摩擦与同一策略，仅把物块 y 从 0.080 m 调至 **0.095 m** 后，物块倾斜方向有利于继续向外滑动。两个独立种子的 32 条评估分别有 **18/32** 和 **18/32** 条满足支持下的中心越沿，其中 **3/32** 和 **2/32** 条按物块最终四元数和三维尺寸计算，**整块最靠桌内的角也越过桌沿**；见 `small_block_y095_seed53`、`small_block_y095_seed55`。seed53 的 env5 中心外移 10.87 cm，root 最低 0.552 m、扶桌手/右脚同时承重 8.60 s、支持状态脚近物块向外运动 1.02 s，最终最靠桌内角 x=0.27933 m，比桌沿 x=0.280 m 向外约 0.67 mm。严格成功视频 `table_supported_full_block_retrieval_seed53_env05.mp4` 是这条 Isaac 物理回放的 MuJoCo 渲染。完整离桌余量很小，因此不应称为鲁棒；但不是搜索批次中的单个偶然命中，跨种子独立评估 64 条共 5 条。

第三个独立种子 57 的 32 条评估有 **22/32** 条中心越沿、**4/32** 条完整几何离桌，三个种子合计分别为 **58/96** 和 **9/96**。seed57 env1 的最靠桌内角最终 x=0.27501 m，比桌沿向外 4.99 mm；root 最低 0.548 m、扶桌手/右脚同时承重 8.68 s。主视频 `table_supported_full_block_retrieval_seed57_env01.mp4`，证据图 `strict_success_evidence_seed57_env01.png`。移除桌面上板后的 y=0.095 m 对照 16 条全部跌倒，0 条满足支持下的中心越沿，见 `small_block_y095_no_table_seed53`。更近的 y=0.0925 m 在 seed53/55 分别得到 4/32、1/32 条完整离桌；y=0.100 m 在 seed53 得到 1/32。它们尚不能替代跨三个种子的 y=0.095 m 结果。

seed57 env1 中，左脚 x 从 0.048 m 伸至最大 0.418 m，超过桌沿 0.138 m；root x 从 0.080 m 前移至最大 0.224 m 后返回，右脚 x 维持约 -0.02 至 0.02 m。说明机器人确实在左手扶桌、右脚承重的条件下向桌内伸脚并改变身体构型。无桌面消融只检验此策略的支撑依赖，不能据此证明其他无支撑策略绝对无法够到该物块。

最终策略的一键复现：`bash scripts/table_retrieval/reproduce_verified_task.sh`。验收代码已统一要求左手/右脚重叠支撑超过 5 秒；原有 15 条严格成功样本的重叠时间最短 8.42 秒，数字不变。按修正后的代码再次实际运行：seed57 的 32 条中仍有 7 条严格成功、23 条中心越沿；选出的 env13 物块、root、关节逐帧 Isaac 轨迹与原回放数组完全一致（13 个数组键中数值数组逐元素最大误差为 0）。运行时只用本地 checkpoint、学习得到的残差参数和 Isaac 物理，渲染不参与控制。修正后复现视频为 `artifacts/table_policy/reproduced_final_policy_seed57_verified/policy_video.mp4`，机器可读汇总为 `artifacts/table_policy/first_task_final_summary_20261111.json`。

尝试给策略后段追加随物块移动的 foot goal：单一目标在独立 32 条中中心越沿 19 条，对照 20 条；16 组目标启用时间/相对位置的每组四条扫描也没有更好的完整离桌样本。见 `small_block_foot_follow_seed55` 和 `foot_follow_sweep_seed58`。直接把组合动作蒸馏进一个神经网络时，教师在 64 条中有 39 条中心越沿，但蒸馏的 300/600/1200 步模型在独立 32 条中均为 0 条，主要是脚接触点在闭环中偏移：1200 步模型虽让左脚 x 最远到 0.421 m，与成功教师约 0.418 m 相近，但约 6 秒时左脚 y=0.245 m，成功教师约 0.178 m，横向错开约 6.7 cm。追加一次学生状态 DAgger 和左腿动作加权损失仍为 0/32。相关模型/日志保存在 `small_block_distilled*`，目前不采用。

最后以旋转后的真实后角位置为奖励重新搜索第七段左腿动作，搜索批次最多出现数条严格离桌，但对第 0/3/8 代候选分别做 seed53 独立 32 条回放，严格成功仅 1/0/1 条；第 1 代候选 seed55 为 0/32。故不以搜索分数替代独立验收，不采用这些候选；见 `strict_backedge_cem_y095` 及对应 `*_eval_*`。

### 最终采用的两阶段残差

继续冻结前五阶段，联合搜索第六、七阶段的五个左腿关节残差。平均评分最高的第 2 代候选在独立 seed53 只有 2/32 条完整离桌，未采用。另存的第 7 代物理成功候选 `late_two_phase_cem_y095/success_gen07.json` 在**三个没有参与搜索的种子** 53/55/57 的 32 条评估中，分别有 **5/32、3/32、7/32** 条完整几何离桌，合计 **15/96**；支持下中心越沿分别为 **20/32、20/32、23/32**，合计 **63/96**。这些数字均由独立 `eval_batch_policy.py` 回放得到，不是 CEM 搜索批次。对应目录 `late_two_phase_cem_y095_eval_success07_seed53`、`..._seed55`、`..._seed57`。相比前一候选的完整离桌 9/96 和中心越沿 58/96 有改善，因此最终采用第 7 代候选。

最终视频 `table_supported_full_retrieval_final_policy_seed53_env04.mp4` 来自独立 seed53 的 env4：物块中心外移 10.95 cm，旋转后最靠桌内角 x=0.27614 m，越过桌沿约 3.86 mm；root 最低 0.543 m，左手向上支撑与右脚承重重叠 8.68 s，支持下左脚邻近物块并使其向外运动 1.12 s。图 `final_policy_success_evidence_seed53_env04.png` 显示物块边界、扶桌手和右脚力、root 高度的逐帧曲线。移除桌面上板的同策略 16 条全部跌倒，0 条满足支持下中心越沿，见 `late_two_phase_cem_y095_no_table_seed53`。这些结果满足当前“扶桌用脚拨到可抓取区域”的演示要求，也有严格完整拨出的真实视频；但 15/96 的严格成功率仍不足以称为鲁棒。机读汇总和最终复现脚本已改用此候选。

距离敏感性：保持最终控制器及其余场景参数不变，把物块中心从 x=0.315 m 移到 x=0.330 m（前沿从桌沿内侧 5 mm 改为 20 mm），独立 seed53 的 32 条中有 30 条保持 root 高于 0.50 m，21 条在支撑下让物块中心越沿，但 **0 条整块离桌**；见 `final_policy_x330_seed53/batch_metrics.json`。以此更深位置重新搜索第六、七阶段左腿残差，12 代、每代 96 条、每候选 3 环境，仍无完整离桌候选；搜索批次最佳最靠桌内角 x=0.29433 m，距桌沿还差 14.33 mm；见 `final_policy_x330_cem/history.json` 与日志。故目前不声称能完成更深物块，也不声称所有无桌面策略均无法够到原摆位。最终控制器保持原参数。

追加同策略 seed53 的 32 条评估：物块中心 x=0.320 m（前沿在桌沿内侧 10 mm）时，31 条 root 稳定、28 条支持下中心越沿、0 条整块离桌；见 `final_policy_x320_seed53/batch_metrics.json`。中心 x=0.330 m 但高度改为 10 cm 时，30 条 root 稳定、仅 1 条支持下中心越沿、0 条整块离桌；见 `final_policy_x330_h100_seed53/batch_metrics.json`。没有把这些较差场景替换成最终演示。

最后对 x=0.320 m 也重新搜索第六、七阶段左腿残差，10 代、每代 96 条、每候选 3 环境；仍然没有完整离桌候选，搜索批次最接近的物块最靠桌内角 x=0.28395 m，距桌沿还差 3.95 mm。见 `final_policy_x320_cem/history.json` 与日志。搜索批次的近似命中没有经过独立成功验收，不采用。

复现最终 32 条批量结果和视频：运行 `bash scripts/table_retrieval/reproduce_verified_task.sh`；脚本使用 `late_two_phase_cem_y095/success_gen07.json`、seed57 和物块中心 (0.315, 0.095) m，选择第一条严格成功轨迹渲染。

两段式控制器由冻结的神经网络 student 和通过 Isaac 物理交互搜索得到的左腿残差构成。初态已经靠近桌面，左手在起步后的短暂过渡中建立接触；右手没有目标点。所有 29 个关节都有 student 动作，四到五个左腿动作叠加经过物理训练的低维残差。残差直接作用于动作空间，尚未蒸馏进单一神经网络权重；使用时须同时加载 `.pt` 和 `.json`。

## 失败原因与对后续训练的指向

| 尝试 | 观察到的效果 | 失败原因或证据支持的解释 | 处理 |
| --- | --- | --- | --- |
| 直接迁移扶墙/旧低桌策略 | 扶墙权重 16/16 在约 1 秒内失稳；旧低桌权重在倒地后才碰到物块 | 侧向扶墙与桌面向上支撑的接触方向、初态和负载分配不同；跌倒后的物块运动不是任务成功 | 改为桌面姿态的行为克隆/DAgger，评估加入 root、手和右脚支撑条件 |
| 只执行关节参考、再行为克隆 | 参考和学生可以扶桌约 9–10 秒，学生物块位移几乎为零 | 参考覆盖的是支撑姿态，没有可靠的脚物块接触；只有模仿损失无法教会外拨 | 冻结稳定全身学生，在真实物理里搜索左腿残差 |
| 直接 PPO 微调 | actor KL 达数千，稳定动作被破坏 | 从已有 BC 权重直接探索导致动作大幅偏离；较早阶段还发现通用 finetune 配置把动作变化惩罚覆盖为过大的 0.5 | 不采用 PPO 分支；保留失败检查点和训练日志在本地，后续需从奖励权重、探索方差及课程重新设计 |
| 初期三至六阶段残差搜索 | 脚最多伸到 x≈0.42 m，但物块只外移毫米到数厘米；部分候选从近侧把物块推进桌内 | 足端路径先碰到物块错误一侧，或横向接触提前中断；仅到达位置不等于外拨 | 调整物块 y 与末段回拨，增加左髋 yaw 和第七阶段 |
| 单次 CEM 命中 | 搜索批次曾出现 14.3 cm 外移和完整离桌；独立 32 条为 0 条严格成功、仅 1 条中心越沿 | 并行搜索中的单条轨迹过拟合，不能替代独立种子验收 | 所有候选用独立 32 条批量评估筛选 |
| 物块重置缺陷 | 早期 CEM 第二代起物块初始 x 偏离指定值，造成虚高分数 | `env.reset()` 没有恢复自定义动态物块的根状态 | 每代显式重置物块位置、姿态和速度；仅采信修复后的搜索与独立评估 |
| 旧严格成功判据 | 原先报告 9/32、5/32；姿态修正后对应样本仍有角留在桌下约 9 mm | 固定半宽忽略物块旋转，误把“中心越沿”当“整块离桌” | 以最终四元数和三维尺寸计算最靠桌内角；台账把旧口径单列 |
| 改物块形状/尺寸 | 6 cm 高方块被脚跨过、32 条无外移；10 cm 高块最多约 3.3 cm；更薄物块虽常中心越沿仍未整块离桌 | 足部碰撞高度和物块翻转几何共同决定有效拨动，不能只调易成功的平面距离 | 保留 12 cm 高、6×9 cm 横截面，并报告所有形状试验 |
| 追加基于物块位置的脚目标或接触反馈 | 单一跟随脚目标 19/32 中心越沿，弱于 20/32 对照；16 组扫描无更好严格结果；反馈独立评估也退化 | 原成功轨迹依赖相位与接触时序，简单闭环修正会把脚带离有效碰撞点 | 不并入最终控制器；保留参数和评估数据 |
| 蒸馏组合控制器为单一神经网络 | 教师 39/64 中心越沿；学生 300/600/1200 步各 0/32，追加 DAgger 和腿部加权损失也 0/32 | 虽然左脚 x 伸展接近教师，接触时 y 偏约 6.7 cm，闭环误差使脚错过物块 | 最终仍运行全身网络加训练得到的残差；未来需接触状态覆盖和闭环损失 |
| 向桌内增加物块距离 | x=0.320/0.330 m 的独立各 32 条均 0 完整离桌；相应 10/12 代 CEM 仍无合格候选 | 稳定支撑仍常见，但末段外拨距离和姿态余量不足；当前演示只适用靠边物块 | 明确限制，后续需新的足部接触课程与可达范围训练 |

这些解释仅限当前场景和控制器；无桌面消融说明它依赖桌面，不能证明任意无支撑策略都无法够到同一物块。原始逐轨迹数字以 [台账](EVALUATION_LEDGER.md) 中链接的 JSON 为准，较早没有三维后角记录的批次不可用于严格成功率比较。

## 后续改进方向

- 完整离桌成功率目前为 15/96；可继续改进首次拨动后物块侧移导致的失接触，并在更多初态和物块位置上训练。
- 当前只完成拨出，尚未训练后续手抓取或从自由站立位置接近桌子的阶段。
- 与另外训练的无支撑脚部 reaching 策略比较可达范围，检验“该物块必须依赖桌面支撑”这一更强的主张；目前只有同一控制器去掉桌面后的消融。
