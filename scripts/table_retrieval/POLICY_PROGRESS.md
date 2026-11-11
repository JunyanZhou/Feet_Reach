# 扶桌拨取：policy 训练进度

任务：G1 从站立开始，左手接触桌面，左脚进入桌下将自由刚体物块向外拨至桌沿外，然后收脚并保持站姿。允许固定的 root、手、双脚笛卡尔目标；关节动作由 student policy 输出。成功视频必须来自完整动力学 rollout 的状态记录。

验收：在 Isaac 自由基座物理仿真中，物块从桌下移到桌沿外；左手持续接触桌体至少 0.5 s、右脚承重至少 0.5 s；物块移出后 root 仍高于 0.55 m。完整站立起步版本还要求左脚回到地面。保存 checkpoint、逐帧状态、指标 JSON 和对应视频；不把 IK 直接驱动或仅运动学演示计作策略成功。

## 环境

- 2026-11-11 05:15 UTC 开始；五小时窗口至约 10:15 UTC。
- 机器：RTX 4090 24 GB、24 CPU 配额。安装了独立 Python 3.10 环境 `/root/Feet_Reach/.venv-isaac`、Isaac Sim 4.5.0、Isaac Lab v2.2.0 和项目依赖。
- `scripts/unified_support/run.sh` 的单点烟雾评估通过，原 student 对一个墙面脚目标的末秒平均误差 2.76 mm。
- `scripts/table_retrieval/prepare_policy_reference.py` 从既有 IK 演示生成 801 帧、50 Hz 参考；该轨迹只用于训练目标，不是动力学成功证据。
- `cfg/task/G1/G1_table_retrieval.yaml` 建立桌面、桌腿和可运动物块；`scripts/table_retrieval/eval_policy.py` 从真实 policy/Isaac rollout 保存机器人与物块状态；`render_policy_rollout.py` 只将保存状态用 MuJoCo 渲染成视频。

## 已完成尝试

1. 原始 unified student（`v2_goal_offset_x10.pt`）直接跑桌下参考，约 40 个控制步（0.8 秒）后 root 降至 0.37 m，物块位移 0。说明墙面策略不能直接迁移到站立扶桌。记录：`artifacts/table_policy/baseline5/metrics.json`。该次记录的 `duration_s` 因评估器沿用随机化 episode 计数而错误，已修复记录时间；控制步数和位移有效。
2. 第一阶段 PPO 微调运行：`artifacts/table_policy/stage1/`。从现有 student 权重出发，使用站立到扶桌拨取的 IK 参考与真实物理桌面/物块。定期保存 checkpoint。
   - 1,048,576 环境帧、最终 checkpoint 已保存。直接动力学评估到第 30 控制步（0.58 秒）跌倒，物块没有向外移动；`artifacts/table_policy/eval_stage1_easy/metrics.json`。这一版不满足任务。
   - 发现默认训练重置会随机设置 episode 已过时间，缩短了很多 episode 可学习的时长；桌下任务已改为从零计时，后续训练生效。
3. 对原演示轨迹直接下发 IK 关节目标（仅诊断，不能作为任务成功）时，机器人第 61 步后倒，说明原运动学站姿在自由动力学中缺少稳定裕量。
4. 后移初始双脚落点后的 `stable` 参考在直接关节诊断下走完 850 步，左手平均约 24 N 接触，但左脚没抬起到目标处，物块位移 0；`artifacts/table_policy/direct_ref_stable_full/metrics.json`。右脚在中段发生大幅滑移/离地。
5. 把脚部前伸目标缩短，并把物块移至桌沿内约 7 cm 的 `easy` 版本，直接关节诊断仍不能拨动物块；`artifacts/table_policy/direct_ref_easy_full/metrics.json`。策略训练需要学会负载转移和足端轨迹，固定关节跟踪不够。
6. 第二阶段 PPO 微调运行：`artifacts/table_policy/stage2_easy_rl/`。修正 episode 计时，加入桌面手部负载、右脚支撑和物块向外位移奖励，从原 student 开始，70% episode 从站立锚点开始，其余从参考轨迹中随机初始化。结果待评估。
   - 第 50 轮从站立起点只坚持 0.8 s；第 225 轮约 1.7 s，物块仍未移动。训练日志中的 `object_retrieved` 较高，但完整起点的评估不支持成功结论；随机中途初态可能使脚与物块重叠并获益。后续必须以完整物理回放判定。
7. 原策略的训练参考初态是左手靠高位侧向支撑、左脚已经抬起，与新设计的双脚站立、低桌面扶持差距很大。抽取原训练数据第 117 段（足端向后运动 0.37 m），建立高桌侧挡板和真实物块的过渡场景。第一次动力学回放中，原策略在约 2 s 将物块向外推动 4.5 cm，右脚承重，左手有真实接触；4.24 s 后失稳，且物块没出桌下。记录：`artifacts/table_policy/eval_trained_motion_0/metrics.json`。这个场景以已经扶桌、抬左脚开始，是策略迁移的过渡任务，尚非完整站立起步任务。
8. 发现 Isaac 评估进程写出结果后仍常驻，占据多份 GPU 内存，并触发 PhysX 内核错误；已清理所有完成的评估进程，评估脚本写完结果后强制退出以释放 GPU。需在独占物理运行条件下复测。
9. 把支撑构件改为高桌侧挡板后，原策略稳定走完 7.2 s，左手持续约 55 N 接触、右脚持续约 477 N 承重；物块位于 x=0.08 时没有碰到它。把物块移到 x=0.02 后，足部真实碰撞令物块向外移动 7.18 cm，但仍未越过桌沿 x=-0.10。`artifacts/table_policy/eval_trained_motion_rail_block02/metrics.json`。降低摩擦没有进一步增加位移；提前追加 0.1 m 后缩目标反而错过物块，仅移动 0.7 mm。下一步基于这条稳定接触轨迹训练拨取奖励，尝试让足端持续推物块。
10. 发现高桌面最初只覆盖 y≥0.20，物块 y=0.09 不在桌面正下方；已增设桌面延伸板。重新回放仍稳定 7.2 s，物块向外移动 6.71 cm；`artifacts/table_policy/eval_trained_motion_under_top/metrics.json`。
11. 微调运行 `stage3_trained_motion` / `stage3b_trained_motion` 均已保存中间检查点，但运行时 `+exp=finetune` 把 YAML 中动作变化惩罚权重覆盖成 0.5，造成每段 loco 回报约 -6200，远大于拨物回报。已停止并明确在命令行设 `task.reward.loco.action_rate_l2.weight=0.0001`，用 `stage3c_trained_motion` 重训。这个教训也能解释之前部分 PPO 阶段的目标追踪不足。
12. 用原参考加上 1 s 双脚站立、先垂直抬脚再前伸的 3 s 前奏，生成 `table_retrieval_standing_prelude_liftfirst/train`。原策略稳定走完 10.2 s，先抬脚版本没有把物块向内误推。原直接关节姿态初态曾把物块推入桌内 11.4 cm，因此保留“先抬后前伸”过渡。
13. 物块高 0.16 m、x=0.08 m、质量 0.12 kg 时，原训练策略从双脚站立出发，在约 4.9 s 把物块向外拨 41.3 cm、明确越过桌沿；但约 6 s 时手离桌、身体向 -Y 滑移跌倒。`artifacts/table_policy/eval_standing_tallblock/metrics.json`。质量 0.11 kg 类似；0.09 kg 能站稳但把物块向内推；0.03 kg 仅向外移 5 cm 后跌倒。手目标预载 +0.05 m、root 侧移 +0.03 m 可站稳，但物块仅移 2.7 cm。
14. `stage3c_trained_motion` 修正奖励权重后完成 1,048,576 帧，但在站立高物块场景中仅稳定站立，把物块向内推 4.8 cm，不能直接迁移。基于原策略重新开始 `stage4_standing_tall`，训练完整站立过渡、真实高物块拨取及接触后平衡；阶段中期 checkpoint 正在按固定种子逐条回放。训练统计不作为成功证据。
15. **第一任务成功**：`stage4_standing_tall` 第 225 轮，随机种子 42 的 32 条并行 Isaac 物理回放中第 25 条满足验收。物块向外移动 35.2 cm 并完全越过桌沿；10.2 s 全程 root 最低 0.764 m，左手扶桌 8.22 s，右脚承重 10.1 s，拨出瞬间左手仍有约 114 N 接触。视频 `artifacts/table_policy/table_retrieval_policy_success.mp4` 来自对应逐帧状态；完整细节在 `SUCCESS.md`。复制检查点到 `artifacts/table_policy/table_retrieval_policy.pt` 后复跑相同种子，仍是环境 25 唯一成功，位移 35.49 cm，物理求解造成的几毫米差异不改变结论。该结果是按用户允许 cherry-pick 的 **1/32**，不代表高成功率。

## 剩余边界

当前成功轨迹不包含走近桌子并建立首次手部支撑，也没有证明拨出物块后左脚落地；这两项属于后续扩展。已保存第 225 轮 checkpoint、完整 Isaac 自由基座物理 rollout、对应指标和 MuJoCo 可视化视频，复现步骤见 `SUCCESS.md`。
