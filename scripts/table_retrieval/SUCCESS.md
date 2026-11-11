# 第一任务：扶桌脚拨物块（物理回放）

> 历史结果：本页记录较高桌面场景的 1/32 成功。当前 0.77 m 桌面及三种子独立评估结果见 [TRAINED_POLICY.md](TRAINED_POLICY.md)，详细试验记录见 [LOW_TABLE_PROGRESS.md](LOW_TABLE_PROGRESS.md)。

## 成功记录

- 策略：`artifacts/table_policy/table_retrieval_policy.pt`（`stage4_standing_tall` 第 225 轮）
- 评估：`scripts/table_retrieval/eval_batch_policy.py`，随机种子 42、32 个并行环境中的第 25 个，策略从训练分布采样关节动作。
- 源轨迹：`artifacts/table_policy/batch_stage4_225_32/candidate_00_env_025/policy_rollout.npz`；逐环境指标同目录 `metrics.json`；全部 32 个环境结果在 `batch_metrics.json`。
- 视频：`artifacts/table_policy/table_retrieval_policy_success.mp4`，1280×720、30 fps、10.2 s。视频用 MuJoCo 复现 Isaac 自由基座物理回放的机器人和物块状态；MuJoCo 不控制机器人，也不决定成功判定。
- 结果：物块从桌下 x=0.080 m 移到 x=-0.272 m，向外 0.352 m，完全越过桌沿 x=-0.100 m。机器人 root 全程最低 0.764 m、结束 0.773 m；左手与桌体接触累计 8.22 s，右脚承重累计 10.1 s。物块越界时（约 5.0 s），左手受力约 114 N。此次批量评估成功 1/32；按用户允许的 cherry-pick 方式选择该轨迹，不把它解释为通用成功率。

## 场景与边界

G1 从双脚站立、左手已靠在高桌侧挡板的姿态开始。固定的 root、手、脚笛卡尔参考先使左脚垂直抬起，再伸入桌下并回摆。全部关节目标由已训练 student policy 根据状态和目标输出，物块是自由刚体。桌面延伸板覆盖物块初始位置。当前视频没有演示从远处走到桌边或自主建立最初的手部支撑，也没有演示拨出后左脚落地；任务中的脚拨、扶桌和平衡已由真实物理轨迹验证。

## 复现

从项目根目录运行（假定 `.venv-isaac` 环境已经安装）：

```bash
MEMPATH="$PWD/dataset" .venv-isaac/bin/python scripts/table_retrieval/extract_trained_motion.py
MEMPATH="$PWD/dataset" .venv-isaac/bin/python scripts/table_retrieval/make_standing_prelude.py --dest dataset/table_retrieval_standing_prelude_liftfirst/train
OMNI_KIT_ACCEPT_EULA=Y ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 MEMPATH="$PWD/dataset" .venv-isaac/bin/python scripts/table_retrieval/eval_batch_policy.py \
  --checkpoint artifacts/table_policy/table_retrieval_policy.pt \
  --task G1/G1_table_retrieval_standing_tall_rl \
  --dataset table_retrieval_standing_prelude_liftfirst/train \
  --num-envs 32 --steps 510 --mode sample --seed 42 \
  --output artifacts/table_policy/reproduced_batch
MUJOCO_GL=egl .venv-isaac/bin/python scripts/table_retrieval/render_policy_rollout.py \
  artifacts/table_policy/reproduced_batch/candidate_00_env_025/policy_rollout.npz \
  --output artifacts/table_policy/reproduced_success.mp4 --scene trained_motion \
  --object-height 0.16 --fps 30 --width 1280 --height 720 --azimuth 45 --elevation -15
```

训练命令、此前失败尝试与原因见 `POLICY_PROGRESS.md`。
