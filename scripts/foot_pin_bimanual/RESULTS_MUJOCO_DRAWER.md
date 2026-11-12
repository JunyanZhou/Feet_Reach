# 脚部固定与双手拉抽屉：MuJoCo 结果

## 任务和验收

机器人面对一个有自由基座的轻型工具柜，左脚压住与柜体相连的挡架，左右手分别接触抽屉的两个把手，把抽屉沿物理滑动关节拉出。柜体、抽屉、机器人根部均由 MuJoCo 动力学推进；任务控制器只给机器人关节位置目标。这个原型对应双手抽出工具箱或设备柜模块、用脚限制轻柜滑走的场景。

主要验收要求：抽屉终态拉出至少 15 cm、柜体最大漂移小于 5 cm、机器人根部始终高于 55 cm，左脚和双手同时接触至少 0.15 s，且抽屉在这些接触时确实向外移动。仿真步长 2 ms，控制频率 50 Hz。

## 标称成功轨迹与因果对照

配置为 [`drawer_pull_stirrup_soft_v2.json`](drawer_pull_stirrup_soft_v2.json)。有挡架时，抽屉拉出 **19.08 cm**，柜体最大漂移 **3.16 cm**，三处同时接触 **0.382 s**，其中抽屉向外移动 **12.33 cm**。记录的左脚接触峰值约 **628 N**，偏高，尚不适合直接用于真机。

对照把有挡架轨迹的全部 **230 × 29** 个关节目标原样送给无挡架模型，两个输入数组最大差值为 **0**。有挡架回放仍成功；移除挡架后机器人在 **1.352 s** 跌倒，抽屉几乎未打开。完整无挡架视频继续渲染到 4.6 s，根部最低约 8.4 cm。视频：[`drawer_pull_stirrup_soft_v2_replay_comparison.mp4`](../../artifacts/foot_pin_bimanual/drawer_pull_stirrup_soft_v2_replay_comparison.mp4)。

也试过两种脚踏板：高摩擦脚踏板的标称轨迹拉出 19.14 cm，脚部接触峰值约 115 N；低摩擦脚踏板标称轨迹拉出 19.00 cm，无板时柜体被拖动约 23.3 cm。相关配置与视频保存在本目录和 `artifacts/foot_pin_bimanual/`。这些方案的随机复位表现没有改善，因此没有作为最终成功率结果。

## 鲁棒性和训练结论

匹配随机复位扰动为：机器人根部水平位置各 ±1 cm、29 个关节高斯标准差 0.01 rad、柜体水平位置各 ±1.5 cm、柜体和抽屉摩擦各 ±10%。在 64 对随机复位上，最好的 19 cm 挡架标称方案 **0/64** 达到完整任务标准；老版 14.3 cm 方案按其 12 cm 标准为 **1/64**。不能把单条标称成功轨迹当作鲁棒策略。

实施了 8 维双手/左脚笛卡尔残差 PPO，并分别训练完整任务、已建立接触后的拉出阶段、从初始状态到把手附近的接近阶段。接近阶段模型在 32 个未见过的随机复位上 **10/32** 达到阶段目标，零残差基线为 **9/32**；将该模型用于完整任务只有 **1/32** 成功。已建立接触状态启动的阶段训练也没有超过相同状态下的零残差基线。因此第二任务没有达到期望的约 30%–40% 完整成功率。

评估记录：`artifacts/foot_pin_bimanual/drawer_pull_stirrup_soft_v2_pairs64/summary.json`、`artifacts/foot_pin_bimanual/stirrup_v2_approach_ppo_resume10/eval_approach32/summary.json`、`artifacts/foot_pin_bimanual/stirrup_v2_approach_ppo_resume10/eval_full32/summary.json`。已有模型和检查点均在 `artifacts/foot_pin_bimanual/`。本任务的有效展示是可复现的单条物理轨迹与无支撑跌倒对照，训练出的 student 尚未达到鲁棒验收。

## 复现

在仓库根目录运行：

```bash
PYTHONPATH=. .venv-isaac/bin/python scripts/foot_pin_bimanual/mujoco_task.py \
  --params-file scripts/foot_pin_bimanual/drawer_pull_stirrup_soft_v2.json \
  --duration 4.6 --record --output artifacts/foot_pin_bimanual/reproduce_v2
```

视频由 `render_mujoco_task.py` 用 MuJoCo EGL 渲染；本任务没有使用 IsaacLab 渲染。
