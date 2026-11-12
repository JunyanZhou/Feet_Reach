# 推动大箱子：自由基座 MuJoCo 结果

## 场景与方法

G1 是自由基座动力学机器人，推动有自由关节的箱体。箱体尺寸约 0.48 × 0.72 × 0.76 m、质量约 17.0 kg，质心比几何中心低 6.3 cm；箱底滑动摩擦系数约 0.174。机器人脚与地面、双手与箱体、其他身体部位与箱体均使用真实碰撞。已有全身 student 提供平衡关节目标；任务控制器给双手笛卡尔目标、腰部前倾目标和一次左脚前迈目标，实际根部和落脚点均由物理仿真决定。没有 mocap 机器人、焊接支撑、箱体位姿脚本或 IsaacLab 渲染。

主配置：[`freebase_all_contact_candidate112.json`](freebase_all_contact_candidate112.json)，物理步长 2 ms、策略控制 50 Hz、每回合 5.5 s。箱体反作用力在前倾和迈步期间支撑机器人。

## 验收条件

- 箱体最终沿前方平移至少 15 cm，最大倾斜小于 0.25 rad，质心高度偏差小于 5 cm。
- 机器人根部未跌破 40 cm；双掌同时接触箱体累计至少 0.30 s，箱体在双掌同时接触的采样期间向前移动至少 4 cm。
- 对照从有箱轨迹记录每 20 ms 的全部 29 个关节目标，在无箱模型中原样重放；无箱机器人应跌倒。

## 标称视频与接触审计

标称有箱轨迹推动箱体 **28.31 cm**，箱体最大倾斜 **0.018 rad**，机器人根部最低 **75.7 cm**；双掌同时接触 **0.982 s**，这些接触期间推动 **8.81 cm**。左脚前迈参考为 10.7 cm，实际左脚落点与全身 student 共同变化。没有箱体、执行完全相同的 **275 × 29** 个关节目标时，机器人在 **3.666 s** 跌倒；两条实际执行目标数组的最大差值为 **0**。

MuJoCo 接触力审计：左右手掌对箱体的累计法向冲量分别为 **19.62、22.95 N·s**；左髋两个碰撞几何合计 **14.54 N·s**。双掌承担箱体非地面法向冲量的约 **74.5%**，因此这是手掌主导、髋部也短暂接触的推箱动作。最大箱体与机器人接触穿透约 **1.6 mm**，没有禁用全身碰撞导致的穿模。手掌瞬时接触峰值约 **77.6 N**。审计明细在 `artifacts/push_box_support/freebase_all_contact_candidate112/contact_audit.json`。

视频：[`有箱／无箱并排对照`](../../artifacts/push_box_support/freebase_all_contact_candidate112_same_targets.mp4)、[`有箱独立视频`](../../artifacts/push_box_support/freebase_all_contact_candidate112_with_box.mp4)、[`无箱跌倒独立视频`](../../artifacts/push_box_support/freebase_all_contact_candidate112_no_box_fall.mp4)。并排视频使用 MuJoCo EGL 侧视渲染。5.5 s 后段机器人会向侧方移动，主要箱体位移发生在 2.5–4.0 s；不应解读为整个回合持续匀速推箱。

## 独立随机复位

初始扰动：机器人根部水平位置各 ±1 cm、29 关节高斯标准差 0.01 rad、箱体水平位置各 ±1.5 cm。成功定义与上文完全相同，不因随机结果调整阈值。

| 评估批次 | 有箱成功 | 无箱同目标跌倒 |
| --- | ---: | ---: |
| 种子 20000–20031，5.5 s | 19/32（59.4%） | 32/32 |
| 独立种子 21000–21063，5.5 s | **25/64（39.1%）** | **64/64** |
| 独立种子 22000–22063，4.5 s | 30/64（46.9%） | 64/64 |

主结果采用完整 5.5 s、第二批 64 组的 **25/64**，达到预期约 30%–40% 的量级。其箱体平移均值为 53.3 cm、中位数 55.2 cm。随机成功率只适用于上述初始扰动和当前箱体／地面摩擦，不代表真实机器人成功率。数据分别在 `artifacts/push_box_support/freebase_all_contact_candidate112_pairs32/`、`freebase_all_contact_candidate112_pairs64_holdout/`、`freebase_all_contact_candidate112_pairs64_4p5/`。

## 试错记录和局限

最早的 `make_demo.py` 是 mocap 机器人，只能用于几何参考，不能证明自由基座平衡。初版自由基座方案允许躯干和髋部顶箱，标称箱体移动 35 cm，但非手掌冲量超过手掌冲量；未作为最终方案。尝试让箱体只与双掌碰撞后虽能标称移动 66 cm，但其他身体部位会穿入箱体最多约 29 cm，已废弃。随后恢复全身真实碰撞，增加真实的落脚点目标，在多种箱体参数和推力时间安排中搜索，并对手掌受力比例和穿透量逐一审计，得到当前方案。

当前方案使用已有平衡 student 与手工任务控制器，并非新训练完成的推箱 student。箱底摩擦较低，任务是滑动箱体；更高摩擦、更重箱体、地面变化和真机安全性尚未验证。第二区别实验“脚部固定与双手拉抽屉”仍只有标称成功，没有达到约 30%–40% 的完整任务随机复位成功率，见 [`RESULTS_MUJOCO_DRAWER.md`](../foot_pin_bimanual/RESULTS_MUJOCO_DRAWER.md)。

## 复现

在仓库根目录运行：

```bash
PYTHONPATH=. .venv-isaac/bin/python scripts/push_box_support/mujoco_freebase_push.py \
  --params scripts/push_box_support/freebase_all_contact_candidate112_full_fall.json \
  --output artifacts/push_box_support/reproduce_candidate112 --duration 5.5
PYTHONPATH=. .venv-isaac/bin/python scripts/push_box_support/render_freebase_push.py \
  --folder artifacts/push_box_support/reproduce_candidate112 \
  --params scripts/push_box_support/freebase_all_contact_candidate112_full_fall.json \
  --output artifacts/push_box_support/reproduce_candidate112.mp4 --azimuth 90
PYTHONPATH=. .venv-isaac/bin/python scripts/push_box_support/evaluate_freebase_push.py \
  --params scripts/push_box_support/freebase_all_contact_candidate112.json \
  --output artifacts/push_box_support/reproduce_pairs64 --episodes 64 \
  --seed-start 21000 --duration 5.5
```
