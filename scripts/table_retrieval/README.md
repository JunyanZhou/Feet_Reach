# 扶桌脚部拨取 Demo

已完成 G1 的连续任务演示：左手扶桌 → 左脚高处绕入 → 在物体后方下降 → 用脚后侧向外拨取 → 收脚与退步 → 伸手确认可达 → 恢复站姿。

主视频约 29 秒，1920×1080、30 FPS。物体由桌下向外移动 **31.0 cm**，完整进入地面绿色区域。最后手掌接近物体顶部约 **4.0 cm**；没有演示手指闭合或抓起物体。

迁移包已解压到独立目录，使用包内源码和模型重新生成；机器人与物体轨迹与原结果逐项一致，并通过了独立目录中的 MuJoCo 渲染测试。这是同一主机上的迁移检查，Isaac 目标机运行仍待验证。

## 方法与验证边界

机器人采用脚本轨迹、全身 IK 和经过碰撞检查的固定恢复轨迹。机器人各连杆通过 MuJoCo mocap 位姿驱动；橙色物体是自由刚体，由脚的碰撞及地面摩擦推动，仿真过程中不覆盖物体位姿。

这是一个**脚本辅助的任务演示**。没有训练新的 RL policy，没有验证自主平衡、真实扶桌负载转移、执行器力矩可行性或 sim-to-real。接触力是这个运动学驱动场景的仿真读数，不构成学习型柔顺控制的实验结论。桌子固定；机器人手是原模型中的固定 rubber-hand 网格。

## 先看这些文件

以下路径均相对于解压后的 `table_retrieval_demo/` 根目录：

| 文件 | 用途 |
| --- | --- |
| `artifacts/table_retrieval/overview.mp4` | 完整 1080p 视频，包含阶段与位移说明 |
| `artifacts/table_retrieval/overview_clean.mp4` | 同一过程的无字幕视频 |
| `artifacts/table_retrieval/detail.mp4` | 5–14 秒脚部操作特写 |
| `artifacts/table_retrieval/detail_clean.mp4` | 无字幕特写 |
| `artifacts/table_retrieval/storyboard.jpg` | 任务关键帧总览 |
| `artifacts/table_retrieval/contact_report.png` | 物体位移、接触力与末端高度曲线 |
| `artifacts/table_retrieval/table_retrieval.usdc` | 完整、自包含的 USD 动画，包含模型、场景、灯光和相机 |
| `artifacts/table_retrieval/trajectory_portable.npz` | 按名称保存的关节、root、物体及连杆轨迹 |
| `artifacts/table_retrieval/scene_manifest.json` | 场景尺寸、坐标约定和关键阶段 |
| `artifacts/table_retrieval/validation.json` | 标称任务验收结果 |
| `artifacts/table_retrieval/robustness_summary.json` | 27 组局部扰动的逐组结果 |
| `artifacts/table_retrieval/PROGRESS.md` | 已保存的进度和方法调整 |
| `SHA256SUMS.json` | 迁移包内文件校验和 |

## 评估结果

- 标称任务成功：物体完整离开桌下，完整进入目标区域，并保持直立。
- 27/27 组局部扰动成功：初始 X/Y 偏移各取 −1.5、0、+1.5 cm；物体/地面摩擦系数取 0.35、0.55、0.8。使用同一条机器人轨迹。
- 关闭主动脚的碰撞后，物体水平位移为 **0**，验证了拨取依赖实际接触。
- 标称脚物法向接触峰值：**6.07 N**；脚物碰撞代理最大穿透约 **2.27 mm**。
- 扶桌阶段手掌目标最大位置偏差：**0.79 mm**。
- 全部关节在模型限位内；保存轨迹的最大关节速度为 **2.40 rad/s**。
- 采用原 G1 碰撞代理、以 20 Hz 进行全身几何复核；非预期代理最大穿透约 **2.44 mm**。原模型相邻腕部代理的微小重叠单独记录在 `geometry_audit.json`。
- USD 的 140 项网格世界包围盒检查通过；没有外部资产依赖或缺失引用。

这些是固定场景脚本的局部测试，不表示通用控制器的成功率。完整结果和验收条件保存在 JSON 文件中。

## 在另一台电脑上用 Isaac 渲染

**本机未启动 Isaac Sim/Isaac Lab。** USD 的文件完整性、动画变换和网格一致性已经本地验证；下面的 Isaac 运行适配器还需在目标电脑验证。

最省事的方式：在 Isaac Sim 中打开 `artifacts/table_retrieval/table_retrieval.usdc`，选择 `/World/Cameras/Overview` 相机，播放时间轴。时间范围为 0–1740，时间码每秒 60。这个 USD 是纯视觉动画，不含 PhysX articulation 或刚体仿真设置，因此重渲染不依赖重新训练或重新求解接触。

自动输出 PNG 帧，在已安装的 Isaac Lab 根目录运行：

```bash
./isaaclab.sh -p /绝对路径/table_retrieval_demo/scripts/table_retrieval/render_isaac.py \
  --headless --enable_cameras \
  --output /绝对路径/isaac_frames
```

Windows 使用对应的 `isaaclab.bat -p ...`。也可用 Isaac Sim 自带 Python 直接运行同一脚本；脚本在没有 `isaaclab` 时使用 `isaacsim.SimulationApp`。

可选参数包括 `--camera /World/Cameras/Detail`、`--start 5 --end 14`、`--width 1920 --height 1080 --fps 30`。输出为 RGB PNG 及 `render_manifest.json`。在装有 imageio/FFmpeg 的普通 Python 环境中编码：

```bash
python scripts/table_retrieval/encode_video.py /绝对路径/isaac_frames \
  --fps 30 --output isaac_render.mp4
```

适配器依据 [Isaac Lab AppLauncher 文档](https://isaac-sim.github.io/IsaacLab/main/source/how-to/launch_app.html) 和 [Isaac Sim Replicator 文档](https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_sdg_workflows.html) 编写。

## 使用原生 Isaac G1 资产

也可只读取 `trajectory_portable.npz`，将轨迹写入自己使用的 G1 articulation。必须按 `joint_names` 建立映射，不要假定 Isaac 与 MuJoCo 的关节数组顺序一致。

- 单位：米、弧度、秒；世界 Z 向上，初始机器人前方为 +X，左侧为 +Y。
- 四元数顺序：**wxyz**。
- 采样：60 Hz、1741 帧。
- `root_position_world_m` / `root_quaternion_wxyz`：机器人 pelvis 的世界位姿。
- `joint_position_rad` / `joint_velocity_rad_s`：29 个关节，顺序见 `joint_names`。
- `object_position_world_m` / `object_quaternion_wxyz`：物体的接触仿真结果。
- `body_position_world_m` / `body_quaternion_wxyz`：按 `body_names` 保存的全部机器人连杆世界位姿。

模型零位、关节轴和网格版本必须与本包匹配。开启自由动力学后，直接使用这些姿态目标不保证机器人能够跟踪；要复现此视频，应采用运动学重放。

## 本地 MuJoCo 重现

已验证环境：Linux、Python 3.12、MuJoCo 3.3.7、EGL 无窗口渲染。现有环境为项目根目录的 `.venv-table-demo`；迁移包不复制虚拟环境。

```bash
python3 -m venv .venv-table-demo
.venv-table-demo/bin/python -m pip install -r scripts/table_retrieval/requirements.txt

# 直接渲染已保存轨迹，不重新求解
MUJOCO_GL=egl .venv-table-demo/bin/python scripts/table_retrieval/replay_mujoco.py \
  --view overview --width 1920 --height 1080 --fps 30

# 完整重新生成、评估、导出
MUJOCO_GL=egl .venv-table-demo/bin/python scripts/table_retrieval/demo.py
MUJOCO_GL=egl .venv-table-demo/bin/python scripts/table_retrieval/export_usd.py
MUJOCO_GL=egl .venv-table-demo/bin/python scripts/table_retrieval/evaluate.py
MUJOCO_GL=egl .venv-table-demo/bin/python scripts/table_retrieval/validate_usd.py
```

`requirements-lock.txt` 保存本机精确版本。Windows 普通 MuJoCo 重放使用对应 Python 路径，并省略 `MUJOCO_GL=egl`。

19.5 秒后的伸手和恢复动作使用 `validated_recovery.npz`，固定经过检查的 IK 解分支，避免微小数值差异导致肘部切换到另一种构型。`--resolve-recovery` 可以重新求解这一段，但属于编辑/开发选项；重新求解后必须再运行评估，不能假定新轨迹安全或无穿模。

## 资产来源

G1 模型和 35 个 STL 网格复用仓库已有的 `deploy/mujoco_whole_body/g1/`。原始资源归属 Unitree，来源链为 `luoxinyuan/ceer_deploy`；原许可标注 HansZ8、CC BY 4.0。许可全文随包保存在 `deploy/mujoco_whole_body/CEER_DEPLOY_LICENSE.txt`，许可链接为 https://creativecommons.org/licenses/by/4.0/ 。原模型和 STL 未改写；本 demo 新增桌子、物体、灯光、碰撞代理、运动与渲染逻辑。
