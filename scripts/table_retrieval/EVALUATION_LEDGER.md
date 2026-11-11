# 扶桌脚拨物块：全部批量评估台账

由 `generate_evaluation_ledger.py` 从本仓库保存的 `batch_metrics.json` 生成。
每一行是一次独立批量评估，不是 CEM 训练批次。场景列依次为物块中心
x/y/高度，单位 m；`—` 表示早期日志没有保存该字段。位移为 root
始终高于 0.50 m 的轨迹中的平均/最大物块净外移，单位 cm。

**严格完整离桌**统一重算：末帧按物块姿态得到的最靠桌内角
x<0.280 m、root 最低高度>0.50 m、左手向上与右脚承重重叠>5 s、
受支撑时脚靠近物块且物块外移累计>0.08 s。只有保存了三维旋转后
后角的日志才可计算。早期 `task_success` 用固定半宽或不同条件，
放在“旧成功”列，**不能**当作严格完成数。中心越沿沿用各批次
日志中的 `front_edge_crossing_success`；早期缺失时显示 `—`。

| 批次原始记录 | n | seed | 场景 x/y/h | 稳定 | 中心越沿 | 严格完整 | 旧成功 | 稳定轨迹位移均值/最大 cm | 结果与限制 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| [`batch_stage4_225_32`](../../artifacts/table_policy/batch_stage4_225_32/batch_metrics.json) | 32 | 42 | 0.080/—/— | 13 | — | — | 1 | 4.83/35.20 | 旧几何口径；严格完成不可判定 |
| [`block10_baseline_seed53`](../../artifacts/table_policy/block10_baseline_seed53/batch_metrics.json) | 32 | 53 | 0.315/0.080/0.100 | 30 | 0 | 0 | — | 1.58/3.34 | 稳定但未满足中心越沿或接触条件 |
| [`block5x_seed53`](../../artifacts/table_policy/block5x_seed53/batch_metrics.json) | 32 | 53 | 0.310/0.080/0.120 | 28 | 14 | 0 | — | 4.89/9.16 | 中心越沿，但旋转后仍有角留在桌内 |
| [`contact_feedback_baseline_seed61`](../../artifacts/table_policy/contact_feedback_baseline_seed61/batch_metrics.json) | 32 | 61 | 0.345/0.080/0.120 | 32 | 4 | — | 0 | 3.93/7.01 | 旧几何口径；严格完成不可判定 |
| [`contact_feedback_cem_v1_eval_gen0`](../../artifacts/table_policy/contact_feedback_cem_v1_eval_gen0/batch_metrics.json) | 32 | 61 | 0.345/0.080/0.120 | 32 | 0 | — | 0 | 0.74/3.62 | 旧几何口径；严格完成不可判定 |
| [`contact_feedback_cem_v2_eval_gen1_seed63`](../../artifacts/table_policy/contact_feedback_cem_v2_eval_gen1_seed63/batch_metrics.json) | 32 | 63 | 0.345/0.080/0.120 | 0 | 0 | — | 0 | — | 全部失稳；倒地后物块运动不计成功 |
| [`contact_feedback_cem_v2_eval_gen1_supported_seed63`](../../artifacts/table_policy/contact_feedback_cem_v2_eval_gen1_supported_seed63/batch_metrics.json) | 32 | 63 | 0.345/0.080/0.120 | 28 | 3 | — | 0 | 3.08/6.99 | 旧几何口径；严格完成不可判定 |
| [`contact_feedback_control_eval`](../../artifacts/table_policy/contact_feedback_control_eval/batch_metrics.json) | 16 | 53 | 0.345/0.080/0.120 | 16 | 2 | — | 0 | 4.31/6.99 | 旧几何口径；严格完成不可判定 |
| [`contact_feedback_zero_eval`](../../artifacts/table_policy/contact_feedback_zero_eval/batch_metrics.json) | 16 | 53 | 0.345/0.080/0.120 | 16 | 2 | — | 0 | 4.31/6.99 | 旧几何口径；严格完成不可判定 |
| [`cube6_baseline_seed53`](../../artifacts/table_policy/cube6_baseline_seed53/batch_metrics.json) | 32 | 53 | 0.315/0.080/0.060 | 27 | 0 | 0 | — | 0.00/0.00 | 稳定但未满足中心越沿或接触条件 |
| [`final_policy_x320_seed53`](../../artifacts/table_policy/final_policy_x320_seed53/batch_metrics.json) | 32 | 53 | 0.320/0.095/0.120 | 31 | 28 | 0 | — | 8.63/10.72 | 中心越沿，但旋转后仍有角留在桌内 |
| [`final_policy_x330_h100_seed53`](../../artifacts/table_policy/final_policy_x330_h100_seed53/batch_metrics.json) | 32 | 53 | 0.330/0.095/0.100 | 30 | 1 | 0 | — | 2.63/5.33 | 中心越沿，但旋转后仍有角留在桌内 |
| [`final_policy_x330_seed53`](../../artifacts/table_policy/final_policy_x330_seed53/batch_metrics.json) | 32 | 53 | 0.330/0.095/0.120 | 30 | 21 | 0 | — | 7.03/10.28 | 中心越沿，但旋转后仍有角留在桌内 |
| [`foot_follow_sweep_seed58`](../../artifacts/table_policy/foot_follow_sweep_seed58/batch_metrics.json) | 64 | 58 | 0.315/0.080/0.120 | 58 | 15 | 0 | — | 2.97/10.21 | 中心越沿，但旋转后仍有角留在桌内 |
| [`late_two_phase_cem_y095_eval_gen2_seed53`](../../artifacts/table_policy/late_two_phase_cem_y095_eval_gen2_seed53/batch_metrics.json) | 32 | 53 | 0.315/0.095/0.120 | 27 | 18 | 2 | — | 5.53/10.97 | 有完整离桌；须结合场景和独立种子判断泛化 |
| [`late_two_phase_cem_y095_eval_success07_seed53`](../../artifacts/table_policy/late_two_phase_cem_y095_eval_success07_seed53/batch_metrics.json) | 32 | 53 | 0.315/0.095/0.120 | 28 | 20 | 5 | — | 6.81/10.95 | 有完整离桌；须结合场景和独立种子判断泛化 |
| [`late_two_phase_cem_y095_eval_success07_seed55`](../../artifacts/table_policy/late_two_phase_cem_y095_eval_success07_seed55/batch_metrics.json) | 32 | 55 | 0.315/0.095/0.120 | 31 | 20 | 3 | — | 5.96/10.70 | 有完整离桌；须结合场景和独立种子判断泛化 |
| [`late_two_phase_cem_y095_eval_success07_seed57`](../../artifacts/table_policy/late_two_phase_cem_y095_eval_success07_seed57/batch_metrics.json) | 32 | 57 | 0.315/0.095/0.120 | 30 | 23 | 7 | — | 7.81/10.83 | 有完整离桌；须结合场景和独立种子判断泛化 |
| [`late_two_phase_cem_y095_no_table_seed53`](../../artifacts/table_policy/late_two_phase_cem_y095_no_table_seed53/batch_metrics.json) | 16 | 53 | 0.315/0.095/0.120 | 0 | 0 | 0 | — | — | 全部失稳；倒地后物块运动不计成功 |
| [`reproduced_batch`](../../artifacts/table_policy/reproduced_batch/batch_metrics.json) | 32 | 42 | 0.080/—/— | 15 | — | — | 1 | 3.03/35.49 | 旧几何口径；严格完成不可判定 |
| [`reproduced_final_policy_seed57`](../../artifacts/table_policy/reproduced_final_policy_seed57/batch_metrics.json) | 32 | 57 | 0.315/0.095/0.120 | 30 | 23 | 7 | — | 7.81/10.83 | 有完整离桌；须结合场景和独立种子判断泛化 |
| [`reproduced_final_policy_seed57_verified`](../../artifacts/table_policy/reproduced_final_policy_seed57_verified/batch_metrics.json) | 32 | 57 | 0.315/0.095/0.120 | 30 | 23 | 7 | — | 7.81/10.83 | 有完整离桌；须结合场景和独立种子判断泛化 |
| [`reproduced_y095_seed57`](../../artifacts/table_policy/reproduced_y095_seed57/batch_metrics.json) | 32 | 57 | 0.315/0.095/0.120 | 30 | 22 | 4 | — | 6.50/11.25 | 有完整离桌；须结合场景和独立种子判断泛化 |
| [`slim_block_seed53`](../../artifacts/table_policy/slim_block_seed53/batch_metrics.json) | 32 | 53 | 0.305/0.080/0.120 | 30 | 24 | 0 | — | 6.36/8.11 | 中心越沿，但旋转后仍有角留在桌内 |
| [`small_block_baseline_seed53`](../../artifacts/table_policy/small_block_baseline_seed53/batch_metrics.json) | 32 | 53 | 0.315/0.080/0.120 | 28 | 17 | — | 9 | 4.79/10.01 | 旧几何口径；严格完成不可判定 |
| [`small_block_baseline_seed55`](../../artifacts/table_policy/small_block_baseline_seed55/batch_metrics.json) | 32 | 55 | 0.315/0.080/0.120 | 29 | 16 | — | 5 | 3.82/10.00 | 旧几何口径；严格完成不可判定 |
| [`small_block_cem_final_eval_seed53`](../../artifacts/table_policy/small_block_cem_final_eval_seed53/batch_metrics.json) | 32 | 53 | 0.315/0.080/0.120 | 28 | 22 | 0 | — | 6.76/10.22 | 中心越沿，但旋转后仍有角留在桌内 |
| [`small_block_cem_final_eval_seed55`](../../artifacts/table_policy/small_block_cem_final_eval_seed55/batch_metrics.json) | 32 | 55 | 0.315/0.080/0.120 | 29 | 20 | 0 | — | 5.58/9.96 | 中心越沿，但旋转后仍有角留在桌内 |
| [`small_block_cem_final_no_table_seed55`](../../artifacts/table_policy/small_block_cem_final_no_table_seed55/batch_metrics.json) | 16 | 55 | 0.315/0.080/0.120 | 0 | 0 | 0 | — | — | 全部失稳；倒地后物块运动不计成功 |
| [`small_block_distilled_dagger1_eval1200_seed73`](../../artifacts/table_policy/small_block_distilled_dagger1_eval1200_seed73/batch_metrics.json) | 32 | 73 | 0.315/0.080/0.120 | 32 | 0 | 0 | — | -0.02/0.01 | 稳定但未满足中心越沿或接触条件 |
| [`small_block_distilled_dagger_weighted_eval_seed73`](../../artifacts/table_policy/small_block_distilled_dagger_weighted_eval_seed73/batch_metrics.json) | 32 | 73 | 0.315/0.080/0.120 | 30 | 0 | 0 | — | -0.01/0.03 | 稳定但未满足中心越沿或接触条件 |
| [`small_block_distilled_eval1200_seed73`](../../artifacts/table_policy/small_block_distilled_eval1200_seed73/batch_metrics.json) | 32 | 73 | 0.315/0.080/0.120 | 13 | 0 | 0 | — | -0.19/-0.07 | 稳定但未满足中心越沿或接触条件 |
| [`small_block_distilled_eval300_seed73`](../../artifacts/table_policy/small_block_distilled_eval300_seed73/batch_metrics.json) | 32 | 73 | 0.315/0.080/0.120 | 27 | 0 | 0 | — | -0.62/-0.30 | 稳定但未满足中心越沿或接触条件 |
| [`small_block_distilled_eval600_seed73`](../../artifacts/table_policy/small_block_distilled_eval600_seed73/batch_metrics.json) | 32 | 73 | 0.315/0.080/0.120 | 15 | 0 | 0 | — | -0.29/-0.14 | 稳定但未满足中心越沿或接触条件 |
| [`small_block_foot_follow_seed55`](../../artifacts/table_policy/small_block_foot_follow_seed55/batch_metrics.json) | 32 | 55 | 0.315/0.080/0.120 | 28 | 19 | 0 | — | 5.83/9.98 | 中心越沿，但旋转后仍有角留在桌内 |
| [`small_block_light_seed53`](../../artifacts/table_policy/small_block_light_seed53/batch_metrics.json) | 32 | 53 | 0.315/0.080/0.120 | 29 | 13 | 0 | — | 3.61/10.26 | 中心越沿，但旋转后仍有角留在桌内 |
| [`small_block_long11s_seed53`](../../artifacts/table_policy/small_block_long11s_seed53/batch_metrics.json) | 32 | 53 | 0.315/0.080/0.120 | 26 | 17 | 0 | — | 5.09/10.01 | 中心越沿，但旋转后仍有角留在桌内 |
| [`small_block_no_table_seed53`](../../artifacts/table_policy/small_block_no_table_seed53/batch_metrics.json) | 16 | 53 | 0.315/0.080/0.120 | 0 | 0 | — | 0 | — | 全部失稳；倒地后物块运动不计成功 |
| [`small_block_y0925_seed53`](../../artifacts/table_policy/small_block_y0925_seed53/batch_metrics.json) | 32 | 53 | 0.315/0.092/0.120 | 29 | 20 | 4 | — | 6.24/11.23 | 有完整离桌；须结合场景和独立种子判断泛化 |
| [`small_block_y0925_seed55`](../../artifacts/table_policy/small_block_y0925_seed55/batch_metrics.json) | 32 | 55 | 0.315/0.092/0.120 | 30 | 26 | 1 | — | 7.88/11.05 | 有完整离桌；须结合场景和独立种子判断泛化 |
| [`small_block_y095_no_table_seed53`](../../artifacts/table_policy/small_block_y095_no_table_seed53/batch_metrics.json) | 16 | 53 | 0.315/0.095/0.120 | 0 | 0 | 0 | — | — | 全部失稳；倒地后物块运动不计成功 |
| [`small_block_y095_seed53`](../../artifacts/table_policy/small_block_y095_seed53/batch_metrics.json) | 32 | 53 | 0.315/0.095/0.120 | 28 | 18 | 3 | — | 5.60/10.87 | 有完整离桌；须结合场景和独立种子判断泛化 |
| [`small_block_y095_seed55`](../../artifacts/table_policy/small_block_y095_seed55/batch_metrics.json) | 32 | 55 | 0.315/0.095/0.120 | 31 | 18 | 2 | — | 5.18/10.99 | 有完整离桌；须结合场景和独立种子判断泛化 |
| [`small_block_y095_seed57`](../../artifacts/table_policy/small_block_y095_seed57/batch_metrics.json) | 32 | 57 | 0.315/0.095/0.120 | 30 | 22 | 4 | — | 6.50/11.25 | 有完整离桌；须结合场景和独立种子判断泛化 |
| [`small_block_y09_seed53`](../../artifacts/table_policy/small_block_y09_seed53/batch_metrics.json) | 32 | 53 | 0.315/0.090/0.120 | 29 | 22 | 0 | — | 7.09/10.54 | 中心越沿，但旋转后仍有角留在桌内 |
| [`small_block_y10_seed53`](../../artifacts/table_policy/small_block_y10_seed53/batch_metrics.json) | 32 | 53 | 0.315/0.100/0.120 | 32 | 20 | 1 | — | 5.17/10.69 | 有完整离桌；须结合场景和独立种子判断泛化 |
| [`strict_backedge_cem_y095_eval_gen0_seed53`](../../artifacts/table_policy/strict_backedge_cem_y095_eval_gen0_seed53/batch_metrics.json) | 32 | 53 | 0.315/0.095/0.120 | 28 | 16 | 1 | — | 4.45/11.73 | 有完整离桌；须结合场景和独立种子判断泛化 |
| [`strict_backedge_cem_y095_eval_gen1_seed55`](../../artifacts/table_policy/strict_backedge_cem_y095_eval_gen1_seed55/batch_metrics.json) | 32 | 55 | 0.315/0.095/0.120 | 31 | 18 | 0 | — | 5.45/10.76 | 中心越沿，但旋转后仍有角留在桌内 |
| [`strict_backedge_cem_y095_eval_gen3_seed53`](../../artifacts/table_policy/strict_backedge_cem_y095_eval_gen3_seed53/batch_metrics.json) | 32 | 53 | 0.315/0.095/0.120 | 28 | 18 | 0 | — | 6.35/10.73 | 中心越沿，但旋转后仍有角留在桌内 |
| [`strict_backedge_cem_y095_eval_gen8_seed53`](../../artifacts/table_policy/strict_backedge_cem_y095_eval_gen8_seed53/batch_metrics.json) | 32 | 53 | 0.315/0.095/0.120 | 28 | 18 | 1 | — | 6.05/10.73 | 有完整离桌；须结合场景和独立种子判断泛化 |
| [`supported_baseline`](../../artifacts/table_policy/supported_baseline/batch_metrics.json) | 16 | 17 | 0.350/—/— | 0 | — | — | 0 | — | 全部失稳；倒地后物块运动不计成功 |
| [`supported_bc_expert_eval`](../../artifacts/table_policy/supported_bc_expert_eval/batch_metrics.json) | 16 | 19 | 0.350/—/— | 16 | — | — | 0 | 0.00/0.00 | 旧几何口径；严格完成不可判定 |
| [`supported_block12_y06_eval`](../../artifacts/table_policy/supported_block12_y06_eval/batch_metrics.json) | 16 | 31 | 0.350/—/— | 16 | — | — | 0 | -8.95/-4.03 | 旧几何口径；严格完成不可判定 |
| [`supported_cem_center_x340_eval`](../../artifacts/table_policy/supported_cem_center_x340_eval/batch_metrics.json) | 32 | 53 | 0.340/0.080/0.120 | 31 | 4 | — | 0 | 4.53/7.03 | 旧几何口径；严格完成不可判定 |
| [`supported_cem_center_x345_eval_gen0`](../../artifacts/table_policy/supported_cem_center_x345_eval_gen0/batch_metrics.json) | 32 | 53 | 0.345/0.080/0.120 | 30 | 6 | — | 0 | 4.30/7.19 | 旧几何口径；严格完成不可判定 |
| [`supported_cem_center_x345_eval_seed55`](../../artifacts/table_policy/supported_cem_center_x345_eval_seed55/batch_metrics.json) | 16 | 55 | 0.345/0.080/0.120 | 15 | 1 | — | 0 | 3.28/6.51 | 旧几何口径；严格完成不可判定 |
| [`supported_cem_center_y085_eval`](../../artifacts/table_policy/supported_cem_center_y085_eval/batch_metrics.json) | 16 | 53 | 0.345/0.085/0.120 | 15 | 4 | — | 0 | 4.69/7.37 | 旧几何口径；严格完成不可判定 |
| [`supported_cem_final_align_eval_gen0`](../../artifacts/table_policy/supported_cem_final_align_eval_gen0/batch_metrics.json) | 16 | 49 | —/0.080/0.120 | 14 | — | — | 0 | 2.66/4.88 | 旧几何口径；严格完成不可判定 |
| [`supported_cem_final_pull_eval_gen0`](../../artifacts/table_policy/supported_cem_final_pull_eval_gen0/batch_metrics.json) | 16 | 48 | —/0.080/0.120 | 16 | — | — | 0 | 2.94/6.18 | 旧几何口径；严格完成不可判定 |
| [`supported_cem_final_pull_y10_eval`](../../artifacts/table_policy/supported_cem_final_pull_y10_eval/batch_metrics.json) | 16 | 48 | —/0.100/0.120 | 16 | — | — | 0 | 1.67/4.20 | 旧几何口径；严格完成不可判定 |
| [`supported_cem_no_table_eval`](../../artifacts/table_policy/supported_cem_no_table_eval/batch_metrics.json) | 8 | 48 | —/0.080/0.120 | 0 | — | — | 0 | — | 全部失稳；倒地后物块运动不计成功 |
| [`supported_cem_pull_friction06_eval`](../../artifacts/table_policy/supported_cem_pull_friction06_eval/batch_metrics.json) | 16 | 45 | —/0.080/0.120 | 15 | — | — | 0 | 1.58/4.77 | 旧几何口径；严格完成不可判定 |
| [`supported_cem_pull_x33_eval`](../../artifacts/table_policy/supported_cem_pull_x33_eval/batch_metrics.json) | 16 | 46 | 0.330/0.080/0.120 | 15 | — | — | 0 | 1.16/3.08 | 旧几何口径；严格完成不可判定 |
| [`supported_cem_pull_y08_eval_gen7`](../../artifacts/table_policy/supported_cem_pull_y08_eval_gen7/batch_metrics.json) | 16 | 43 | —/0.080/0.120 | 15 | — | — | 0 | 0.96/1.65 | 旧几何口径；严格完成不可判定 |
| [`supported_cem_pull_y08_wide_eval_gen0`](../../artifacts/table_policy/supported_cem_pull_y08_wide_eval_gen0/batch_metrics.json) | 16 | 45 | —/0.080/0.120 | 15 | — | — | 0 | 1.58/4.77 | 旧几何口径；严格完成不可判定 |
| [`supported_cem_pull_y08_width12_eval`](../../artifacts/table_policy/supported_cem_pull_y08_width12_eval/batch_metrics.json) | 16 | 43 | —/0.080/0.120 | 16 | — | — | 0 | -0.34/0.68 | 旧几何口径；严格完成不可判定 |
| [`supported_cem_robust_x345_eval_gen0`](../../artifacts/table_policy/supported_cem_robust_x345_eval_gen0/batch_metrics.json) | 32 | 54 | 0.345/0.080/0.120 | 31 | 1 | — | 0 | 3.41/6.51 | 旧几何口径；严格完成不可判定 |
| [`supported_cem_safe_y06_eval`](../../artifacts/table_policy/supported_cem_safe_y06_eval/batch_metrics.json) | 8 | 32 | 0.350/—/— | 8 | — | — | 0 | -1.22/-0.57 | 旧几何口径；严格完成不可判定 |
| [`supported_cem_second_pull_eval_gen2`](../../artifacts/table_policy/supported_cem_second_pull_eval_gen2/batch_metrics.json) | 16 | 51 | —/0.080/0.120 | 16 | — | — | 0 | 2.35/6.44 | 旧几何口径；严格完成不可判定 |
| [`supported_cem_yaw_fullout_eval`](../../artifacts/table_policy/supported_cem_yaw_fullout_eval/batch_metrics.json) | 32 | 52 | —/0.080/0.120 | 30 | 1 | — | 0 | 2.96/7.01 | 旧几何口径；严格完成不可判定 |
| [`supported_cem_yaw_pull_eval_gen1`](../../artifacts/table_policy/supported_cem_yaw_pull_eval_gen1/batch_metrics.json) | 16 | 52 | —/0.080/0.120 | 15 | 0 | — | 0 | 3.41/6.30 | 旧几何口径；严格完成不可判定 |
| [`supported_cem_yaw_x345_eval`](../../artifacts/table_policy/supported_cem_yaw_x345_eval/batch_metrics.json) | 16 | 52 | 0.345/0.080/0.120 | 15 | 4 | — | 0 | 4.18/7.13 | 旧几何口径；严格完成不可判定 |
| [`supported_cem_yaw_x345_no_table_eval`](../../artifacts/table_policy/supported_cem_yaw_x345_no_table_eval/batch_metrics.json) | 8 | 52 | 0.345/0.080/0.120 | 0 | 0 | — | 0 | — | 全部失稳；倒地后物块运动不计成功 |
| [`supported_dagger_noise03`](../../artifacts/table_policy/supported_dagger_noise03/batch_metrics.json) | 16 | 21 | 0.350/—/— | 16 | — | — | 0 | 0.00/0.00 | 旧几何口径；严格完成不可判定 |
| [`supported_high_policy_baseline`](../../artifacts/table_policy/supported_high_policy_baseline/batch_metrics.json) | 16 | 17 | 0.350/—/— | 0 | — | — | 0 | — | 全部失稳；倒地后物块运动不计成功 |
| [`supported_residual_block12_eval`](../../artifacts/table_policy/supported_residual_block12_eval/batch_metrics.json) | 16 | 25 | 0.350/—/— | 15 | — | — | 0 | -18.04/-14.81 | 旧几何口径；严格完成不可判定 |
| [`supported_stage2_baseline`](../../artifacts/table_policy/supported_stage2_baseline/batch_metrics.json) | 16 | 17 | 0.350/—/— | 0 | — | — | 0 | — | 全部失稳；倒地后物块运动不计成功 |
| [`supported_stage2b_50_eval`](../../artifacts/table_policy/supported_stage2b_50_eval/batch_metrics.json) | 16 | 18 | 0.350/—/— | 0 | — | — | 0 | — | 全部失稳；倒地后物块运动不计成功 |

共 78 个批量评估；原始逐轨迹数据见每行链接。
