"""Build a compact, auditable index of saved table-retrieval batch evaluations.

This script reads only saved metrics; it never launches a simulator or changes a
rollout.  The geometry-aware column is recomputed from the recorded block corner
and support measurements so old versions of ``task_success`` cannot be confused
with the current task criterion.
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def fmt(value):
    return "—" if value is None else str(value)


def stable(row):
    return row.get("root_min_height_m", float("-inf")) > 0.50


def strict(row):
    corner = row.get("object_final_backmost_x_m")
    return bool(
        corner is not None
        and corner < 0.280
        and stable(row)
        and row.get("simultaneous_hand_and_foot_support_s", 0) > 5.0
        and row.get("supported_foot_push_s", 0) > 0.08
    )


def scene(rows):
    row = rows[0]
    obj = row.get("object_config") or {}
    x = obj.get("x_m", row.get("object_initial_x_m"))
    y = obj.get("y_m")
    h = obj.get("height_m")
    return "/".join("—" if value is None else f"{value:.3f}" for value in (x, y, h))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts/table_policy")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "scripts/table_retrieval/EVALUATION_LEDGER.md",
    )
    args = parser.parse_args()
    paths = sorted(args.artifacts.glob("*/batch_metrics.json"))
    lines = [
        "# 扶桌脚拨物块：全部批量评估台账",
        "",
        "由 `generate_evaluation_ledger.py` 从本仓库保存的 `batch_metrics.json` 生成。",
        "每一行是一次独立批量评估，不是 CEM 训练批次。场景列依次为物块中心",
        "x/y/高度，单位 m；`—` 表示早期日志没有保存该字段。位移为 root",
        "始终高于 0.50 m 的轨迹中的平均/最大物块净外移，单位 cm。",
        "",
        "**严格完整离桌**统一重算：末帧按物块姿态得到的最靠桌内角",
        "x<0.280 m、root 最低高度>0.50 m、左手向上与右脚承重重叠>5 s、",
        "受支撑时脚靠近物块且物块外移累计>0.08 s。只有保存了三维旋转后",
        "后角的日志才可计算。早期 `task_success` 用固定半宽或不同条件，",
        "放在“旧成功”列，**不能**当作严格完成数。中心越沿沿用各批次",
        "日志中的 `front_edge_crossing_success`；早期缺失时显示 `—`。",
        "",
        "| 批次原始记录 | n | seed | 场景 x/y/h | 稳定 | 中心越沿 | 严格完整 | 旧成功 | 稳定轨迹位移均值/最大 cm | 结果与限制 |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    rows_written = 0
    for path in paths:
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or not rows:
            continue
        geometry = all("object_final_backmost_x_m" in row for row in rows)
        n = len(rows)
        stable_rows = [row for row in rows if stable(row)]
        strict_n = sum(strict(row) for row in rows) if geometry else None
        center_n = (
            sum(bool(row["front_edge_crossing_success"]) for row in rows)
            if all("front_edge_crossing_success" in row for row in rows) else None
        )
        legacy_n = None if geometry else sum(bool(row.get("task_success")) for row in rows)
        displacements = [
            row["object_outward_displacement_m"] * 100
            for row in stable_rows if "object_outward_displacement_m" in row
        ]
        displacement = (
            f"{sum(displacements)/len(displacements):.2f}/{max(displacements):.2f}"
            if displacements else "—"
        )
        if not stable_rows:
            interpretation = "全部失稳；倒地后物块运动不计成功"
        elif not geometry:
            interpretation = "旧几何口径；严格完成不可判定"
        elif strict_n:
            interpretation = "有完整离桌；须结合场景和独立种子判断泛化"
        elif center_n:
            interpretation = "中心越沿，但旋转后仍有角留在桌内"
        else:
            interpretation = "稳定但未满足中心越沿或接触条件"
        relative = path.relative_to(ROOT).as_posix()
        link = f"[\u0060{path.parent.name}\u0060](../../{relative})"
        lines.append(
            f"| {link} | {n} | {fmt(rows[0].get('seed'))} | {scene(rows)} | "
            f"{len(stable_rows)} | {fmt(center_n)} | {fmt(strict_n)} | "
            f"{fmt(legacy_n)} | {displacement} | {interpretation} |"
        )
        rows_written += 1
    lines.extend(["", f"共 {rows_written} 个批量评估；原始逐轨迹数据见每行链接。", ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"Wrote {args.output} with {rows_written} batches")


if __name__ == "__main__":
    main()
