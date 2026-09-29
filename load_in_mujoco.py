#!/usr/bin/env python3
"""
将 linkerhand_o6_right.urdf 导入 MuJoCo，并修复以下 URDF 已知问题：
  1. MuJoCo 完全忽略 <mimic> 标签 → 手动注入 equality 约束
  2. 为非 mimic 主动关节注入位置执行器（可选）

依赖: pip install "mujoco>=3.1.0"
网格：URDF 同目录下 meshes/*.STL 必须存在。
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer


def default_urdf_path() -> Path:
    return Path(__file__).resolve().parent / "linkerhand_o6_right.urdf"


# ── URDF 解析 ────────────────────────────────────────────────────────────────

def _parse_joints(urdf_path: Path) -> list[dict]:
    """解析 URDF 所有关节，返回关节信息列表。"""
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    joints = []
    for j in root.findall("joint"):
        jtype = j.attrib.get("type", "fixed")
        if jtype == "fixed":
            continue
        limit = j.find("limit")
        mimic = j.find("mimic")
        joints.append({
            "name": j.attrib["name"],
            "type": jtype,
            "lower": float(limit.attrib.get("lower", "0")) if limit is not None else 0.0,
            "upper": float(limit.attrib.get("upper", "0")) if limit is not None else 0.0,
            "mimic": mimic.attrib.get("joint") if mimic is not None else None,
            "multiplier": float(mimic.attrib.get("multiplier", "1")) if mimic is not None else 1.0,
            "offset": float(mimic.attrib.get("offset", "0")) if mimic is not None else 0.0,
        })
    return joints


def parse_mimic_joints(urdf_path: Path) -> list[tuple[str, str, float, float]]:
    """返回 (slave, master, multiplier, offset) 列表。"""
    return [
        (j["name"], j["mimic"], j["multiplier"], j["offset"])
        for j in _parse_joints(urdf_path)
        if j["mimic"] is not None
    ]


def parse_master_joints(urdf_path: Path) -> list[dict]:
    """返回非 mimic（主动）关节列表，每个元素含 name/lower/upper。"""
    return [j for j in _parse_joints(urdf_path) if j["mimic"] is None]


# ── XML 构建 ─────────────────────────────────────────────────────────────────

def _build_equality_xml(mimic_joints: list[tuple[str, str, float, float]]) -> str:
    """
    构造 <equality> 段落。
    MuJoCo: q_slave = polycoef[0] + polycoef[1]*q_master
    URDF mimic: q_slave = offset + multiplier * q_master
    """
    if not mimic_joints:
        return ""
    lines = ["  <equality>"]
    for slave, master, mult, off in mimic_joints:
        lines.append(
            f'    <joint joint1="{slave}" joint2="{master}" '
            f'polycoef="{off} {mult} 0 0 0"/>'
        )
    lines.append("  </equality>")
    return "\n".join(lines)


def _build_actuator_xml(master_joints: list[dict], kp: float = 5.0) -> str:
    """
    为每个主动关节注入位置执行器（position actuator）。
    只对 master 关节添加，slave (mimic) 关节跟随约束自动运动。

    kp : 位置增益（N·m/rad），根据实际手调整。
    """
    if not master_joints:
        return ""
    lines = ["  <actuator>"]
    for j in master_joints:
        lo, hi = j["lower"], j["upper"]
        lines.append(
            f'    <position name="act_{j["name"]}" joint="{j["name"]}" '
            f'kp="{kp}" ctrlrange="{lo} {hi}" forcelimited="true" forcerange="-5 5"/>'
        )
    lines.append("  </actuator>")
    return "\n".join(lines)


# ── 资源加载 ─────────────────────────────────────────────────────────────────

def _load_mesh_assets(urdf_path: Path) -> dict[str, bytes]:
    meshes_dir = urdf_path.parent / "meshes"
    assets: dict[str, bytes] = {}
    if meshes_dir.is_dir():
        for stl in list(meshes_dir.glob("*.STL")) + list(meshes_dir.glob("*.stl")):
            assets[f"meshes/{stl.name}"] = stl.read_bytes()
    return assets


# ── 主加载接口 ────────────────────────────────────────────────────────────────

def load_hand(
    urdf_path: Path | str,
    *,
    fix_mimic: bool = True,
    add_actuators: bool = False,
    actuator_kp: float = 5.0,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """
    加载并编译灵巧手模型。

    Parameters
    ----------
    urdf_path     : URDF 文件路径
    fix_mimic     : 注入 mimic equality 约束（默认 True，必须开启）
    add_actuators : 为主动关节注入位置执行器（默认 False）
    actuator_kp   : 位置执行器增益 kp（N·m/rad）
    """
    path = Path(urdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到 URDF: {path}")

    spec = mujoco.MjSpec.from_file(str(path))
    xml: str = spec.to_xml()

    inserts: list[str] = []
    if fix_mimic:
        eq_xml = _build_equality_xml(parse_mimic_joints(path))
        if eq_xml:
            inserts.append(eq_xml)
    if add_actuators:
        act_xml = _build_actuator_xml(parse_master_joints(path), kp=actuator_kp)
        if act_xml:
            inserts.append(act_xml)

    if inserts:
        xml = xml.replace("</mujoco>", "\n".join(inserts) + "\n</mujoco>")

    model = mujoco.MjModel.from_xml_string(xml, _load_mesh_assets(path))
    data = mujoco.MjData(model)
    return model, data


def get_joint_index(model: mujoco.MjModel, name: str) -> int:
    """按名称查关节 qpos 索引，找不到抛 KeyError。"""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(f"关节 '{name}' 不存在")
    return model.jnt_qposadr[jid]


def get_actuator_index(model: mujoco.MjModel, name: str) -> int:
    """按名称查执行器索引，找不到抛 KeyError。"""
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        raise KeyError(f"执行器 '{name}' 不存在")
    return aid


# ── 摘要打印 ──────────────────────────────────────────────────────────────────

def print_model_summary(model: mujoco.MjModel) -> None:
    print(f"\n── 模型统计 ──────────────────────────────────")
    print(f"  nbody={model.nbody}  nq={model.nq}  nv={model.nv}  nu={model.nu}")
    print(f"  neq (等效约束)={model.neq}")

    print(f"\n── 关节列表 ───────────────────────────────────")
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        lo = model.jnt_range[j, 0]
        hi = model.jnt_range[j, 1]
        print(f"  [{j:2d}] {name:<30s} range=[{lo:.3f}, {hi:.3f}]")

    if model.neq > 0:
        print(f"\n── Equality 约束（mimic） ─────────────────────")
        for i in range(model.neq):
            j1 = model.eq_obj1id[i]
            j2 = model.eq_obj2id[i]
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j1) or str(j1)
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j2) or str(j2)
            coef = model.eq_data[i, :2]
            print(f"  [{i}] {n1} = {coef[0]:.3f} + {coef[1]:.3f} * {n2}")

    if model.nu > 0:
        print(f"\n── 执行器列表（data.ctrl 索引） ───────────────")
        for a in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"act_{a}"
            lo, hi = model.actuator_ctrlrange[a]
            print(f"  [{a:2d}] {name:<35s} ctrl=[{lo:.3f}, {hi:.3f}]")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Linker O6 右手 URDF -> MuJoCo")
    parser.add_argument("--urdf", type=Path, default=None,
                        help="URDF 路径（默认：脚本同目录 linkerhand_o6_right.urdf）")
    parser.add_argument("--viewer", action="store_true",
                        help="启动 MuJoCo 被动查看器")
    parser.add_argument("--simulate", action="store_true",
                        help="查看器中运行物理仿真")
    parser.add_argument("--no-gravity", action="store_true",
                        help="禁用重力（与 --simulate 配合）")
    parser.add_argument("--no-mimic", action="store_true",
                        help="不注入 mimic 约束（调试用）")
    parser.add_argument("--actuators", action="store_true",
                        help="为主动关节注入位置执行器")
    parser.add_argument("--kp", type=float, default=5.0,
                        help="位置执行器增益 kp（默认 5.0）")
    parser.add_argument("--export-mjcf", type=Path, default=None,
                        help="将修正后的 MJCF 导出为 XML 文件")
    args = parser.parse_args()

    urdf = args.urdf if args.urdf is not None else default_urdf_path()

    try:
        model, data = load_hand(
            urdf,
            fix_mimic=not args.no_mimic,
            add_actuators=args.actuators,
            actuator_kp=args.kp,
        )
    except Exception as e:
        print(f"加载失败: {e}", file=sys.stderr)
        return 1

    print_model_summary(model)

    if args.export_mjcf is not None:
        urdf_r = Path(urdf).expanduser().resolve()
        spec = mujoco.MjSpec.from_file(str(urdf_r))
        xml = spec.to_xml()
        inserts: list[str] = []
        if not args.no_mimic:
            eq = _build_equality_xml(parse_mimic_joints(urdf_r))
            if eq:
                inserts.append(eq)
        if args.actuators:
            act = _build_actuator_xml(parse_master_joints(urdf_r), kp=args.kp)
            if act:
                inserts.append(act)
        if inserts:
            xml = xml.replace("</mujoco>", "\n".join(inserts) + "\n</mujoco>")
        args.export_mjcf.write_text(xml, encoding="utf-8")
        print(f"已写入修正后 MJCF: {args.export_mjcf.resolve()}")

    if args.viewer:
        if args.simulate and args.no_gravity:
            model.opt.gravity[:] = 0
            print("已禁用重力")

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                if args.simulate:
                    mujoco.mj_step(model, data)
                else:
                    mujoco.mj_forward(model, data)
                viewer.sync()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
