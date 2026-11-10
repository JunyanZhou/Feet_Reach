#!/usr/bin/env python3
"""Scripted G1 foot-braced, two-hand drawer pull with MuJoCo object dynamics."""
from __future__ import annotations
import argparse, copy, json, os, sys, xml.etree.ElementTree as ET
from pathlib import Path
if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/table_retrieval'))
from demo import IK, robot_xml, smooth, vec, MODEL

OUT = ROOT / 'artifacts/foot_brace_drawer'
FPS = 60
KEYS = [
    # time, root xyz, root pitch, left sole, right sole, left palm, right palm, phase
    (0.0, [-.20, 0, .73], .05, [.035, .12, 0], [-.06, -.14, 0], [.04, .22, .76], [.04, -.22, .76], 'ready'),
    (.8, [-.16, 0, .70], .10, [.10, .17, .31], [-.06, -.14, 0], [.14, .20, .74], [.14, -.20, .74], 'brace approach'),
    (1.25, [-.12, 0, .68], .15, [.36, .22, .31], [-.06, -.14, 0], [.20, .17, .84], [.17, -.17, .84], 'foot above base'),
    (1.7, [-.12, 0, .68], .15, [.36, .22, .242], [-.06, -.14, 0], [.20, .15, .82], [.18, -.15, .82], 'foot braces base'),
    (2.6, [-.12, 0, .68], .15, [.36, .22, .242], [-.06, -.14, 0], [.193, .115, .80], [.203, -.115, .80], 'both hands grip'),
    (4.8, [-.16, 0, .68], .15, [.36, .22, .242], [-.06, -.14, 0], [-.037, .115, .80], [-.027, -.115, .80], 'pull drawer'),
    (5.1, [-.16, 0, .68], .15, [.36, .22, .242], [-.06, -.14, 0], [-.037, .115, .80], [-.027, -.115, .80], 'drawer extracted'),
    (5.8, [-.16, 0, .68], .10, [.36, .22, .242], [-.06, -.14, 0], [-.04, .18, .90], [-.04, -.18, .90], 'release handles'),
    (6.15, [-.16, 0, .68], .10, [.36, .22, .31], [-.06, -.14, 0], [-.02, .19, .90], [-.02, -.19, .90], 'lift foot'),
    (6.7, [-.18, 0, .70], .08, [.13, .17, .31], [-.06, -.14, 0], [.03, .22, .90], [.03, -.22, .90], 'withdraw foot'),
    (7.5, [-.20, 0, .72], .04, [-.08, .14, 0], [-.06, -.14, 0], [0, .22, .75], [0, -.22, .75], 'stand'),
    (8.0, [-.20, 0, .72], .04, [-.08, .14, 0], [-.06, -.14, 0], [0, .22, .75], [0, -.22, .75], 'done'),
]


def target_at(t):
    key_times = [k[0] for k in KEYS]
    j = min(max(np.searchsorted(key_times, t, side='right') - 1, 0), len(KEYS) - 2)
    a, b = KEYS[j], KEYS[j + 1]
    u = float(smooth((t - a[0]) / (b[0] - a[0])))
    interp = lambda i: (1-u)*np.array(a[i]) + u*np.array(b[i])
    targets = dict(left_sole=interp(3), right_sole=interp(4),
                   left_palm=interp(5), right_palm=interp(6))
    return interp(1), [0, float(interp(2)), 0], targets, b[7]


def make_scene(ik, out):
    root = ET.Element('mujoco', model='foot_braced_two_hand_drawer')
    ET.SubElement(root, 'compiler', angle='radian', meshdir=str(MODEL.parent / 'meshes'))
    ET.SubElement(root, 'option', timestep='.002', gravity='0 0 -9.81',
                  integrator='implicitfast', iterations='100', cone='elliptic')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1920', offheight='1080')
    ET.SubElement(visual, 'quality', shadowsize='4096', offsamples='4')
    ET.SubElement(visual, 'headlight', ambient='.34 .34 .34',
                  diffuse='.58 .58 .58', specular='.2 .2 .2')
    asset = copy.deepcopy(ik.xml.find('asset'))
    root.append(asset)
    ET.SubElement(asset, 'texture', name='sky', type='skybox', builtin='gradient',
                  rgb1='.22 .28 .36', rgb2='.065 .09 .13', width='512', height='3072')
    ET.SubElement(asset, 'texture', name='floor_tex', type='2d', builtin='checker',
                  rgb1='.19 .23 .27', rgb2='.22 .26 .30', width='512', height='512')
    ET.SubElement(asset, 'material', name='floor_mat', texture='floor_tex',
                  texrepeat='4 4', reflectance='0')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', name='key', pos='-1 -2 4', dir='.3 .4 -1',
                  diffuse='.85 .85 .82', directional='true', castshadow='true')
    ET.SubElement(world, 'light', name='fill', pos='2 2 3', dir='-.3 -.3 -1',
                  diffuse='.5 .6 .7', directional='true', castshadow='false')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='5 5 .05',
                  material='floor_mat', friction='.17 .01 .001', contype='1', conaffinity='2')
    for src in ik.xml.findall('.//body'):
        body = ET.SubElement(world, 'body', name=src.get('name'), mocap='true')
        visuals = [g for g in src.findall('geom')
                   if g.get('type') == 'mesh' and g.get('group') == '1']
        for j, geom in enumerate(visuals):
            geom = copy.deepcopy(geom)
            geom.set('name', src.get('name') + '_visual_' + str(j))
            geom.set('contype', '0')
            geom.set('conaffinity', '0')
            body.append(geom)
        if src.get('name') not in ['left_ankle_roll_link', 'right_ankle_roll_link']:
            for j, original_geom in enumerate(src.findall('geom')):
                is_hand = original_geom.get('mesh', '').endswith('_rubber_hand')
                if original_geom.get('group') == '1' and not is_hand:
                    continue
                proxy = copy.deepcopy(original_geom)
                proxy.set('name', src.get('name') + '_proxy_' + str(j))
                # During a grasp the weld constraint transmits hand forces.
                # Hand mesh collision response is disabled for this gripped
                # object to avoid imposing two incompatible constraints;
                # the mesh remains available for explicit distance auditing.
                proxy.set('contype', '0' if is_hand else '1')
                proxy.set('conaffinity', '0' if is_hand else '2')
                proxy.set('group', '3')
                proxy.set('rgba', '0 0 0 0')
                proxy.set('density', '0')
                body.append(proxy)
        if src.get('name') == 'left_ankle_roll_link':
            ET.SubElement(body, 'geom', name='brace_foot', type='box',
                          pos='.035 0 -.018', size='.09 .038 .020',
                          rgba='0 0 0 0', group='3', contype='1',
                          conaffinity='2', friction='1.2 .01 .001',
                          solref='.05 1', solimp='.9 .99 .001')
    grip = ET.SubElement(world, 'body', name='grip_mocap', mocap='true',
                         pos='.335 0 .80')
    ET.SubElement(grip, 'site', name='grip_site', pos='0 0 0', size='.004', rgba='0 0 0 0')
    base = ET.SubElement(world, 'body', name='base', pos='.48 0 .005')
    ET.SubElement(base, 'freejoint', name='base_free')
    ET.SubElement(base, 'geom', name='base_box', type='box', pos='0 0 .12',
                  size='.18 .27 .12', mass='1.3', friction='.10 .01 .001',
                  contype='2', conaffinity='1', rgba='.30 .39 .48 1')
    for side, y in [('far_side', -.205)]:
        ET.SubElement(base, 'geom', name='frame_' + side, type='box',
                      pos=vec([0, y, .79]), size='.18 .014 .11',
                      mass='0', contype='2', conaffinity='1',
                      rgba='.16 .21 .27 1')
    ET.SubElement(base, 'geom', name='frame_top', type='box',
                  pos='0 0 .915', size='.18 .20 .015', mass='0',
                  contype='2', conaffinity='1', rgba='.17 .22 .28 1')
    ET.SubElement(base, 'geom', name='frame_back', type='box',
                  pos='.16 0 .565', size='.014 .19 .34', mass='0',
                  contype='2', conaffinity='1', rgba='.16 .21 .27 1')
    drawer = ET.SubElement(base, 'body', name='drawer', pos='-.145 0 .795')
    ET.SubElement(drawer, 'joint', name='drawer_slide', type='slide',
                  axis='-1 0 0', range='0 .245', limited='true',
                  damping='2', frictionloss='5')
    ET.SubElement(drawer, 'geom', name='drawer_tray', type='box',
                  pos='.11 0 0', size='.115 .16 .07', mass='.42',
                  contype='2', conaffinity='1', rgba='.68 .43 .24 1')
    ET.SubElement(drawer, 'geom', name='drawer_front', type='box',
                  pos='0 0 0', size='.018 .18 .075', mass='.08',
                  contype='2', conaffinity='1', rgba='.81 .54 .30 1')
    for side, x, y in [('left', -.063, .115), ('right', -.058, -.123)]:
        ET.SubElement(drawer, 'geom', name='handle_' + side, type='sphere',
                      pos=vec([x, y, 0]), size='.010', mass='0',
                      contype='2', conaffinity='1', rgba='.15 .21 .27 1')
    equality = ET.SubElement(root, 'equality')
    ET.SubElement(equality, 'weld', name='two_hand_grasp', body1='grip_mocap',
                  body2='drawer', active='false', solref='.03 1')
    path = out / 'scene.xml'
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='unicode')
    return mujoco.MjModel.from_xml_path(str(path))


def build_motion(out):
    ik = IK()
    t = np.arange(round(KEYS[-1][0] * FPS) + 1) / FPS
    qs, bp, bq, sites, errors, phases = [], [], [], [], [], []
    nominal = ik.nominal.copy()
    for i, ti in enumerate(t):
        root, rpy, targets, phase = target_at(ti)
        ik.nominal = nominal
        ik.max_joint_delta = None if i == 0 else .065
        q, err = ik.solve(root, rpy, targets,
                          dict(left_sole=[0, 0, 0], right_sole=[0, 0, 0],
                               right_palm=[.04, -.20, -.20]),
                          iterations=60 if i == 0 else 22)
        qs.append(q)
        bp.append(ik.data.xpos[1:].copy())
        bq.append(ik.data.xquat[1:].copy())
        sites.append([ik.data.site_xpos[ik.sites[n]].copy() for n in targets])
        errors.append(list(err.values()))
        phases.append(phase)
        if i % 60 == 0:
            print(f'IK {i}/{len(t)} t={ti:.1f} max_error={max(err.values()):.4f}', flush=True)
    motion = dict(time=t, qpos=np.array(qs), body_pos=np.array(bp),
                  body_quat_wxyz=np.array(bq), site_pos=np.array(sites),
                  ik_error=np.array(errors), phase=np.array(phases),
                  fps=np.array(FPS), joint_names=np.array(ik.joints),
                  body_names=np.array(ik.body_names),
                  site_names=np.array(list(targets)))
    np.savez_compressed(out / 'motion.npz', **motion)
    return ik, motion


def set_robot(model, data, motion, t):
    fps = float(motion['fps'])
    idx = min(int(t * fps + 1e-7), len(motion['time']) - 2)
    u = np.clip(t * fps - idx, 0, 1)
    n = len(motion['body_names'])
    data.mocap_pos[:n] = (1-u) * motion['body_pos'][idx] + u * motion['body_pos'][idx+1]
    quat = (1-u) * motion['body_quat_wxyz'][idx] + u * motion['body_quat_wxyz'][idx+1]
    data.mocap_quat[:n] = quat / np.linalg.norm(quat, axis=1, keepdims=True)


def simulate(model, motion, out, foot=True):
    if not foot:
        fid = model.geom('brace_foot').id
        model.geom_contype[fid] = model.geom_conaffinity[fid] = 0
    data = mujoco.MjData(model)
    n = len(motion['body_names'])
    grip_id = model.body('grip_mocap').mocapid[0]
    weld_id = model.equality('two_hand_grasp').id
    foot_id = model.geom('brace_foot').id
    base_id = model.geom('base_box').id
    drawer_id = model.body('drawer').id
    duration = float(motion['time'][-1])
    nstep = round(duration / model.opt.timestep)
    samples, foot_forces, times = [], [], []
    next_sample = 0.0
    initial_grip = None
    initial_grip_quat = None
    peak_foot = 0.0
    worst_unintended = {'penetration_m': 0.0, 'time_s': None, 'geoms': []}
    furniture = {'base_box', 'frame_far_side', 'frame_top', 'frame_back',
                 'drawer_tray', 'drawer_front', 'handle_left', 'handle_right'}
    for step in range(nstep + 1):
        ti = step * model.opt.timestep
        set_robot(model, data, motion, ti)
        if ti < 2.6:
            data.eq_active[weld_id] = 0
            mujoco.mj_forward(model, data)
            data.mocap_pos[grip_id] = data.xpos[drawer_id]
            data.mocap_quat[grip_id] = data.xquat[drawer_id]
        else:
            if initial_grip is None:
                mujoco.mj_forward(model, data)
                initial_grip = data.xpos[drawer_id].copy()
                initial_grip_quat = data.xquat[drawer_id].copy()
            data.eq_active[weld_id] = 1 if ti < 5.2 else 0
            u = float(smooth((ti - 2.6) / (4.8 - 2.6)))
            data.mocap_pos[grip_id] = initial_grip + [-.23*u, 0, 0]
            data.mocap_quat[grip_id] = initial_grip_quat
        mujoco.mj_forward(model, data)
        normal_force = 0.0
        for ci in range(data.ncon):
            c = data.contact[ci]
            names = [model.geom(c.geom1).name, model.geom(c.geom2).name]
            if (c.dist < -worst_unintended['penetration_m']
                and any(n in furniture for n in names)
                and any(n == 'brace_foot' or '_proxy_' in n for n in names)
                and set(names) != {'brace_foot', 'base_box'}):
                worst_unintended = {'penetration_m': -float(c.dist),
                                    'time_s': ti, 'geoms': names}
            if {c.geom1, c.geom2} == {foot_id, base_id}:
                f = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, f)
                normal_force += f[0]
        peak_foot = max(peak_foot, normal_force)
        if ti + 1e-8 >= next_sample:
            samples.append(data.qpos.copy())
            foot_forces.append(normal_force)
            times.append(ti)
            next_sample += 1 / FPS
        if step < nstep:
            mujoco.mj_step(model, data)
    samples = np.array(samples)
    # The 2 ms integrator is sampled at the first step crossing each 60 Hz tick.
    sample_times = np.array(times)
    result = dict(time=sample_times, base_qpos=samples[:, :7],
                  drawer_slide=samples[:, 7], foot_base_force=np.array(foot_forces))
    suffix = '' if foot else '_no_foot'
    np.savez_compressed(out / f'simulation{suffix}.npz', **result)
    summary = dict(duration_s=duration, foot_enabled=foot,
                   drawer_extension_m=float(samples[-1, 7]),
                   max_drawer_extension_m=float(samples[:, 7].max()),
                   base_horizontal_shift_m=float(np.linalg.norm(samples[-1, :2] - samples[0, :2])),
                   peak_foot_base_normal_force_N=float(peak_foot),
                   max_unintended_robot_furniture_penetration_m=worst_unintended['penetration_m'],
                   worst_unintended_robot_furniture_contact=worst_unintended,
                   foot_contact_samples=int(np.count_nonzero(result['foot_base_force'] > 0)),
                   task_success=bool(samples[:, 7].max() > .20 and
                                     np.linalg.norm(samples[-1, :2] - samples[0, :2]) < .05))
    (out / f'evaluation{suffix}.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    return result, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--reuse-motion', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.reuse_motion:
        ik = IK()
        motion = dict(np.load(args.output / 'motion.npz'))
    else:
        ik, motion = build_motion(args.output)
    model = make_scene(ik, args.output)
    simulate(model, motion, args.output)
    simulate(make_scene(ik, args.output), motion, args.output, foot=False)


if __name__ == '__main__':
    main()
