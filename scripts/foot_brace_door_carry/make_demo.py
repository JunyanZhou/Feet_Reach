#!/usr/bin/env python3
"""Scripted G1 carries a box through a spring-closing door braced by its foot."""
from __future__ import annotations
import argparse
import copy
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/table_retrieval'))
from demo import IK, MODEL, smooth, vec

OUT = ROOT / 'artifacts/foot_brace_door_carry'
FPS = 60
DOOR_OPEN = -1.15
# time, root xyz, pitch, left sole, right sole, box center, phase
KEYS = [
    (0.0, [-.31, -.15, .73], .05, [-.23, -.01, 0], [-.23, -.30, 0], [.00, -.15, .94], 'carry approach'),
    (.7, [-.26, -.15, .72], .05, [.15, -.05, .15], [-.23, -.30, 0], [.07, -.15, .94], 'lift left foot'),
    (1.35, [-.13, -.10, .71], .08, [.22, .10, .13], [-.23, -.30, 0], [.15, -.15, .94], 'foot around door tip'),
    (1.75, [-.10, -.10, .71], .08, [.21, .205, .04], [-.23, -.30, 0], [.18, -.15, .94], 'foot braces door'),
    (2.65, [0, -.10, .72], .07, [.21, .205, .04], [-.23, -.30, 0], [.31, -.15, .94], 'carry forward'),
    (3.65, [.08, -.10, .73], .04, [.21, .205, .04], [.12, -.30, 0], [.43, -.15, .94], 'step right through'),
    (4.15, [.14, -.10, .73], .03, [.21, .205, .04], [.12, -.30, 0], [.50, -.15, .94], 'hold door open'),
    (4.55, [.18, -.10, .73], .03, [.37, .06, .20], [.12, -.30, 0], [.54, -.15, .94], 'release door'),
    (4.8, [.21, -.10, .73], .03, [.42, -.04, .17], [.12, -.30, 0], [.57, -.15, .94], 'withdraw foot'),
    (5.55, [.33, -.10, .74], .02, [.60, -.12, 0], [.12, -.30, 0], [.69, -.15, .94], 'step left through'),
    (6.5, [.48, -.10, .74], .02, [.60, -.12, 0], [.70, -.30, 0], [.84, -.15, .94], 'step right through'),
    (7.5, [.65, -.10, .74], .01, [.96, -.01, 0], [.70, -.30, 0], [1.01, -.15, .94], 'leave doorway'),
    (8.0, [.71, -.10, .74], .01, [.96, -.01, 0], [.70, -.30, 0], [1.07, -.15, .94], 'carry away'),
    (8.8, [.91, -.10, .74], .01, [.96, -.01, 0], [1.16, -.30, 0], [1.27, -.15, .94], 'step right away'),
    (9.6, [1.10, -.10, .74], .01, [1.40, -.01, 0], [1.16, -.30, 0], [1.46, -.15, .94], 'step left away'),
    (10.0, [1.18, -.10, .74], .01, [1.40, -.01, 0], [1.16, -.30, 0], [1.54, -.15, .94], 'clear doorway'),
]


def target_at(t):
    times = [k[0] for k in KEYS]
    j = min(max(np.searchsorted(times, t, side='right') - 1, 0), len(KEYS)-2)
    a, b = KEYS[j], KEYS[j+1]
    u = float(smooth((t-a[0])/(b[0]-a[0])))
    interp = lambda i: (1-u)*np.array(a[i]) + u*np.array(b[i])
    center = interp(5)
    palms = [center + [-.015, .166, 0], center + [-.015, -.166, 0]]
    return interp(1), [0, float(interp(2)), 0], dict(
        left_sole=interp(3), right_sole=interp(4),
        left_palm=palms[0], right_palm=palms[1]), center, b[6]


def make_scene(ik, out):
    root = ET.Element('mujoco', model='foot_braced_door_carry')
    ET.SubElement(root, 'compiler', angle='radian', meshdir=str(MODEL.parent/'meshes'))
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
                  material='floor_mat', friction='.8 .01 .001', contype='1', conaffinity='2')
    for src in ik.xml.findall('.//body'):
        body = ET.SubElement(world, 'body', name=src.get('name'), mocap='true')
        for j, original in enumerate(src.findall('geom')):
            if original.get('type') == 'mesh' and original.get('group') == '1':
                geom = copy.deepcopy(original)
                geom.set('name', src.get('name') + '_visual_' + str(j))
                geom.set('contype', '0')
                geom.set('conaffinity', '0')
                body.append(geom)
            if src.get('name') in ['left_ankle_roll_link', 'right_ankle_roll_link']:
                continue
            is_hand = original.get('mesh', '').endswith('_rubber_hand')
            # The sole box is the designated load-bearing foot contact.
            # The ankle mesh overlaps its contact margin and is distance
            # audited, but must not act as a second door stop.
            is_contact_duplicate = src.get('name') in (
                'left_ankle_pitch_link', 'left_knee_link')
            if original.get('group') == '1' and not is_hand:
                continue
            proxy = copy.deepcopy(original)
            proxy.set('name', src.get('name') + '_proxy_' + str(j))
            proxy.set('contype', '0' if is_hand or is_contact_duplicate else '1')
            proxy.set('conaffinity', '0' if is_hand or is_contact_duplicate else '2')
            proxy.set('group', '3')
            proxy.set('rgba', '0 0 0 0')
            proxy.set('density', '0')
            body.append(proxy)
        if src.get('name') in ['left_ankle_roll_link', 'right_ankle_roll_link']:
            side = src.get('name').split('_')[0]
            ET.SubElement(body, 'geom', name=side+'_foot_collision', type='box',
                          pos='.035 0 -.018', size='.09 .038 .020',
                          rgba='0 0 0 0', group='3', contype='1',
                          conaffinity='2', friction='1.2 .01 .001',
                          solref='.04 1', solimp='.9 .99 .001')
    for name, y in [('hinge_post', .575), ('latch_post', -.475)]:
        ET.SubElement(world, 'geom', name=name, type='box', pos=vec([.65,y,.65]),
                      size='.055 .045 .65', contype='2', conaffinity='1',
                      rgba='.19 .25 .31 1')
    ET.SubElement(world, 'geom', name='lintel', type='box', pos='.65 0 1.35',
                  size='.06 .52 .055', contype='2', conaffinity='1', rgba='.19 .25 .31 1')
    door = ET.SubElement(world, 'body', name='door', pos='.65 .53 .02')
    ET.SubElement(door, 'joint', name='door_hinge', type='hinge', axis='0 0 1',
                  range='-1.35 .04', limited='true', damping='2.5',
                  stiffness='8', springref='0', armature='.15')
    ET.SubElement(door, 'geom', name='door_panel', type='box', pos='0 -.45 .60',
                  size='.025 .45 .60', mass='8', contype='2', conaffinity='1',
                  friction='.8 .01 .001', rgba='.63 .42 .27 1')
    ET.SubElement(door, 'geom', name='door_rail', type='box', pos='0 -.45 1.17',
                  size='.033 .45 .035', mass='.3', contype='2', conaffinity='1',
                  rgba='.33 .24 .19 1')
    ET.SubElement(door, 'geom', name='door_push_plate', type='box', pos='.032 -.61 .56',
                  size='.008 .075 .10', mass='.1', contype='2', conaffinity='1',
                  rgba='.73 .75 .75 1')
    box = ET.SubElement(world, 'body', name='cargo', mocap='true', pos='0 -.15 .94')
    ET.SubElement(box, 'geom', name='cargo_box', type='box', pos='.04 0 0',
                  size='.115 .130 .105',
                  contype='2', conaffinity='1', rgba='.83 .52 .20 1')
    for y in [-.157, .157]:
        ET.SubElement(box, 'geom', name='cargo_grip_'+('left' if y>0 else 'right'),
                      type='box', pos=vec([-.015,y,0]), size='.025 .006 .055',
                      contype='0', conaffinity='0', rgba='.25 .22 .19 1')
    path = out/'scene.xml'
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='unicode')
    return mujoco.MjModel.from_xml_path(str(path))


def build_motion(out):
    ik = IK()
    times = np.arange(round(KEYS[-1][0]*FPS)+1)/FPS
    qs, bp, bq, sites, errors, centers, phases = [], [], [], [], [], [], []
    nominal = ik.nominal.copy()
    for i, ti in enumerate(times):
        root, rpy, targets, center, phase = target_at(ti)
        ik.nominal = nominal
        ik.max_joint_delta = None if i == 0 else .065
        q, err = ik.solve(root, rpy, targets,
                          dict(left_sole=[0,0,0], right_sole=[0,0,0],
                               right_palm=[.04,-.20,-.20]),
                          iterations=60 if i==0 else 22)
        qs.append(q)
        bp.append(ik.data.xpos[1:].copy())
        bq.append(ik.data.xquat[1:].copy())
        sites.append([ik.data.site_xpos[ik.sites[n]].copy() for n in targets])
        errors.append(list(err.values()))
        centers.append(center)
        phases.append(phase)
        if i%60 == 0:
            print(f'IK {i}/{len(times)} t={ti:.1f} max_error={max(err.values()):.4f}', flush=True)
    motion = dict(time=times, qpos=np.array(qs), body_pos=np.array(bp),
                  body_quat_wxyz=np.array(bq), site_pos=np.array(sites),
                  ik_error=np.array(errors), cargo_pos=np.array(centers),
                  phase=np.array(phases), fps=np.array(FPS),
                  joint_names=np.array(ik.joints), body_names=np.array(ik.body_names),
                  site_names=np.array(list(targets)))
    np.savez_compressed(out/'motion.npz', **motion)
    return ik, motion


def set_robot(model, data, motion, t):
    fps = float(motion['fps'])
    idx = min(int(t*fps+1e-7), len(motion['time'])-2)
    u = np.clip(t*fps-idx,0,1)
    n = len(motion['body_names'])
    data.mocap_pos[:n] = (1-u)*motion['body_pos'][idx]+u*motion['body_pos'][idx+1]
    quat = (1-u)*motion['body_quat_wxyz'][idx]+u*motion['body_quat_wxyz'][idx+1]
    data.mocap_quat[:n] = quat/np.linalg.norm(quat,axis=1,keepdims=True)
    data.mocap_pos[model.body('cargo').mocapid[0]] = (
        (1-u)*motion['cargo_pos'][idx]+u*motion['cargo_pos'][idx+1])


def simulate(model, motion, out, foot=True):
    if not foot:
        gid = model.geom('left_foot_collision').id
        model.geom_contype[gid] = model.geom_conaffinity[gid] = 0
    data = mujoco.MjData(model)
    hid = model.joint('door_hinge').qposadr[0]
    hvel = model.joint('door_hinge').dofadr[0]
    data.qpos[hid] = DOOR_OPEN
    duration = float(motion['time'][-1])
    nstep = round(duration/model.opt.timestep)
    leftid = model.geom('left_foot_collision').id
    panelids = {model.geom('door_panel').id, model.geom('door_rail').id,
                model.geom('door_push_plate').id}
    times, angles, velocities, foot_force, foot_contacts, sample_qpos = [], [], [], [], [], []
    worst = dict(penetration_m=0., time_s=None, geoms=[])
    furnishings = {model.geom(n).id for n in ['door_panel','door_rail',
        'door_push_plate','hinge_post','latch_post','lintel','cargo_box']}
    for step in range(nstep+1):
        t=step*model.opt.timestep
        set_robot(model,data,motion,t)
        # An opener holds the door until the foot reaches it; then the closer
        # acts alone. This is the same in foot and no-foot simulations.
        if t<1.55:
            data.qpos[hid]=DOOR_OPEN
            data.qvel[hvel]=0
        mujoco.mj_forward(model,data)
        normal=0.
        contact_count=0
        for ci in range(data.ncon):
            c=data.contact[ci]
            if c.dist < -worst['penetration_m'] and (
                c.geom1 in furnishings or c.geom2 in furnishings) and (
                '_proxy_' in model.geom(c.geom1).name or '_proxy_' in model.geom(c.geom2).name
                or 'foot_collision' in model.geom(c.geom1).name
                or 'foot_collision' in model.geom(c.geom2).name):
                worst=dict(penetration_m=-float(c.dist),time_s=t,
                           geoms=[model.geom(c.geom1).name,model.geom(c.geom2).name])
            if {c.geom1,c.geom2} & panelids and leftid in {c.geom1,c.geom2}:
                force=np.zeros(6)
                mujoco.mj_contactForce(model,data,ci,force)
                normal += abs(force[0])
                contact_count += 1
        times.append(t)
        angles.append(data.qpos[hid])
        velocities.append(data.qvel[hvel])
        foot_force.append(normal)
        foot_contacts.append(contact_count)
        sample_qpos.append(data.qpos[hid])
        if step<nstep:
            mujoco.mj_step(model,data)
    result=dict(time=np.array(times),door_angle=np.array(angles),
                door_velocity=np.array(velocities),foot_force=np.array(foot_force),
                foot_contact_count=np.array(foot_contacts),
                max_robot_object_penetration_m=np.array(worst['penetration_m']))
    np.savez_compressed(out/('simulation.npz' if foot else 'simulation_no_foot.npz'),**result)
    report=dict(foot_enabled=foot,peak_foot_force_N=float(np.max(foot_force)),
                foot_contact_start_s=float(times[np.argmax(np.array(foot_contacts)>0)])
                    if np.any(foot_contacts) else None,
                door_angle_at_4_3_rad=float(np.interp(4.3,times,angles)),
                door_angle_at_6_0_rad=float(np.interp(6.,times,angles)),
                final_door_angle_rad=float(angles[-1]),
                worst_robot_furniture_contact=worst)
    (out/('evaluation.json' if foot else 'evaluation_no_foot.json')).write_text(
        json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--motion-only',action='store_true')
    a=parser.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    ik,motion=build_motion(a.output)
    if a.motion_only:return
    model=make_scene(ik,a.output)
    simulate(model,motion,a.output,True)
    simulate(mujoco.MjModel.from_xml_path(str(a.output/'scene.xml')),
             motion,a.output,False)


if __name__=='__main__':
    main()
