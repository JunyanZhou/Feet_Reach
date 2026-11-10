#!/usr/bin/env python3
"""Scripted forward-leaning G1 pushes a free box using physical palm contacts."""
from __future__ import annotations
import argparse
import copy
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/table_retrieval'))
from demo import IK, MODEL, smooth, vec
OUT=ROOT/'artifacts/push_box_support'
FPS=60
# time, root xyz, forward pitch, left sole, right sole, palm x, phase
KEYS=[
    (0.0,[-.32,0,.72],.03,[-.32,.15,0],[-.22,-.15,0],.07,'approach'),
    (.8,[-.29,0,.71],.20,[-.32,.15,0],[-.22,-.15,0],.15,'hands approach box'),
    (1.5,[-.18,0,.69],.38,[-.32,.15,0],[-.22,-.15,0],.20,'both palms contact'),
    (2.5,[-.03,0,.68],.46,[-.32,.15,0],[-.22,-.15,0],.34,'push begins'),
    (3.1,[.03,0,.68],.47,[-.17,.15,.06],[-.22,-.15,0],.43,'left step swing'),
    (3.8,[.12,0,.69],.46,[-.02,.15,0],[-.22,-.15,0],.55,'left foot lands'),
    (4.4,[.20,0,.69],.47,[-.02,.15,0],[-.13,-.15,.05],.64,'right step swing'),
    (5.0,[.31,0,.69],.46,[-.02,.15,0],[.15,-.15,0],.76,'right foot lands'),
    (5.7,[.40,0,.69],.47,[.17,.15,.06],[.15,-.15,0],.87,'left step swing'),
    (6.4,[.50,0,.69],.46,[.36,.15,0],[.15,-.15,0],.99,'left foot lands'),
    (7.1,[.60,0,.69],.47,[.36,.15,0],[.37,-.15,.06],1.10,'right step swing'),
    (7.8,[.70,0,.69],.45,[.36,.15,0],[.56,-.15,0],1.22,'right foot lands'),
    (8.6,[.80,0,.71],.36,[.72,.15,0],[.56,-.15,0],1.30,'finish push'),
    (9.2,[.82,0,.74],.08,[.72,.15,0],[.56,-.15,0],1.30,'release and stand'),
]
PALM_SPLINE=PchipInterpolator([k[0] for k in KEYS],[k[5] for k in KEYS])


def target_at(t):
    times=[k[0] for k in KEYS]
    j=min(max(np.searchsorted(times,t,side='right')-1,0),len(KEYS)-2)
    a,b=KEYS[j],KEYS[j+1]
    u=float(smooth((t-a[0])/(b[0]-a[0])))
    lerp=lambda i:(1-u)*np.array(a[i])+u*np.array(b[i])
    x=float(PALM_SPLINE(t))
    targets=dict(left_sole=lerp(3),right_sole=lerp(4),
                 left_palm=[x,.225,.73],right_palm=[x,-.225,.73])
    return lerp(1),[0,float(lerp(2)),0],targets,b[6]


def make_scene(ik,out):
    root=ET.Element('mujoco',model='two_palm_free_box_push')
    ET.SubElement(root,'compiler',angle='radian',meshdir=str(MODEL.parent/'meshes'))
    ET.SubElement(root,'option',timestep='.002',gravity='0 0 -9.81',
                  integrator='implicitfast',iterations='100',cone='elliptic')
    visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',offwidth='1920',offheight='1080')
    ET.SubElement(visual,'quality',shadowsize='4096',offsamples='4')
    ET.SubElement(visual,'headlight',ambient='.34 .34 .34',
                  diffuse='.58 .58 .58',specular='.2 .2 .2')
    asset=copy.deepcopy(ik.xml.find('asset'));root.append(asset)
    ET.SubElement(asset,'texture',name='sky',type='skybox',builtin='gradient',
                  rgb1='.22 .28 .36',rgb2='.065 .09 .13',width='512',height='3072')
    ET.SubElement(asset,'texture',name='floor_tex',type='2d',builtin='checker',
                  rgb1='.19 .23 .27',rgb2='.22 .26 .30',width='512',height='512')
    ET.SubElement(asset,'material',name='floor_mat',texture='floor_tex',
                  texrepeat='4 4',reflectance='0')
    world=ET.SubElement(root,'worldbody')
    ET.SubElement(world,'light',name='key',pos='-1 -2 4',dir='.3 .4 -1',
                  diffuse='.85 .85 .82',directional='true',castshadow='true')
    ET.SubElement(world,'light',name='fill',pos='2 2 3',dir='-.3 -.3 -1',
                  diffuse='.5 .6 .7',directional='true',castshadow='false')
    ET.SubElement(world,'geom',name='floor',type='plane',size='5 5 .05',
                  material='floor_mat',friction='.28 .01 .001',
                  contype='3',conaffinity='3')
    for src in ik.xml.findall('.//body'):
        name=src.get('name')
        body=ET.SubElement(world,'body',name=name,mocap='true')
        if name.startswith('viz_'):
            continue
        for j,original in enumerate(src.findall('geom')):
            if original.get('type')=='mesh' and original.get('group')=='1':
                geom=copy.deepcopy(original)
                geom.set('name',name+'_visual_'+str(j))
                geom.set('contype','0');geom.set('conaffinity','0')
                body.append(geom)
            if name in ('left_ankle_roll_link','right_ankle_roll_link'):
                continue
            is_hand=original.get('mesh','').endswith('_rubber_hand')
            if original.get('group')=='1' and not is_hand:
                continue
            proxy=copy.deepcopy(original)
            proxy.set('name',name+'_proxy_'+str(j))
            # Palm spheres are the load-bearing contact. The finger meshes
            # are still included for an explicit distance audit.
            proxy.set('contype','0' if is_hand else '1')
            proxy.set('conaffinity','0' if is_hand else '2')
            proxy.set('group','3');proxy.set('rgba','0 0 0 0')
            proxy.set('density','0')
            body.append(proxy)
        if name in ('left_ankle_roll_link','right_ankle_roll_link'):
            side=name.split('_')[0]
            ET.SubElement(body,'geom',name=side+'_foot_collision',type='box',
                          pos='.035 0 -.018',size='.09 .038 .020',
                          rgba='0 0 0 0',group='3',contype='1',conaffinity='2',
                          friction='1.2 .01 .001',solref='.04 1')
        if name in ('left_wrist_yaw_link','right_wrist_yaw_link'):
            side=name.split('_')[0]
            pos='.15 -.03 0' if side=='left' else '.14 0 -.018'
            ET.SubElement(body,'geom',name=side+'_push_pad',type='sphere',
                          pos=pos,size='.035',rgba='0 0 0 0',group='3',
                          contype='1',conaffinity='2',friction='1 .005 .0005',
                          solref='.006 1',solimp='.9 .99 .001')
    box=ET.SubElement(world,'body',name='box',pos='.535 0 .455')
    ET.SubElement(box,'freejoint',name='box_free')
    ET.SubElement(box,'geom',name='box_collision',type='box',
                  size='.25 .42 .45',mass='30',friction='.28 .005 .0005',
                  contype='2',conaffinity='3',solref='.006 1',
                  solimp='.9 .99 .001',rgba='.68 .44 .22 1')
    for z in [.13,.77]:
        ET.SubElement(box,'geom',name='box_band_'+str(z),type='box',
                      pos=vec([0,0,z-.455]),size='.253 .423 .015',
                      mass='0',contype='0',conaffinity='0',rgba='.36 .25 .17 1')
    path=out/'scene.xml';ET.indent(root)
    ET.ElementTree(root).write(path,encoding='unicode')
    return mujoco.MjModel.from_xml_path(str(path))


def build_motion(out):
    ik=IK();times=np.arange(round(KEYS[-1][0]*FPS)+1)/FPS
    qs,bp,bq,sites,errors,phases=[],[],[],[],[],[]
    nominal=ik.nominal.copy()
    for i,t in enumerate(times):
        root,rpy,targets,phase=target_at(t)
        ik.nominal=nominal
        ik.max_joint_delta=None if i==0 else .065
        q,err=ik.solve(root,rpy,targets,
            dict(left_sole=[0,0,0],right_sole=[0,0,0],
                 left_palm=[0,.1,0],right_palm=[0,-.1,0]),
            iterations=60 if i==0 else 22)
        qs.append(q);bp.append(ik.data.xpos[1:].copy())
        bq.append(ik.data.xquat[1:].copy())
        sites.append([ik.data.site_xpos[ik.sites[n]].copy() for n in targets])
        errors.append(list(err.values()));phases.append(phase)
        if i%60==0:
            print(f'IK {i}/{len(times)} t={t:.1f} max_error={max(err.values()):.4f}',flush=True)
    motion=dict(time=times,qpos=np.array(qs),body_pos=np.array(bp),
                body_quat_wxyz=np.array(bq),site_pos=np.array(sites),
                ik_error=np.array(errors),phase=np.array(phases),
                fps=np.array(FPS),joint_names=np.array(ik.joints),
                body_names=np.array(ik.body_names),site_names=np.array(list(targets)))
    np.savez_compressed(out/'motion.npz',**motion)
    return ik,motion


def set_robot(model,data,motion,t):
    fps=float(motion['fps'])
    idx=min(int(t*fps+1e-7),len(motion['time'])-2)
    u=np.clip(t*fps-idx,0,1)
    n=len(motion['body_names'])
    data.mocap_pos[:n]=(1-u)*motion['body_pos'][idx]+u*motion['body_pos'][idx+1]
    q=(1-u)*motion['body_quat_wxyz'][idx]+u*motion['body_quat_wxyz'][idx+1]
    data.mocap_quat[:n]=q/np.linalg.norm(q,axis=1,keepdims=True)


def simulate(model,motion,out,hands=True):
    if not hands:
        for i in range(model.ngeom):
            n=model.geom(i).name
            if '_proxy_' in n or n.endswith('_push_pad') or n.endswith('_foot_collision'):
                model.geom_contype[i]=model.geom_conaffinity[i]=0
    data=mujoco.MjData(model)
    duration=float(motion['time'][-1]);dt=model.opt.timestep
    nstep=round(duration/dt)
    boxid=model.geom('box_collision').id
    pads={model.geom(n).id for n in ('left_push_pad','right_push_pad')}
    samples=[];times=[];forces=[];counts=[]
    worst=dict(penetration_m=0.,time_s=None,geoms=[])
    for step in range(nstep+1):
        t=step*dt
        set_robot(model,data,motion,t)
        mujoco.mj_forward(model,data)
        force=0.;count=0
        for ci in range(data.ncon):
            c=data.contact[ci]
            pair={c.geom1,c.geom2}
            if boxid in pair and any(g in pads for g in pair):
                f=np.zeros(6);mujoco.mj_contactForce(model,data,ci,f)
                force+=abs(f[0]);count+=1
            if boxid in pair and c.dist < -worst['penetration_m']:
                other=(pair-{boxid}).pop()
                if other!=model.geom('floor').id:
                    worst=dict(penetration_m=-float(c.dist),time_s=t,
                               geoms=['box_collision',model.geom(other).name])
        samples.append(data.qpos[:7].copy())
        times.append(t);forces.append(force);counts.append(count)
        if step<nstep:mujoco.mj_step(model,data)
    result=dict(time=np.array(times),box_qpos=np.array(samples),
                push_force_N=np.array(forces),push_contact_count=np.array(counts),
                max_box_robot_contact_penetration_m=np.array(worst['penetration_m']))
    np.savez_compressed(out/('simulation.npz' if hands else 'simulation_no_hands.npz'),**result)
    idx=np.flatnonzero(np.array(counts)>0)
    report=dict(hands_enabled=hands,
                box_displacement_x_m=float(samples[-1][0]-samples[0][0]),
                box_final_y_m=float(samples[-1][1]),
                peak_palm_contact_force_N=float(max(forces)),
                palm_contact_start_s=float(times[idx[0]]) if len(idx) else None,
                palm_contact_end_s=float(times[idx[-1]]) if len(idx) else None,
                max_box_robot_contact_penetration=worst)
    (out/('evaluation.json' if hands else 'evaluation_no_hands.json')).write_text(
        json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    ik,motion=build_motion(a.output)
    model=make_scene(ik,a.output)
    simulate(model,motion,a.output,True)
    simulate(mujoco.MjModel.from_xml_path(str(a.output/'scene.xml')),
             motion,a.output,False)


if __name__=='__main__':main()
