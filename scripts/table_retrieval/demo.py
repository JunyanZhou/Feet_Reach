#!/usr/bin/env python3
"""Table-assisted retrieval: scripted G1 kinematics and a dynamic contact object.

The robot is kinematically prescribed. This is task choreography, not a learned
controller or a claim of dynamically balanced, load-bearing table support.
"""
from __future__ import annotations
import argparse, copy, json, os, sys, time
from pathlib import Path
import xml.etree.ElementTree as ET
if sys.platform.startswith('linux'):os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

ROOT=Path(__file__).resolve().parents[2]
MODEL=ROOT/'deploy/mujoco_whole_body/g1/g1_29dof_rev_1_0.xml'
OUT=ROOT/'artifacts/table_retrieval'

def vec(x): return ' '.join(f'{v:.9g}' for v in x)
def quat_euler(x):
    q=Rotation.from_euler('xyz',x).as_quat(); return q[[3,0,1,2]]
def smooth(u):
    u=np.clip(u,0,1); return u*u*u*(10+u*(-15+6*u))

def robot_xml():
    root=ET.parse(MODEL).getroot()
    root.find('compiler').set('meshdir',str(MODEL.parent/'meshes'))
    for tag in ('actuator','sensor','keyframe'):
        e=root.find(tag)
        if e is not None: root.remove(e)
    for g in root.findall('.//geom'):
        g.set('contype','0');g.set('conaffinity','0')
    for side in ['left','right']:
        b=root.find(f'.//body[@name="{side}_ankle_roll_link"]')
        ET.SubElement(b,'site',name=side+'_sole',pos='.035 0 -.035',size='.005',rgba='0 0 0 0')
        b=root.find(f'.//body[@name="{side}_wrist_yaw_link"]')
        ET.SubElement(b,'site',name=side+'_palm',pos=('.10 -.03 0' if side=='left' else '.09 0 -.018'),size='.005',rgba='0 0 0 0')
    return root

class IK:
    def __init__(self):
        self.xml=robot_xml();self.model=mujoco.MjModel.from_xml_string(ET.tostring(self.xml,encoding='unicode'))
        self.data=mujoco.MjData(self.model)
        self.joints=[self.model.joint(i).name for i in range(1,self.model.njnt)]
        self.limits=self.model.jnt_range[1:].copy()
        self.q=np.zeros(self.model.nq);self.q[:7]=[0,0,.76,1,0,0,0]
        for side in ['left','right']:
            for part,value in [('hip_pitch',-.15),('knee',.3),('ankle_pitch',-.15),('elbow',.5)]:
                self.q[7+self.joints.index(side+'_'+part+'_joint')]=value
        self.nominal=self.q[7:].copy()
        self.sites={n:self.model.site(n).id for n in ['left_sole','right_sole','left_palm','right_palm']}
        self.body_names=[self.model.body(i).name for i in range(1,self.model.nbody)]
        self.collision=None
        self.collision_enabled=False
        self.max_joint_delta=None
    def solve(self,root_pos,root_rpy,targets,orientations,iterations=30):
        self.q[:3]=root_pos;self.q[3:7]=quat_euler(root_rpy)
        jpos=np.zeros((3,self.model.nv));jrot=np.zeros_like(jpos)
        reg=np.ones(29)*.025;reg[12:15]=.5
        desired_rots={n:Rotation.from_euler('xyz',r).as_matrix() for n,r in orientations.items()}
        def evaluate(x, jac=False):
            self.data.qpos[:]=self.q;self.data.qpos[7:]=x;mujoco.mj_forward(self.model,self.data)
            rows=[];errors=[]
            for name,target in targets.items():
                sid=self.sites[name]
                errors.extend((self.data.site_xpos[sid]-np.array(target))*12)
                if jac:
                    mujoco.mj_jacSite(self.model,self.data,jpos,jrot,sid);rows.append(jpos[:,6:].copy()*12)
                if name in orientations:
                    current=self.data.site_xmat[sid].reshape(3,3)
                    error=Rotation.from_matrix(current@desired_rots[name].T).as_rotvec()
                    w=2.0 if 'sole' in name else 1.2
                    errors.extend(error*w)
                    if jac:
                        theta=np.linalg.norm(error)
                        ex,ey,ez=error;K=np.array([[0,-ez,ey],[ez,0,-ex],[-ey,ex,0]])
                        c=1/12 if theta<1e-4 else 1/theta**2-(1+np.cos(theta))/(2*theta*np.sin(theta))
                        inverse=np.eye(3)-.5*K+c*(K@K)
                        rows.append(inverse@jrot[:,6:]*w)
            if self.collision_enabled and self.collision is not None:
                ce,cj=self.collision.evaluate(self.data.qpos,jac)
                errors.extend(ce)
                if jac:rows.append(cj)
            errors.extend((x-self.nominal)*reg)
            if jac:rows.append(np.diag(reg));return np.concatenate(rows)
            return np.array(errors)
        lower=self.limits[:,0]+.002;upper=self.limits[:,1]-.002
        if self.max_joint_delta is not None:
            lower=np.maximum(lower,self.q[7:]-self.max_joint_delta);upper=np.minimum(upper,self.q[7:]+self.max_joint_delta)
        fit=least_squares(evaluate,self.q[7:],jac=lambda x:evaluate(x,True),bounds=(lower,upper),max_nfev=max(iterations,12),ftol=1e-7,xtol=1e-7,gtol=1e-7)
        self.q[7:]=fit.x
        self.data.qpos[:]=self.q;mujoco.mj_forward(self.model,self.data)
        err={n:float(np.linalg.norm(np.array(p)-self.data.site_xpos[self.sites[n]])) for n,p in targets.items()}
        return self.q.copy(),err


class CollisionAvoidance:
    def __init__(self,out):
        from audit import audit_model
        self.model=audit_model(out);self.data=mujoco.MjData(self.model);self.pairs=[]
        m=self.model
        for side in ['left','right']:
            arms=[side+'_elbow_link_g1',side+'_wrist_roll_link_g1',side+'_wrist_pitch_link_g1',side+'_wrist_yaw_link_g1',side+'_hand_collision']
            for arm in arms:
                for leg in ['left','right']:
                    for body in [leg+'_hip_yaw_link_g1',leg+'_knee_link_g1']:
                        self.pairs.append((m.geom(arm).id,m.geom(body).id))
                for other in ['table_top','object_collision','floor']:
                    self.pairs.append((m.geom(arm).id,m.geom(other).id))
            self.pairs.append((m.geom(side+'_shoulder_yaw_link_g1').id,m.geom('torso_link_g1').id))
            for corner in range(1,5):
                self.pairs.append((m.geom(side+'_ankle_roll_link_g'+str(corner)).id,m.geom('floor').id))
            for hip in ['hip_pitch','hip_roll','hip_yaw']:
                self.pairs.append((m.geom(side+'_'+hip+'_link_g1').id,m.geom('torso_link_g1').id))
    def evaluate(self,q,jac=False):
        m,d=self.model,self.data;d.qpos[:36]=q;d.qpos[36:]=[.135,.12,.04,1,0,0,0];mujoco.mj_forward(m,d)
        residual=[];rows=[];line=np.zeros(6);ja=np.zeros((3,m.nv));jb=np.zeros_like(ja)
        for a,b in self.pairs:
            if m.geom(b).name=='floor':
                if m.geom_type[a]==mujoco.mjtGeom.mjGEOM_SPHERE:
                    point=d.geom_xpos[a].copy();point[2]-=m.geom_size[a,0]
                else:
                    mid=m.geom_dataid[a];start=m.mesh_vertadr[mid];count=m.mesh_vertnum[mid]
                    vertices=m.mesh_vert[start:start+count]@d.geom_xmat[a].reshape(3,3).T+d.geom_xpos[a]
                    point=vertices[np.argmin(vertices[:,2])]
                distance=float(point[2]);weight=60; margin=0.0
                residual.append(max(margin-distance,0)*weight)
                if jac:
                    row=np.zeros(29)
                    if distance<margin:
                        mujoco.mj_jac(m,d,ja,None,point,m.geom_bodyid[a]);row=-weight*ja[2,6:35].copy()
                    rows.append(row)
                continue
            distance=mujoco.mj_geomDistance(m,d,a,b,.08,line)
            if abs(distance)<1e-10 and not any({int(c.geom1),int(c.geom2)}=={a,b} for c in d.contact):distance=.08
            foot_floor='ankle_roll' in m.geom(a).name and m.geom(b).name=='floor'
            margin=0.0 if foot_floor else .006;weight=60 if foot_floor else 30
            residual.append(max(margin-distance,0)*weight)
            if jac:
                row=np.zeros(29)
                if distance<margin and abs(distance)>1e-8:
                    normal=(line[3:]-line[:3])/abs(distance)
                    mujoco.mj_jac(m,d,ja,None,line[:3],m.geom_bodyid[a]);mujoco.mj_jac(m,d,jb,None,line[3:],m.geom_bodyid[b])
                    row=-weight*(normal@(jb-ja))[6:35]
                rows.append(row)
        return np.array(residual),np.array(rows) if jac else None

# World frame: +X faces the table, +Y is robot-left, +Z is up.
# Sole positions, palm contact point, and pelvis are independently choreographed.
KEYS=[
 # t, root xyz, root pitch, left sole xyz, left palm xyz, right palm xyz, phase
 (0,  [0,0,.75],0, [.035,.12,0], [.16,.23,.83], [.15,-.24,.83], 'Ready'),
 (1,  [0,0,.75],0, [.035,.12,0], [.16,.23,.83], [.15,-.24,.83], 'Ready'),
 (3,  [.04,-.025,.73],.04,[-.07,.12,0],[.32,.30,.770],[.17,-.27,.85],'Establish table support'),
 (4.5,[.09,-.065,.68],.08,[.10,.18,.13],[.32,.30,.770],[.18,-.28,.70],'Unload and lift left foot'),
 (6.5,[.18,-.035,.645],.10,[.57,.28,.09],[.32,.30,.770],[.20,-.29,.62],'Reach around the object'),
 (7.7,[.18,-.035,.645],.10,[.60,.12,.10],[.32,.30,.770],[.20,-.29,.62],'Place heel behind object'),
 (8.4,[.17,-.035,.645],.10,[.60,.12,.018],[.32,.30,.770],[.20,-.29,.62],'Lower heel behind object'),
 (9.0,[.16,-.035,.645],.10,[.545,.12,.018],[.32,.30,.770],[.20,-.29,.62],'Establish foot contact'),
 (12.5,[.07,-.045,.64],.08,[.27,.12,.018],[.32,.30,.770],[.18,-.29,.64],'Draw object out'),
 (13.3,[.06,-.05,.66],.07,[.27,.12,.16],[.32,.30,.770],[.18,-.29,.66],'Release object'),
 (14.8,[-.08,-.015,.68],.06,[-.20,.14,.13],[.32,.30,.770],[.10,-.25,.80],'Recover left foot'),
 (15.8,[-.08,0,.70],.08,[-.20,.14,0],[.32,.30,.770],[.10,-.24,.80],'Restore double support'),
 (17.0,[-.10,.015,.70],.08,[-.20,.14,0],[.32,.30,.770],[.10,-.24,.80],'Step back for hand access',[.08,-.14,.12]),
 (18.5,[-.12,.015,.69],.08,[-.20,.14,0],[.32,.30,.770],[.10,-.24,.80],'Step back for hand access',[-.20,-.14,.10]),
 (19.5,[-.12,0,.70],.08,[-.20,.14,0],[.32,.30,.770],[.10,-.24,.80],'Establish a clear reaching stance',[-.20,-.14,0]),
 (20.7,[-.15,0,.69],.10,[-.20,.14,0],[.19,.24,.82],[.06,-.24,.77],'Release table support',[-.20,-.14,0]),
 (23.7,[-.26754974,-.04554613,.32247303],.90053561,[-.20,.14,0],[.135,.12,.120],[0,-.28,.43],'Reach to the retrieved object',[-.20,-.14,0]),
 (24.7,[-.26754974,-.04554613,.32247303],.90053561,[-.20,.14,0],[.135,.12,.120],[0,-.28,.43],'Object within hand reach',[-.20,-.14,0]),
 (27.7,[-.22,0,.72],.02,[-.20,.14,0],[-.03,.23,.78],[-.03,-.24,.78],'Return to standing',[-.20,-.14,0]),
 (29.0,[-.22,0,.72],.02,[-.20,.14,0],[-.03,.23,.78],[-.03,-.24,.78],'Retrieval complete',[-.20,-.14,0]),
]

def target_at(t):
    times=np.array([k[0] for k in KEYS]);i=min(max(np.searchsorted(times,t,side='right')-1,0),len(KEYS)-2)
    a,b=KEYS[i],KEYS[i+1];u=smooth((t-a[0])/(b[0]-a[0]))
    interp=lambda idx: (1-u)*np.array(a[idx])+u*np.array(b[idx])
    root=interp(1);rpy=[0,float(interp(2)),0]
    targets={'left_sole':interp(3),'right_sole':(1-u)*np.array(a[7] if len(a)>7 else [.12,-.12,0.])+u*np.array(b[7] if len(b)>7 else [.12,-.12,0.]),'left_palm':interp(4),'right_palm':interp(5)}
    orientations={'left_sole':[0,0,0],'right_sole':[0,0,0],'left_palm':[np.pi/2,-.2,0]}
    recovery_pitch=.45*float(smooth((t-13.3)/1.5)) if t<14.8 else .45*(1-float(smooth((t-14.8)/1.0)))
    swing_roll=.10*float(smooth((t-4.5)/2.0)) if t<6.5 else .10*(1-float(smooth((t-6.5)/1.2)))
    orientations['left_sole']=[swing_roll,recovery_pitch,0]
    right_pitch=.3*float(smooth((t-17)/1.5)) if t<18.5 else .3*(1-float(smooth((t-18.5)/1.0)))
    orientations['right_sole']=[0,right_pitch,0]
    hand_blend=float(smooth((t-20.7)/3.0)) if t<24.7 else 1-float(smooth((t-24.7)/3.0))
    root[0]-=.055*np.sin(np.pi*hand_blend)**2
    orientations['left_palm']=(1-hand_blend)*np.array([np.pi/2,-.2,0])+hand_blend*np.array([1.497711264656079,-.13939302293633649,-.09975584713399199])
    return root,rpy,targets,orientations,b[6] if t>a[0] else a[6]

def make_scene(ik,out):
    root=ET.Element('mujoco',model='table_assisted_retrieval')
    ET.SubElement(root,'compiler',angle='radian',meshdir=str(MODEL.parent/'meshes'))
    ET.SubElement(root,'option',timestep='.002',gravity='0 0 -9.81',integrator='implicitfast',iterations='80',cone='elliptic')
    visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',offwidth='1920',offheight='1080',azimuth='140',elevation='-18')
    ET.SubElement(visual,'quality',shadowsize='4096',offsamples='4')
    ET.SubElement(visual,'headlight',ambient='.32 .32 .32',diffuse='.55 .55 .55',specular='.2 .2 .2')
    ET.SubElement(visual,'map',znear='.01',zfar='30',fogstart='8',fogend='20')
    asset=copy.deepcopy(ik.xml.find('asset'));root.append(asset)
    ET.SubElement(asset,'texture',name='sky',type='skybox',builtin='gradient',rgb1='.22 .28 .36',rgb2='.065 .09 .13',width='512',height='3072')
    ET.SubElement(asset,'texture',name='floor_tex',type='2d',builtin='checker',rgb1='.19 .23 .27',rgb2='.22 .26 .30',width='512',height='512')
    ET.SubElement(asset,'material',name='floor_mat',texture='floor_tex',texrepeat='4 4',reflectance='0',shininess='.3')
    world=ET.SubElement(root,'worldbody')
    ET.SubElement(world,'light',name='key',pos='-1 -2 4',dir='.3 .4 -1',diffuse='.85 .85 .82',castshadow='true',directional='true')
    ET.SubElement(world,'light',name='fill',pos='2 2 3',dir='-.3 -.3 -1',diffuse='.5 .6 .7',castshadow='false',directional='true')
    ET.SubElement(world,'geom',name='floor',type='plane',size='5 5 .05',material='floor_mat',friction='.55 .01 .001',contype='1',conaffinity='2')
    def box(name,pos,size,rgba,collision=True):
        return ET.SubElement(world,'geom',name=name,type='box',pos=vec(pos),size=vec(size),rgba=rgba,contype='1' if collision else '0',conaffinity='2' if collision else '0')
    box('table_top',[.78,0,.745],[.50,.53,.025],'.58 .38 .21 1')
    # Thin edge band and four legs keep the under-table action unobstructed.
    for x in [.35,1.21]:
        for y in [-.46,.46]:
            box('table_leg_'+str(x)+'_'+str(y),[x,y,.36],[.025,.025,.36],'.13 .17 .21 1')
    box('table_apron_back',[1.18,0,.665],[.025,.47,.055],'.18 .23 .28 1')
    # Destination marking is decorative and does not alter object physics.
    box('retrieval_zone',[.135,.12,.0008],[.105,.12,.0008],'.15 .62 .51 .42',False)
    for y in [-.003,.243]:box('zone_edge_'+str(y),[.135,y,.002],[.108,.003,.001],'.3 .88 .72 1',False)
    for x in [.027,.243]:box('zone_edge_x_'+str(x),[x,.12,.002],[.003,.123,.001],'.3 .88 .72 1',False)
    # Each original body is a mocap body: prescribed motion, no autonomous dynamics.
    for src in ik.xml.findall('.//body'):
        body=ET.SubElement(world,'body',name=src.get('name'),mocap='true',pos='0 0 0')
        geoms=src.findall('geom')
        visual_meshes=[g for g in geoms if g.get('type')=='mesh' and g.get('group')=='1']
        for index,g in enumerate(visual_meshes):
            g=copy.deepcopy(g);g.set('name',src.get('name')+'_visual_'+str(index));g.set('contype','0');g.set('conaffinity','0');g.set('group','1')
            body.append(g)
        if src.get('name') in ['left_ankle_roll_link','right_ankle_roll_link']:
            ET.SubElement(body,'geom',name=src.get('name')+'_collision',type='box',pos='.035 0 -.018',size='.09 .038 .018',rgba='0 0 0 0',group='3',contype='1',conaffinity='2',friction='.65 .01 .001',solref='.008 1',solimp='.95 .99 .001')
        if src.get('name')=='left_wrist_yaw_link':
            ET.SubElement(body,'geom',name='support_palm',type='box',pos='.10 -.028 0',size='.025 .002 .025',rgba='0 0 0 0',group='3',contype='1',conaffinity='2')
    obj=ET.SubElement(world,'body',name='object',pos='.445 .12 .041')
    ET.SubElement(obj,'freejoint',name='object_free')
    ET.SubElement(obj,'geom',name='object_collision',type='box',size='.045 .045 .04',mass='.22',rgba='1 .38 .055 1',friction='.55 .008 .001',contype='2',conaffinity='1',solref='.008 1')
    ET.SubElement(obj,'geom',name='object_stripe',type='box',pos='0 0 .0402',size='.008 .044 .0003',rgba='.98 .86 .61 1',contype='0',conaffinity='0',mass='0')
    path=out/'scene.xml';ET.indent(root);ET.ElementTree(root).write(path,encoding='unicode')
    return mujoco.MjModel.from_xml_path(str(path))

def build_motion(out,fps=60,use_recovery=True):
    ik=IK();make_scene(ik,out);ik.collision=CollisionAvoidance(out)
    reach_nominal=np.array([-2.404714087036439, -0.09910366577293214, -0.27454695861965456, 2.3665297051672938, -0.8407506398909547, -0.25879992662746304, -2.4500012791904653, -0.014854086667911654, -0.09568099462691754, 2.390724055297459, -0.8405486383111535, -0.09682251229468518, 0.01480528471168438, -0.17443840421182008, -0.04999356198609149, -0.9369688767997968, 0.20763442961621845, -0.0615755184823686, 1.25000796154557, 1.5143301125737945, -0.07716159559666025, 1.307953844213174, 0.3941185777933557, 0.05292588067980887, -0.8266512152134446, -0.3219935388419508, -0.1985409750321896, -1.2500457346184142, -0.11159429611579595])
    default_nominal=ik.nominal.copy()
    recovery=dict(np.load(Path(__file__).with_name('validated_recovery.npz')))
    if not np.array_equal(recovery['joint_names'],np.array(ik.joints)):raise ValueError('Recovery joint order mismatch')
    times=np.arange(round(KEYS[-1][0]*fps)+1)/fps
    qs=[];bp=[];bq=[];errors=[];phases=[];sites=[]
    start=time.time()
    for i,t in enumerate(times):
        pos,rpy,targets,orientations,phase=target_at(t)
        blend=float(smooth((t-20.7)/3)) if t<24.7 else 1-float(smooth((t-24.7)/3))
        ik.nominal=(1-blend)*default_nominal+blend*reach_nominal
        ik.collision_enabled=t>19.5
        ik.max_joint_delta=None if i==0 else .04
        if use_recovery and t>=float(recovery['time_s'][0]):
            # Freeze the collision-audited recovery branch. Tiny floating-point
            # changes in a redundant IK solve can otherwise select another
            # elbow branch; replay is the explicit goal of this demo.
            ri=min(round((t-float(recovery['time_s'][0]))*float(recovery['fps'])),len(recovery['qpos'])-1)
            q=recovery['qpos'][ri].copy();ik.q=q.copy();ik.data.qpos[:]=q;mujoco.mj_forward(ik.model,ik.data)
            err={n:float(np.linalg.norm(np.array(p)-ik.data.site_xpos[ik.sites[n]])) for n,p in targets.items()}
        else:
            q,err=ik.solve(pos,rpy,targets,orientations,60 if i==0 else 25)
        qs.append(q);bp.append(ik.data.xpos[1:].copy());bq.append(ik.data.xquat[1:].copy());errors.append(list(err.values()));phases.append(phase)
        sites.append([ik.data.site_xpos[ik.sites[n]].copy() for n in targets])
        if i%180==0:print(f'IK {i}/{len(times)} t={t:.1f} max error={max(err.values()):.4f} elapsed={time.time()-start:.1f}s',flush=True)
    data=dict(time=times,qpos=np.array(qs),body_pos=np.array(bp),body_quat_wxyz=np.array(bq),joint_names=np.array(ik.joints),body_names=np.array(ik.body_names),site_names=np.array(list(targets)),site_pos=np.array(sites),ik_error=np.array(errors),phase=np.array(phases),fps=np.array(fps))
    np.savez_compressed(out/'motion.npz',**data)
    return ik,data

def set_robot(model,data,motion,t):
    fps=float(motion['fps']);idx=min(int(t*fps+1e-7),len(motion['time'])-2);u=np.clip(t*fps-idx,0,1)
    # make_scene inserts mocap bodies in exactly the saved FK body order.
    # Vectorized assignment avoids repeated named-body lookups in each 2 ms step.
    data.mocap_pos[:]=(1-u)*motion['body_pos'][idx]+u*motion['body_pos'][idx+1]
    q=(1-u)*motion['body_quat_wxyz'][idx]+u*motion['body_quat_wxyz'][idx+1]
    data.mocap_quat[:]=q/np.linalg.norm(q,axis=1,keepdims=True)

def simulate(model,motion,out,object_offset=(0,0),friction=.55,tag=''):
    data=mujoco.MjData(model);data.qpos[:3]+=[object_offset[0],object_offset[1],0]
    model.geom_friction[model.geom('object_collision').id,0]=friction
    model.geom_friction[model.geom('floor').id,0]=friction
    set_robot(model,data,motion,0);mujoco.mj_forward(model,data)
    objid=model.geom('object_collision').id;footid=model.geom('left_ankle_roll_link_collision').id
    records=[];forces=[];contacts=[];nstep=round(motion['time'][-1]/model.opt.timestep)
    next_sample=0;sample_step=1/float(motion['fps']);max_pen=0.;impulse=0.;peak_force=0.;contact_times=[]
    for step in range(nstep+1):
        t=step*model.opt.timestep;set_robot(model,data,motion,t);mujoco.mj_forward(model,data)
        force=0.;ncontact=0
        for ci in range(data.ncon):
            c=data.contact[ci]
            if {c.geom1,c.geom2}=={objid,footid}:
                f=np.zeros(6);mujoco.mj_contactForce(model,data,ci,f);force+=f[0];ncontact+=1;max_pen=max(max_pen,-float(c.dist))
        if ncontact:contact_times.append(t)
        impulse+=force*model.opt.timestep;peak_force=max(peak_force,force)
        if t+1e-8>=next_sample:
            records.append(data.qpos.copy());forces.append(force);contacts.append(ncontact);next_sample+=sample_step
        if step<nstep:mujoco.mj_step(model,data)
    obj=np.array(records);end=obj[-1,:3];start=obj[0,:3]
    qend=obj[-1,3:7];rot=Rotation.from_quat(qend[[1,2,3,0]]).as_matrix();extent=abs(rot)@np.array([.045,.045,.04])
    lower=end-extent;upper=end+extent
    summary={'mode':'scripted_kinematic_robot_dynamic_object','autonomous_balance_validated':False,'table_support_load_validated':False,'object_motion':'MuJoCo rigid-body contact integration; no object pose overwrite during simulation',
      'duration_s':float(motion['time'][-1]),'object_initial_xyz_m':start.tolist(),'object_final_xyz_m':end.tolist(),'object_outward_displacement_m':float(start[0]-end[0]),
      'object_entirely_outside_table':bool(upper[0]<.28),'object_in_retrieval_zone':bool(lower[0]>=.027 and upper[0]<=.243 and lower[1]>=-.003 and upper[1]<=.243),'object_upright':bool(rot[2,2]>np.cos(np.deg2rad(10))),
      'foot_object_peak_normal_force_N':float(peak_force),'foot_object_normal_impulse_Ns':float(impulse),'foot_object_max_penetration_m':max_pen,
      'foot_contact_start_s':float(contact_times[0]) if contact_times else None,'foot_contact_end_s':float(contact_times[-1]) if contact_times else None,
      'ik_max_position_error_m':dict(zip(map(str,motion['site_names']),np.max(motion['ik_error'],axis=0).tolist())),
      'object_offset_xy_m':list(object_offset),'object_friction':friction}
    summary['task_success']=summary['object_entirely_outside_table'] and summary['object_in_retrieval_zone'] and summary['object_upright'] and bool(contact_times)
    np.savez_compressed(out/('simulation'+tag+'.npz'),time=motion['time'][:len(obj)],object_qpos=obj,foot_object_force=np.array(forces),foot_object_contacts=np.array(contacts))
    (out/('evaluation'+tag+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    return dict(object_qpos=obj,foot_object_force=np.array(forces),foot_object_contacts=np.array(contacts)),summary

def camera_for(view):
    cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam)
    if view=='detail':cam.lookat[:]=[.32,.12,.22];cam.distance=1.6;cam.azimuth=330;cam.elevation=-10
    elif view=='side':cam.lookat[:]=[.4,0,.64];cam.distance=2.8;cam.azimuth=95;cam.elevation=-12
    else:cam.lookat[:]=[.40,0,.64];cam.distance=2.85;cam.azimuth=315;cam.elevation=-18
    return cam

def render(model,motion,sim,out,width=1280,height=720,fps=30,view='overview',stills=False,start_s=0.,end_s=None,clean_only=False):
    data=mujoco.MjData(model);renderer=mujoco.Renderer(model,height,width);cam=camera_for(view)
    opt=mujoco.MjvOption();opt.geomgroup[3]=0
    candidates=['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','C:/Windows/Fonts/arial.ttf','/System/Library/Fonts/Supplemental/Arial.ttf']
    fontpath=next((p for p in candidates if Path(p).is_file()),None)
    getfont=lambda size:ImageFont.truetype(fontpath,size) if fontpath else ImageFont.load_default()
    font=getfont(round(width/53));small=getfont(round(width/75));title=getfont(round(width/42))
    output=out/(view+'.mp4')
    writer=None if clean_only else imageio.get_writer(output,fps=fps,codec='libx264',quality=8,macro_block_size=1,ffmpeg_params=['-threads','4'])
    clean_writer=imageio.get_writer(out/(view+'_clean.mp4'),fps=fps,codec='libx264',quality=9,macro_block_size=1,ffmpeg_params=['-threads','4'])
    end_s=float(motion['time'][-1]) if end_s is None else min(end_s,float(motion['time'][-1]))
    nframes=round((end_s-start_s)*fps)+1
    shot_times=[0,3,6.5,8.7,11.5,13,16.5,19.5,23.7,24.7,29]
    try:
        for frame in range(nframes):
            t=start_s+frame/fps;i=min(round(t*float(motion['fps'])),len(sim['object_qpos'])-1)
            set_robot(model,data,motion,t);data.qpos[:]=sim['object_qpos'][i];mujoco.mj_forward(model,data)
            renderer.update_scene(data,cam,scene_option=opt)
            raw=renderer.render();clean_writer.append_data(raw)
            if not clean_only:
                pic=Image.fromarray(raw);draw=ImageDraw.Draw(pic,'RGBA')
                draw.rectangle((0,0,width,round(height*.135)),fill=(12,20,30,235))
                draw.text((24,10),'TABLE-ASSISTED FOOT RETRIEVAL',font=title,fill=(240,245,250))
                phase=str(motion['phase'][i]);draw.text((25,round(height*.073)),phase,font=font,fill=(89,222,189))
                draw.text((width-24,round(height*.077)),f'{t:04.1f} / {motion["time"][-1]:.0f} s',font=small,fill='white',anchor='ra')
                draw.rectangle((0,height-46,width,height),fill=(12,20,30,225))
                draw.text((22,height-33),'SCRIPTED KINEMATICS  /  PHYSICAL OBJECT CONTACT',font=small,fill=(193,204,216))
                displacement=sim['object_qpos'][0,0]-data.qpos[0]
                draw.text((width-24,height-33),f'Object retrieved: {max(0,displacement)*100:4.1f} cm',font=small,fill=(255,185,89),anchor='ra')
                writer.append_data(np.array(pic))
                if stills and any(abs(t-s)<=.5/fps for s in shot_times):pic.save(out/f'{view}_{t:05.2f}.png')
            if frame%120==0:print(f'Render {view} {frame}/{nframes}',flush=True)
    finally:
        if writer is not None:writer.close()
        clean_writer.close();renderer.close()
    return out/(view+'_clean.mp4') if clean_only else output

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--reuse-motion',action='store_true');p.add_argument('--resolve-recovery',action='store_true',help='Experimental: re-solve the cached recovery; run audit before use');p.add_argument('--render',action='store_true');p.add_argument('--view',default='overview');p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=720);p.add_argument('--fps',type=int,default=30);p.add_argument('--stills',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.reuse_motion:ik=IK();motion=dict(np.load(a.output/'motion.npz'))
    else:ik,motion=build_motion(a.output,use_recovery=not a.resolve_recovery)
    model=make_scene(ik,a.output);sim,summary=simulate(model,motion,a.output)
    if a.render:render(model,motion,sim,a.output,a.width,a.height,a.fps,a.view,a.stills)
if __name__=='__main__':main()
