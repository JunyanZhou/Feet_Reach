"""Solve a hand-access demonstration pose with convex collision clearance."""
import json
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from demo import IK,OUT,quat_euler
from audit import audit_model
ik=IK();m=audit_model();d=mujoco.MjData(m)
# Move left foot laterally to leave the reaching arm a clear path.
targets={'left_sole':[-.20,.14,0],'right_sole':[-.20,-.14,0],'left_palm':[.135,.12,.12],'right_palm':[0,-.28,.43]}
left_ori=Rotation.from_euler('xyz',[np.pi/2,-.2,0]).as_matrix()
pairs=[]
for side in ['left','right']:
 arms=[side+'_elbow_link_g1',side+'_wrist_roll_link_g1',side+'_wrist_pitch_link_g1',side+'_wrist_yaw_link_g1',side+'_hand_collision']
 for a in arms:
  for leg in ['left','right']:
   for b in [leg+'_hip_yaw_link_g1',leg+'_knee_link_g1']:
    pairs.append((m.geom(a).id,m.geom(b).id))
  for b in ['table_top','object_collision','floor']:
   pairs.append((m.geom(a).id,m.geom(b).id))
# Upper arm and chest clearance.
for side in ['left','right']:
 pairs.append((m.geom(side+'_shoulder_yaw_link_g1').id,m.geom('torso_link_g1').id))
line=np.zeros(6)
def fun(x):
 q=np.r_[x[:3],quat_euler([0,x[3],0]),x[4:]]
 ik.data.qpos[:]=q;mujoco.mj_forward(ik.model,ik.data)
 d.qpos[:36]=q;d.qpos[36:]=[.135,.12,.04,1,0,0,0];mujoco.mj_forward(m,d)
 e=[]
 for name,target in targets.items():
  sid=ik.sites[name];e.extend((ik.data.site_xpos[sid]-target)*(45 if name!='right_palm' else 8))
  current=ik.data.site_xmat[sid].reshape(3,3)
  if 'sole' in name:e.extend(Rotation.from_matrix(current).as_rotvec()*6)
  if name=='left_palm':e.extend(Rotation.from_matrix(current@left_ori.T).as_rotvec()*2)
 for a,b in pairs:
  distance=mujoco.mj_geomDistance(m,d,a,b,.08,line)
  e.append(max(.006-distance,0)*30)
 e.extend((x[:4]-[-.32,0,.40,1.0])*.1)
 e.extend(x[16:19]*.003)
 e.extend((x[4:]-ik.nominal)*.0003)
 return np.array(e)
q0=np.array(json.load(open(OUT/'hand_reach_collision_solution.json'))['qpos'])
x0=np.r_[q0[:3],2*np.arctan2(q0[5],q0[3]),q0[7:]]
lower=np.r_[[-.48,-.08,.27,.0],ik.limits[:,0]+.003];upper=np.r_[[-.05,.15,.65,1.4],ik.limits[:,1]-.003]
sol=least_squares(fun,x0,bounds=(lower,upper),max_nfev=1000,ftol=1e-8,xtol=1e-8,gtol=1e-8)
fun(sol.x)
r={'cost':sol.cost,'root':sol.x[:4].tolist(),'qpos':ik.data.qpos.tolist(),'targets':targets,'errors':{n:float(np.linalg.norm(ik.data.site_xpos[ik.sites[n]]-t)) for n,t in targets.items()},'min_collision_clearance_m':min(mujoco.mj_geomDistance(m,d,a,b,.08,line) for a,b in pairs),'left_palm_rpy':Rotation.from_matrix(ik.data.site_xmat[ik.sites['left_palm']].reshape(3,3)).as_euler('xyz').tolist()}
print(json.dumps(r,indent=2));(OUT/'hand_reach_collision_solution.json').write_text(json.dumps(r,indent=2))
