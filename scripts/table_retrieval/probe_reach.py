import sys,json
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from demo import IK,OUT,quat_euler
ik=IK();m=np.load(OUT/'motion.npz');q0=m['qpos'][-1].copy()
# Release support only after both feet are down; reach with the formerly supporting hand.
targets={'left_sole':[-.07,.12,0],'right_sole':[.12,-.12,0],'left_palm':[.135,.12,.110],'right_palm':[.05,-.28,.42]}
def fun(x):
 ik.data.qpos[:3]=x[:3];ik.data.qpos[3:7]=quat_euler([0,x[3],0]);ik.data.qpos[7:]=x[4:];mujoco.mj_forward(ik.model,ik.data)
 e=[]
 for name,target in targets.items():
  sid=ik.sites[name];e.extend((ik.data.site_xpos[sid]-target)*20)
  if 'sole' in name:e.extend(Rotation.from_matrix(ik.data.site_xmat[sid].reshape(3,3)).as_rotvec()*2)
 e.extend((x[:4]-[-.12,0,.43,.65])*.1)
 e.extend(x[16:19]*.15)
 e.extend((x[4:]-ik.nominal)*.003)
 return np.array(e)
x0=np.r_[[-.12,0,.43,.65],q0[7:]]
lower=np.r_[[-.35,-.10,.25,-.1],ik.limits[:,0]+.003];upper=np.r_[[.25,.10,.75,1.2],ik.limits[:,1]-.003]
sol=least_squares(fun,x0,bounds=(lower,upper),max_nfev=500,ftol=1e-9,xtol=1e-9,gtol=1e-9)
fun(sol.x)
r={'cost':sol.cost,'root':sol.x[:4].tolist(),'qpos':ik.data.qpos.tolist(),'errors':{n:float(np.linalg.norm(ik.data.site_xpos[ik.sites[n]]-t)) for n,t in targets.items()},'actual_sites':{n:ik.data.site_xpos[ik.sites[n]].tolist() for n in targets}}
print(json.dumps(r,indent=2));(OUT/'hand_reach_solution.json').write_text(json.dumps(r,indent=2))
