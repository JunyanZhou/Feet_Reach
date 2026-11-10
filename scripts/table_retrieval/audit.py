"""Geometric audit against original robot collision shapes (not a dynamics test)."""
from pathlib import Path
import copy,json,xml.etree.ElementTree as ET
import numpy as np,mujoco
from demo import MODEL,OUT

def audit_model(out=OUT):
 root=ET.parse(MODEL).getroot();root.find('compiler').set('meshdir',str(MODEL.parent/'meshes'))
 for tag in ['actuator','sensor','keyframe']:
  e=root.find(tag)
  if e is not None:root.remove(e)
 for body in root.findall('.//body'):
  for j,g in enumerate(body.findall('geom')):
   g.set('name',body.get('name')+'_g'+str(j))
   if g.get('contype','1')!='0':g.set('contype','1');g.set('conaffinity','7')
 for side in ['left','right']:
  body=root.find(f'.//body[@name="{side}_wrist_yaw_link"]')
  ET.SubElement(body,'geom',name=side+'_hand_collision',type='mesh',mesh=side+'_rubber_hand',pos=('.0415 .003 0' if side=='left' else '.0415 -.003 0'),contype='1',conaffinity='7',density='0')
 scene=ET.parse(out/'scene.xml').getroot();world=root.find('worldbody')
 for wb in root.findall('worldbody'):
  for g in list(wb.findall('geom')):wb.remove(g)
 for g in scene.find('worldbody').findall('geom'):
  if g.get('contype','1')=='0':continue
  g=copy.deepcopy(g);g.attrib.pop('material',None);g.set('contype','2');g.set('conaffinity','5');world.append(g)
 obj=copy.deepcopy(scene.find('.//body[@name="object"]'))
 for g in obj.findall('geom'):
  if g.get('contype')!='0':g.set('contype','4');g.set('conaffinity','3')
 world.append(obj)
 return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))

def audit(out=OUT):
 m=audit_model(out);d=mujoco.MjData(m)
 motion=np.load(out/'motion.npz');sim=np.load(out/'simulation.npz')
 worst={};frames=[]
 for i in range(0,len(motion['time']),3):
  d.qpos[:36]=motion['qpos'][i];d.qpos[36:]=sim['object_qpos'][i];mujoco.mj_forward(m,d)
  for ci in range(d.ncon):
   c=d.contact[ci]
   if c.dist>=-.001:continue
   a,b=m.geom(c.geom1).name,m.geom(c.geom2).name
   pair=' | '.join(sorted([a,b]));entry={'penetration_m':-float(c.dist),'time_s':float(motion['time'][i]),'geoms':[a,b]}
   if pair not in worst or worst[pair]['penetration_m']<entry['penetration_m']:worst[pair]=entry
 results={'note':'Static geometric collision audit using original G1 collision approximations and convex hand mesh; not a stability/force validation. Adjacent-body contacts are filtered by MuJoCo.','contacts_over_1mm':sorted(worst.values(),key=lambda e:-e['penetration_m'])}
 (out/'geometry_audit.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
if __name__=='__main__':audit()
