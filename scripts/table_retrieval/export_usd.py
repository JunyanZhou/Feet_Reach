#!/usr/bin/env python3
"""Export a self-contained animated USD and named-joint portable trajectory.

The USD is visual replay: it deliberately contains no PhysX articulation or
rigid-body APIs. Its object poses were integrated in MuJoCo before baking.
"""
from __future__ import annotations
import argparse,json,re,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh
from pxr import Usd,UsdGeom,UsdShade,UsdLux,Gf,Sdf,Vt
from demo import OUT,MODEL,camera_for,KEYS

def identifier(s):return re.sub('[^a-zA-Z0-9_]','_',s)
def numbers(s,default):return np.array([float(x) for x in s.split()]) if s else np.array(default,dtype=float)
def qgf(q):return Gf.Quatf(float(q[0]),Gf.Vec3f(*map(float,q[1:])))

def export(out):
 motion=dict(np.load(out/'motion.npz'));sim=dict(np.load(out/'simulation.npz'));xml=ET.parse(out/'scene.xml').getroot()
 frames=len(motion['time']);fps=float(motion['fps']);path=out/'table_retrieval.usdc'
 stage=Usd.Stage.CreateNew(str(path));UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.0)
 stage.SetTimeCodesPerSecond(fps);stage.SetFramesPerSecond(fps);stage.SetStartTimeCode(0);stage.SetEndTimeCode(frames-1)
 world=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(world.GetPrim())
 world.GetPrim().SetCustomDataByKey('demo_mode','Scripted robot kinematics, MuJoCo contact-integrated object; visual replay only')
 materials={}
 def material(rgba):
  rgba=tuple(round(float(v),5) for v in rgba)
  if rgba in materials:return materials[rgba]
  name=f'/World/Materials/m{len(materials):03d}';mat=UsdShade.Material.Define(stage,name);shader=UsdShade.Shader.Define(stage,name+'/PreviewSurface')
  shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgba[:3]));shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.48)
  shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(.10 if max(rgba[:3])-min(rgba[:3])<.08 else 0)
  shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(float(rgba[3]))
  mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface');materials[rgba]=mat;return mat
 mesh_defs={m.get('name'):m for m in xml.findall('./asset/mesh')};mesh_cache={}
 def mesh_data(name):
  if name not in mesh_cache:
   entry=mesh_defs[name];mesh=trimesh.load(MODEL.parent/'meshes'/entry.get('file'),process=False)
   mesh_cache[name]=(np.asarray(mesh.vertices,dtype=np.float32)*numbers(entry.get('scale'),[1,1,1]),np.asarray(mesh.faces,dtype=np.int32))
  return mesh_cache[name]
 def geom(parent,element,index):
  rgba=numbers(element.get('rgba'),[.55,.58,.63,1])
  if rgba[3]<.01 or element.get('group')=='3':return
  kind=element.get('type','sphere');name=identifier(element.get('name',f'geometry_{index}'));gp=parent+'/'+name
  transform=UsdGeom.Xform.Define(stage,gp)
  transform.AddTranslateOp().Set(Gf.Vec3d(*numbers(element.get('pos'),[0,0,0])))
  transform.AddOrientOp().Set(qgf(numbers(element.get('quat'),[1,0,0,0])))
  shape=gp+'/Shape';size=numbers(element.get('size'),[.01])
  if kind=='mesh':
   points,faces=mesh_data(element.get('mesh'));g=UsdGeom.Mesh.Define(stage,shape)
   g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
   g.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces),3,dtype=np.int32)))
   g.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.flatten()))
   g.CreateSubdivisionSchemeAttr('none');g.CreateDoubleSidedAttr(True)
  elif kind in ['box','plane']:
   g=UsdGeom.Cube.Define(stage,shape);g.CreateSizeAttr(2)
   if kind=='plane':size=np.array([15,15,.001]);transform.GetOrderedXformOps()[0].Set(Gf.Vec3d(0,0,-.002));rgba=np.array([.21,.26,.31,1])
   UsdGeom.Xformable(g).AddScaleOp().Set(Gf.Vec3f(*map(float,size)))
  elif kind=='sphere':g=UsdGeom.Sphere.Define(stage,shape);g.CreateRadiusAttr(float(size[0]))
  else:raise ValueError(f'Unhandled geometry: {kind}')
  UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(material(rgba))
 worldxml=xml.find('worldbody')
 for i,g in enumerate(worldxml.findall('geom')):geom('/World/Environment',g,i)
 for body in worldxml.findall('body'):
  name=body.get('name');xf=UsdGeom.Xform.Define(stage,'/World/'+('Object' if name=='object' else 'G1/'+name))
  p=xf.AddTranslateOp();q=xf.AddOrientOp()
  if name=='object':positions=sim['object_qpos'][:,:3];quats=sim['object_qpos'][:,3:7]
  else:
   j=list(motion['body_names']).index(name);positions=motion['body_pos'][:,j];quats=motion['body_quat_wxyz'][:,j]
  p.Set(Gf.Vec3d(*map(float,positions[0])));q.Set(qgf(quats[0]))
  for i in range(frames):p.Set(Gf.Vec3d(*map(float,positions[i])),i);q.Set(qgf(quats[i]),i)
  for i,g in enumerate(body.findall('geom')):geom(str(xf.GetPath()),g,i)
 # Standard USD lights are understood by Isaac Sim RTX.
 dome=UsdLux.DomeLight.Define(stage,'/World/Lighting/Dome');dome.CreateIntensityAttr(550);dome.CreateColorAttr(Gf.Vec3f(.80,.88,1.0))
 key=UsdLux.DistantLight.Define(stage,'/World/Lighting/Key');key.CreateIntensityAttr(2100);key.CreateAngleAttr(1.5);key.CreateColorAttr(Gf.Vec3f(1,.93,.83));key.AddRotateXYZOp().Set(Gf.Vec3f(25,-35,-25))
 for view in ['overview','detail','side']:
  c=camera_for(view);az=np.deg2rad(c.azimuth);el=np.deg2rad(c.elevation);target=np.array(c.lookat)
  eye=target+c.distance*np.array([-np.cos(el)*np.cos(az),-np.cos(el)*np.sin(az),-np.sin(el)])
  cam=UsdGeom.Camera.Define(stage,'/World/Cameras/'+view.title())
  cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse())
  cam.CreateHorizontalApertureAttr(36);cam.CreateVerticalApertureAttr(20.25);cam.CreateFocalLengthAttr(24.44);cam.CreateClippingRangeAttr(Gf.Vec2f(.01,100));cam.CreateFocusDistanceAttr(c.distance)
 stage.GetRootLayer().Save()
 # Simulator-neutral joint naming and coordinate conventions are explicit.
 np.savez_compressed(out/'trajectory_portable.npz',time_s=motion['time'],joint_names=motion['joint_names'],joint_position_rad=motion['qpos'][:,7:],joint_velocity_rad_s=np.gradient(motion['qpos'][:,7:],1/fps,axis=0),root_position_world_m=motion['qpos'][:,:3],root_quaternion_wxyz=motion['qpos'][:,3:7],object_position_world_m=sim['object_qpos'][:,:3],object_quaternion_wxyz=sim['object_qpos'][:,3:7],body_names=motion['body_names'],body_position_world_m=motion['body_pos'],body_quaternion_wxyz=motion['body_quat_wxyz'],phase=motion['phase'],fps=np.array(fps))
 timeline=json.loads((out/'timeline.json').read_text()) if (out/'timeline.json').exists() else None
 keytime=lambda old: float(np.interp(old,timeline['old_anchor_s'],timeline['new_anchor_s'])) if timeline else old
 manifest={'units':'metres, radians, seconds','up_axis':'Z','world_axes':{'X':'robot initial forward, toward table','Y':'robot initial left','Z':'up'},'quaternion_order':'wxyz','joint_names':motion['joint_names'].tolist(),'trajectory_hz':fps,'frame_count':frames,'duration_s':float(motion['time'][-1]),'robot_model':'bundled G1 29-DOF rev 1.0','robot_mode':'kinematically scripted','object_mode':'MuJoCo contact-integrated, then baked for replay','usd_mode':'visual-only animated transforms; no PhysX robot/articulation; no hidden force claims','isaac_runtime_validation':'not run on this device','table_top_center_m':[.78,0,.745],'table_top_dimensions_m':[1.0,1.06,.05],'object_dimensions_m':[.09,.09,.08],'object_mass_kg':.22,'table_near_edge_x_m':.28,'target_zone_xy_bounds_m':[[.027,.243],[-.003,.243]],'object_ground_friction_coefficient':.55,'keyframes':[{'time_s':keytime(k[0]),'root_xyz_m':k[1],'root_pitch_rad':k[2],'left_sole_xyz_m':k[3],'left_palm_xyz_m':k[4],'right_palm_xyz_m':k[5],'right_sole_xyz_m':k[7] if len(k)>7 else [.12,-.12,0.],'phase':k[6]} for k in KEYS]}
 (out/'scene_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 check=Usd.Stage.Open(str(path));assert check and check.GetEndTimeCode()==frames-1
 print(f'Exported {path} ({path.stat().st_size/1e6:.1f} MB), {frames} frames, {len(motion["joint_names"])} joints')
 return path
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);args=p.parse_args();export(args.output)
