"""Compare baked USD meshes/transforms with MuJoCo saved states."""
import json
import numpy as np,mujoco
from pxr import Usd,UsdGeom,UsdUtils
from demo import OUT,set_robot

def validate(out=OUT):
 stage=Usd.Stage.Open(str(out/'table_retrieval.usdc'));motion=dict(np.load(out/'motion.npz'));sim=dict(np.load(out/'simulation.npz'))
 model=mujoco.MjModel.from_xml_path(str(out/'scene.xml'));data=mujoco.MjData(model)
 errors=[];checks=0
 for frame in [0,390,690,len(motion['time'])-1]:
  set_robot(model,data,motion,float(motion['time'][frame]));data.qpos[:]=sim['object_qpos'][frame];mujoco.mj_forward(model,data)
  for gid in range(model.ngeom):
   if model.geom_type[gid]!=mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[gid]!=1:continue
   name=model.geom(gid).name;body=model.body(model.geom_bodyid[gid]).name
   path='/World/G1/'+body+'/'+name+'/Shape';prim=stage.GetPrimAtPath(path)
   if not prim.IsValid():raise ValueError('Missing USD mesh: '+path)
   mesh=UsdGeom.Mesh(prim);points=np.array(mesh.GetPointsAttr().Get(),dtype=float);mat=np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(frame))
   usd_points=points@mat[:3,:3]+mat[3,:3]
   mid=model.geom_dataid[gid];start=model.mesh_vertadr[mid];count=model.mesh_vertnum[mid]
   mj_points=model.mesh_vert[start:start+count]@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
   error=max(abs(usd_points.min(axis=0)-mj_points.min(axis=0)).max(),abs(usd_points.max(axis=0)-mj_points.max(axis=0)).max())
   errors.append(float(error));checks+=1
 layers,assets,unresolved=UsdUtils.ComputeAllDependencies(str(out/'table_retrieval.usdc'))
 result={'mesh_world_bounds_checks':checks,'maximum_mesh_world_bounds_error_m':max(errors),'external_asset_dependencies':[str(a) for a in assets],'unresolved_dependencies':[str(a) for a in unresolved],'layer_count':len(layers),'usd_replay_matches_mujoco_meshes':bool(max(errors)<1e-5 and not unresolved),'isaac_renderer_executed':False}
 (out/'usd_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
 assert result['usd_replay_matches_mujoco_meshes'],result
if __name__=='__main__':validate()
