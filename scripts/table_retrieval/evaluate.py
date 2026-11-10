#!/usr/bin/env python3
"""Task, continuity, geometric, replay and local-variation checks for the demo."""
import argparse,json,itertools,hashlib
from pathlib import Path
import numpy as np,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pxr import Usd,UsdGeom,Gf
from demo import IK,OUT,make_scene,simulate
from audit import audit

def evaluate(out,sweep=True):
 timeline=json.loads((out/'timeline.json').read_text()) if (out/'timeline.json').exists() else {}
 motion=dict(np.load(out/'motion.npz'));ik=IK();model=make_scene(ik,out)
 sim,summary=simulate(model,motion,out)
 t=motion['time'];q=motion['qpos'];vel=np.diff(q[:,7:],axis=0)*float(motion['fps'])
 margin=np.minimum(q[:,7:]-ik.limits[:,0],ik.limits[:,1]-q[:,7:])
 support_start,support_end=timeline.get('support_interval_s',[3,19.5])
 support=(t>=support_start)&(t<=support_end);hand=motion['site_pos'][:,2]
 palm_support_error=np.linalg.norm(hand[support]-[.32,.30,.770],axis=1)
 before=t<timeline.get('before_contact_end_s',8.0);pre_object=np.linalg.norm(sim['object_qpos'][before,:2]-sim['object_qpos'][0,:2],axis=1)
 access_start,access_end=timeline.get('hand_access_interval_s',[23.7,24.7])
 hand_access=(t>=access_start)&(t<=access_end)
 object_top=sim['object_qpos'][:,:3]+[0,0,.04]
 hand_gap=np.linalg.norm(hand[hand_access]-object_top[hand_access],axis=1)
 validation={'task_success':summary['task_success'],'finite_trajectory':bool(np.isfinite(q).all()),'all_joints_within_limits':bool(margin.min()>=0),'minimum_joint_limit_margin_rad':float(margin.min()),'maximum_joint_speed_rad_s':float(abs(vel).max()),'maximum_per_frame_joint_step_rad':float(abs(np.diff(q[:,7:],axis=0)).max()),'support_palm_max_position_error_m':float(palm_support_error.max()),'object_max_horizontal_motion_before_contact_m':float(pre_object.max()),'hand_to_object_top_min_distance_m':float(hand_gap.min()),'hand_to_object_top_max_distance_during_hold_m':float(hand_gap.max()),'mode':'scripted kinematic robot, dynamic object','autonomous_balance_validated':False,'table_support_load_validated':False,'isaac_runtime_tested':False}
 # Causal negative control: identical robot trajectory, active-foot collider removed.
 nofoot_model=make_scene(ik,out);fid=nofoot_model.geom('left_ankle_roll_link_collision').id
 nofoot_model.geom_contype[fid]=0;nofoot_model.geom_conaffinity[fid]=0
 control_dir=out/'controls';control_dir.mkdir(exist_ok=True)
 control,control_metrics=simulate(nofoot_model,motion,control_dir,tag='_no_foot_collision')
 validation['no_foot_collision_object_displacement_m']=control_metrics['object_outward_displacement_m']
 validation['object_motion_requires_foot_collision']=bool(abs(control_metrics['object_outward_displacement_m'])<.001 and summary['object_outward_displacement_m']>.25)
 audit(out)
 geometry=json.loads((out/'geometry_audit.json').read_text())
 unintended=[]
 for entry in geometry['contacts_over_1mm']:
  names=entry['geoms']
  if 'object_collision' in names and any('left_ankle' in n for n in names):continue
  if 'table_top' in names and 'left_hand_collision' in names and entry['time_s']<timeline.get('support_release_end_s',19.6):continue
  if all(n.startswith('left_wrist') for n in names):continue # adjacent wrist collision proxies in the original asset
  unintended.append(entry)
 validation['max_unintended_proxy_penetration_m']=max([e['penetration_m'] for e in unintended],default=0.)
 validation['geometry_audit_sample_hz']=20
 if sweep:
  sweepdir=out/'robustness';sweepdir.mkdir(exist_ok=True);rows=[]
  for i,(dx,dy,mu) in enumerate(itertools.product([-.015,0,.015],[-.015,0,.015],[.35,.55,.8])):
   _,metrics=simulate(make_scene(ik,out),motion,sweepdir,object_offset=(dx,dy),friction=mu,tag=f'_{i:02d}')
   rows.append(metrics)
  sweep_report={'description':'Fixed scripted motion, object x/y offsets and object/ground friction varied; no policy adaptation or generalization claim.','motion_sha256':hashlib.sha256((out/'motion.npz').read_bytes()).hexdigest(),'trials':len(rows),'successful_trials':sum(r['task_success'] for r in rows),'cases':rows}
  (out/'robustness_summary.json').write_text(json.dumps(sweep_report,indent=2))
  validation['local_variation_successes']=sweep_report['successful_trials'];validation['local_variation_trials']=len(rows)
 if not sweep and (out/'robustness_summary.json').exists():
  saved=json.loads((out/'robustness_summary.json').read_text())
  current=hashlib.sha256((out/'motion.npz').read_bytes()).hexdigest()
  validation['robustness_report_is_current']=saved.get('motion_sha256')==current
  if validation['robustness_report_is_current']:
   validation['local_variation_successes']=saved['successful_trials'];validation['local_variation_trials']=saved['trials']
 # A self-contained USD must reproduce saved world body/object transforms exactly.
 usdpath=out/'table_retrieval.usdc'
 if usdpath.exists():
  stage=Usd.Stage.Open(str(usdpath));max_error=0.
  for i in [0,180,600,len(t)-1]:
   for j,name in enumerate(motion['body_names']):
    xf=UsdGeom.Xformable(stage.GetPrimAtPath('/World/G1/'+str(name)));actual=xf.ComputeLocalToWorldTransform(i).ExtractTranslation()
    max_error=max(max_error,float(np.linalg.norm(np.array(actual)-motion['body_pos'][i,j])))
   actual=UsdGeom.Xformable(stage.GetPrimAtPath('/World/Object')).ComputeLocalToWorldTransform(i).ExtractTranslation()
   max_error=max(max_error,float(np.linalg.norm(np.array(actual)-sim['object_qpos'][i,:3])))
  validation['usd_world_translation_max_error_m']=max_error
  validation['usd_has_no_external_layers']=len(stage.GetUsedLayers())<=2
  validation['usd_frame_count']=int(stage.GetEndTimeCode()+1)
 validation['checks']={'task_completed':validation['task_success'],'finite_states':validation['finite_trajectory'],'joint_limits':validation['all_joints_within_limits'],'joint_speed_bounded':validation['maximum_joint_speed_rad_s']<=timeline.get('joint_speed_limit_rad_s',2.40001),'palm_remains_at_support':validation['support_palm_max_position_error_m']<.005,'hand_access_within_6cm':validation['hand_to_object_top_max_distance_during_hold_m']<.06,'no_unintended_proxy_penetration_over_5mm':validation['max_unintended_proxy_penetration_m']<.005,'causal_foot_contact':validation['object_motion_requires_foot_collision']}
 validation['all_nominal_checks_passed']=all(validation['checks'].values())
 (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,layout='constrained')
 axes[0].plot(t,sim['object_qpos'][:,0],color='#e98219',lw=2,label='Object X')
 axes[0].axhline(.28,color='#526b7e',ls='--',label='Table front edge');axes[0].axhspan(.027,.243,color='#45b493',alpha=.12,label='Retrieval zone (center range)');axes[0].set_ylabel('Object X [m]');axes[0].legend(loc='upper right',ncols=3,fontsize=9)
 axes[1].plot(t,sim['foot_object_force'],color='#295c83',lw=1.5);axes[1].set_ylabel('Foot-object normal force [N]')
 axes[2].plot(t,motion['site_pos'][:,0,2],label='Active left sole');axes[2].plot(t,motion['site_pos'][:,1,2],label='Right sole');axes[2].plot(t,hand[:,2],label='Left palm');axes[2].set_ylabel('World height [m]');axes[2].set_xlabel('Time [s]');axes[2].legend(ncols=3,fontsize=9)
 for ax in axes:ax.grid(alpha=.18);ax.set_xlim(t[0],t[-1])
 fig.suptitle('Table-assisted foot retrieval — scripted robot, contact-simulated object',fontsize=14)
 fig.savefig(out/'contact_report.png',dpi=180);plt.close(fig)
 print('VALIDATION\n'+json.dumps(validation,indent=2))
 if not validation['all_nominal_checks_passed']:raise RuntimeError('Nominal acceptance checks failed; see validation.json')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--skip-sweep',action='store_true');a=p.parse_args();evaluate(a.output,not a.skip_sweep)
