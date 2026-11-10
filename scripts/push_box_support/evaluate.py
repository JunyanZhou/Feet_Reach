#!/usr/bin/env python3
"""Check box displacement, sustained contact, lean, footsteps, and collision."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from make_demo import OUT, IK
from collision_audit import audit


def evaluate(out=OUT):
    motion=dict(np.load(out/'motion.npz'))
    sim=dict(np.load(out/'simulation.npz'))
    baseline=dict(np.load(out/'simulation_no_hands.npz'))
    t=sim['time'];interval=(t>=1.6)&(t<=8.5)
    force=sim['push_force_N'];count=sim['push_contact_count']
    w=100
    rolling=np.convolve(force,np.ones(w)/w,mode='valid')
    rt=t[w//2:w//2+len(rolling)]
    mid=rolling[(rt>=1.6)&(rt<=8.5)]
    audit_result=audit(out)
    q=motion['qpos'][:,7:]
    ik=IK();limits=ik.limits
    rpy=Rotation.from_quat(motion['qpos'][:,[4,5,6,3]]).as_euler('xyz')
    masses=ik.model.body_mass[1:]
    com_ahead=[]
    for i,ti in enumerate(motion['time']):
        if not 1.6<=ti<=8.5:
            continue
        step=int(round(float(ti)/.002))
        if sim['push_contact_count'][step]==0:
            continue
        ik.data.qpos[:]=motion['qpos'][i]
        mujoco.mj_forward(ik.model,ik.data)
        com=(masses[:,None]*ik.data.xpos[1:]).sum(axis=0)/masses.sum()
        feet=motion['site_pos'][i,:2]
        stance=feet[:,2]<.025
        if stance.any():
            front_toe=float(np.max(feet[stance,0])+.09)
            com_ahead.append((float(com[0]-front_toe),float(ti),
                              float(force[step])))
    max_excess=max(com_ahead,key=lambda x:x[0])
    pelvis_index=list(motion['body_names']).index('pelvis')
    pelvis_ahead=(motion['body_pos'][:,pelvis_index,0]
                  -np.max(motion['site_pos'][:,:2,0],axis=1))
    push_pose=(motion['time']>=2.0)&(motion['time']<=8.0)
    result=dict(duration_s=float(motion['time'][-1]),
        mode='scripted G1 kinematics, free dynamic box, physical palm contacts',
        box_displacement_with_hands_m=float(sim['box_qpos'][-1,0]-sim['box_qpos'][0,0]),
        box_displacement_without_robot_contact_m=float(
            baseline['box_qpos'][-1,0]-baseline['box_qpos'][0,0]),
        box_lateral_drift_m=float(sim['box_qpos'][-1,1]),
        contact_fraction_during_push=float(np.mean(count[interval]>0)),
        min_200ms_average_palm_force_N=float(np.min(mid)),
        peak_palm_contact_force_N=float(np.max(force)),
        max_forward_root_pitch_deg=float(np.degrees(np.max(rpy[:,1]))),
        min_pelvis_ahead_of_both_feet_m=float(np.min(pelvis_ahead[push_pose])),
        mean_pelvis_ahead_of_both_feet_m=float(np.mean(pelvis_ahead[push_pose])),
        max_pelvis_ahead_of_both_feet_m=float(np.max(pelvis_ahead[push_pose])),
        estimated_max_com_ahead_of_stance_toe_m=max_excess[0],
        time_of_max_com_ahead_s=max_excess[1],
        palm_reaction_at_max_com_ahead_N=max_excess[2],
        fraction_of_contact_frames_com_ahead_of_stance_toe=float(
            np.mean([x[0]>0 for x in com_ahead])),
        left_foot_forward_shift_m=float(motion['site_pos'][-1,0,0]-motion['site_pos'][0,0,0]),
        right_foot_forward_shift_m=float(motion['site_pos'][-1,1,0]-motion['site_pos'][0,1,0]),
        max_30hz_robot_box_penetration_m=float(audit_result['robot_box_worst']['penetration_m']),
        max_30hz_robot_floor_penetration_m=float(audit_result['robot_floor_worst']['penetration_m']),
        max_500hz_box_robot_contact_penetration_m=float(sim['max_box_robot_contact_penetration_m']),
        max_hand_mesh_box_clearance_m=float(audit_result['max_hand_mesh_box_clearance_m']),
        max_ik_error_m=float(np.max(motion['ik_error'])),
        max_joint_speed_rad_s=float(np.max(np.abs(np.diff(q,axis=0)))*60),
        min_joint_limit_margin_rad=float(min(np.min(q-limits[:,0]),np.min(limits[:,1]-q))),
        isaac_runtime_tested=False,
        autonomous_balance_validated=False)
    result['checks']={k:bool(v) for k,v in dict(
        box_moves_over_1m=result['box_displacement_with_hands_m']>1,
        no_contact_control_stays_put=abs(result['box_displacement_without_robot_contact_m'])<.01,
        sustained_hand_contact=result['contact_fraction_during_push']>.99,
        sustained_push_force=result['min_200ms_average_palm_force_N']>3,
        forward_lean_over_25deg=result['max_forward_root_pitch_deg']>25,
        pelvis_ahead_of_both_feet_through_push=(
            result['min_pelvis_ahead_of_both_feet_m']>.10),
        com_beyond_stance_toe_with_palm_contact=(
            result['estimated_max_com_ahead_of_stance_toe_m']>.10
            and result['palm_reaction_at_max_com_ahead_N']>10),
        both_feet_reposition=(result['left_foot_forward_shift_m']>.9 and
                              result['right_foot_forward_shift_m']>.75),
        box_stays_straight=abs(result['box_lateral_drift_m'])<.02,
        robot_box_penetration_under_5mm=result['max_30hz_robot_box_penetration_m']<.005,
        robot_floor_penetration_under_5mm=result['max_30hz_robot_floor_penetration_m']<.005,
        physical_box_contact_penetration_under_5mm=result['max_500hz_box_robot_contact_penetration_m']<.005,
        hand_meshes_near_box=result['max_hand_mesh_box_clearance_m']<.015,
        ik_error_under_15mm=result['max_ik_error_m']<.015,
        joint_limits=result['min_joint_limit_margin_rad']>=.0019).items()}
    result['all_checks_passed']=all(result['checks'].values())
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['all_checks_passed']:raise SystemExit('Demo checks failed')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT)
    evaluate(p.parse_args().output)
