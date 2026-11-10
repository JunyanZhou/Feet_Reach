#!/usr/bin/env python3
"""Check door bracing, passage, and collision evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
from make_demo import OUT, IK
from collision_audit import audit


def evaluate(out=OUT):
    motion=dict(np.load(out/'motion.npz'))
    sim=dict(np.load(out/'simulation.npz'))
    baseline=dict(np.load(out/'simulation_no_foot.npz'))
    contact=sim['foot_force']>1.
    idx=np.flatnonzero(contact)
    t=sim['time']
    i4=int(round(4.0/.002))
    angle_with=float(sim['door_angle'][i4])
    angle_without=float(baseline['door_angle'][i4])
    audit_result=audit(out)
    q=motion['qpos'][:,7:]
    limits=IK().limits
    result=dict(
        duration_s=float(motion['time'][-1]),
        mode='scripted G1, kinematic carried cargo, dynamic spring door and contacts',
        door_angle_with_foot_at_4s_deg=float(np.degrees(angle_with)),
        door_angle_without_foot_at_4s_deg=float(np.degrees(angle_without)),
        foot_bracing_effect_deg=float(np.degrees(angle_without-angle_with)),
        foot_contact_start_s=float(t[idx[0]]) if len(idx) else None,
        foot_contact_end_s=float(t[idx[-1]]) if len(idx) else None,
        peak_foot_door_force_N=float(np.max(sim['foot_force'])),
        final_door_angle_deg=float(np.degrees(sim['door_angle'][-1])),
        final_root_x_m=float(motion['qpos'][-1,0]),
        final_cargo_back_x_m=float(motion['cargo_pos'][-1,0]+.04-.115),
        max_ik_error_m=float(np.max(motion['ik_error'])),
        max_joint_speed_rad_s=float(np.max(np.abs(np.diff(q,axis=0)))*60),
        min_joint_limit_margin_rad=float(min(np.min(q-limits[:,0]),
                                            np.min(limits[:,1]-q))),
        max_30hz_geometry_penetration_m=float(audit_result['robot_furniture_worst']['penetration_m']),
        max_cargo_furniture_penetration_m=float(audit_result['cargo_furniture_worst']['penetration_m']),
        max_500hz_contact_penetration_m=float(sim['max_robot_object_penetration_m']),
        isaac_runtime_tested=False)
    result['checks']=dict(
        foot_contacts_door_during_carry=(len(idx)>0 and t[idx[0]]<2 and t[idx[-1]]>4),
        no_late_foot_contact=(len(idx)>0 and t[idx[-1]]<4.3),
        foot_delays_door_rebound=(result['foot_bracing_effect_deg']>10),
        cargo_fully_crosses_door=(result['final_cargo_back_x_m']>1.3),
        robot_root_crosses_door=(result['final_root_x_m']>1.0),
        door_rebounds_after_release=(result['final_door_angle_deg']>-10),
        geometry_penetration_under_5mm=(result['max_30hz_geometry_penetration_m']<.005),
        physics_penetration_under_5mm=(result['max_500hz_contact_penetration_m']<.005),
        cargo_clears_furniture=(result['max_cargo_furniture_penetration_m']<.005),
        ik_error_under_15mm=(result['max_ik_error_m']<.015),
        joint_limits=(result['min_joint_limit_margin_rad']>=.0019))
    result['checks']={k:bool(v) for k,v in result['checks'].items()}
    result['all_checks_passed']=all(result['checks'].values())
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['all_checks_passed']:
        raise SystemExit('Demo checks failed')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=OUT)
    evaluate(p.parse_args().output)
