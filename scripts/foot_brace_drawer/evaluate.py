#!/usr/bin/env python3
"""Check the visual task and its foot-contact causal control."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from make_demo import OUT, IK, set_robot
from collision_audit import audit


def evaluate(out):
    motion = dict(np.load(out / 'motion.npz'))
    sim = dict(np.load(out / 'simulation.npz'))
    yes = json.loads((out / 'evaluation.json').read_text())
    no = json.loads((out / 'evaluation_no_foot.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(out / 'scene.xml'))
    data = mujoco.MjData(model)
    gaps = []
    line = np.zeros(6)
    for i, t in enumerate(motion['time']):
        if 2.6 <= t <= 4.8:
            set_robot(model, data, motion, float(t))
            data.qpos[:7] = sim['base_qpos'][i]
            data.qpos[7] = sim['drawer_slide'][i]
            mujoco.mj_forward(model, data)
            gaps.extend([
                mujoco.mj_geomDistance(model, data,
                    model.geom('left_wrist_yaw_link_proxy_2').id,
                    model.geom('handle_left').id, .2, line),
                mujoco.mj_geomDistance(model, data,
                    model.geom('right_wrist_yaw_link_proxy_2').id,
                    model.geom('handle_right').id, .2, line),
            ])
    collision = audit(out)
    worst_collision = max((e['penetration_m'] for e in collision['contacts_over_1mm']), default=0)
    colliders = ['base_box', 'frame_far_side', 'frame_top', 'frame_back',
                 'drawer_tray', 'drawer_front', 'handle_left', 'handle_right']
    colliders_present = all(model.geom_contype[model.geom(n).id] != 0 for n in colliders)
    joints = motion['qpos'][:, 7:]
    ik = IK()
    margins = np.minimum(joints - ik.limits[:, 0], ik.limits[:, 1] - joints)
    contact = np.flatnonzero(sim['foot_base_force'] > 1)
    v = np.diff(joints, axis=0) * float(motion['fps'])
    report = {
        'scripted_robot_dynamic_base_and_drawer': True,
        'grip_weld_used_for_hand_forces': True,
        'gripped_hand_meshes_audited_without_contact_response': True,
        'autonomous_balance_validated': False,
        'isaac_runtime_tested': False,
        'duration_s': float(motion['time'][-1]),
        'drawer_extension_with_foot_m': yes['max_drawer_extension_m'],
        'drawer_extension_without_foot_m': no['max_drawer_extension_m'],
        'base_shift_with_foot_m': yes['base_horizontal_shift_m'],
        'base_shift_without_foot_m': no['base_horizontal_shift_m'],
        'foot_base_peak_normal_force_N': yes['peak_foot_base_normal_force_N'],
        'foot_base_contact_start_s': float(sim['time'][contact[0]]) if len(contact) else None,
        'foot_base_contact_end_s': float(sim['time'][contact[-1]]) if len(contact) else None,
        'max_hand_mesh_to_handle_clearance_m': float(max(0, max(gaps))),
        'max_hand_mesh_to_handle_penetration_m': float(max(0, -min(gaps))),
        'max_robot_or_object_proxy_penetration_m': worst_collision,
        'max_unintended_500hz_contact_penetration_m': yes['max_unintended_robot_furniture_penetration_m'],
        'cabinet_and_drawer_colliders_present': colliders_present,
        'max_site_ik_error_m': float(motion['ik_error'].max()),
        'max_joint_speed_rad_s': float(abs(v).max()),
        'min_joint_limit_margin_rad': float(margins.min()),
    }
    report['checks'] = {
        'drawer_extracted_over_20cm': report['drawer_extension_with_foot_m'] > .20,
        'base_shift_under_2cm': report['base_shift_with_foot_m'] < .02,
        'foot_contact_during_pull': report['foot_base_contact_start_s'] < 2.6 and
                                    report['foot_base_contact_end_s'] > 4.8,
        'foot_force_below_150N': report['foot_base_peak_normal_force_N'] < 150,
        'both_hand_meshes_within_15mm_of_handles': report['max_hand_mesh_to_handle_clearance_m'] < .015,
        'max_geometry_penetration_under_5mm': worst_collision < .005,
        'max_unintended_500hz_penetration_under_5mm': yes['max_unintended_robot_furniture_penetration_m'] < .005,
        'cabinet_and_drawer_have_colliders': colliders_present,
        'joint_limits': report['min_joint_limit_margin_rad'] >= 0,
        'no_foot_base_slides_over_10cm': report['base_shift_without_foot_m'] > .10,
        'no_foot_drawer_stays_under_15cm': report['drawer_extension_without_foot_m'] < .15,
    }
    report['all_checks_passed'] = all(report['checks'].values())
    (out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['all_checks_passed']:
        raise RuntimeError('Task validation failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    evaluate(parser.parse_args().output)
