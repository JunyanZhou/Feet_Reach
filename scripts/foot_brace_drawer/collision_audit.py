#!/usr/bin/env python3
"""Probe the saved motion with G1 collision proxies and collidable furniture."""
import argparse, json
from pathlib import Path
import numpy as np
import mujoco
from make_demo import OUT, set_robot


def collision_model(out):
    return mujoco.MjModel.from_xml_path(str(out / 'scene.xml'))


def audit(out):
    model = collision_model(out)
    data = mujoco.MjData(model)
    motion = dict(np.load(out / 'motion.npz'))
    sim = dict(np.load(out / 'simulation.npz'))
    by_pair = {}
    for i in range(0, len(motion['time']), 2):
        t = float(motion['time'][i])
        set_robot(model, data, motion, t)
        data.qpos[:7] = sim['base_qpos'][i]
        data.qpos[7] = sim['drawer_slide'][i]
        mujoco.mj_forward(model, data)
        for j in range(data.ncon):
            c = data.contact[j]
            if c.dist >= -.001:
                continue
            names = [model.geom(c.geom1).name, model.geom(c.geom2).name]
            key = ' | '.join(sorted(names))
            entry = {'time_s': t, 'penetration_m': -float(c.dist), 'geoms': names}
            if key not in by_pair or entry['penetration_m'] > by_pair[key]['penetration_m']:
                by_pair[key] = entry
        # The gripped hand meshes are excluded from physical contact response
        # while the weld transmits grasp forces. Audit their actual geometry
        # against every cabinet/drawer collider explicitly.
        line = np.zeros(6)
        for side in ['left', 'right']:
            hand = side + '_wrist_yaw_link_proxy_2'
            a = model.geom(hand).id
            for other in ['base_box', 'frame_far_side', 'frame_top',
                          'frame_back', 'drawer_tray', 'drawer_front',
                          'handle_left', 'handle_right']:
                b = model.geom(other).id
                distance = mujoco.mj_geomDistance(model, data, a, b, .2, line)
                if distance >= -.001:
                    continue
                key = ' | '.join(sorted([hand, other]))
                entry = {'time_s': t, 'penetration_m': -float(distance),
                         'geoms': [hand, other],
                         'contact_response': 'grasp_constraint' if other.startswith('handle') else 'audited_only'}
                if key not in by_pair or entry['penetration_m'] > by_pair[key]['penetration_m']:
                    by_pair[key] = entry
    result = {'sample_hz': 30,
              'note': 'Static collision audit of saved robot and object poses. Intentional grasp and brace contacts are listed too.',
              'contacts_over_1mm': sorted(by_pair.values(), key=lambda e: -e['penetration_m'])}
    (out / 'collision_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    audit(parser.parse_args().output)
