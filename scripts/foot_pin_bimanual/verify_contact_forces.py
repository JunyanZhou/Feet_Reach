"""Audit partner-filtered MuJoCo contact impulses in a task rollout."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scripts.foot_pin_bimanual.mujoco_task import TaskSimulation


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--params', type=Path, required=True)
    p.add_argument('--without-brace', action='store_true')
    p.add_argument('--duration', type=float, default=4.)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    saved = json.loads(a.params.read_text())
    params = saved.get('params', saved)
    sim = TaskSimulation(not a.without_brace, params)
    names = ('left_hand', 'right_hand', 'foot_brace')
    force_sum = {n: 0. for n in names}
    force_max = {n: 0. for n in names}
    above = {n: 0 for n in names}
    simultaneous_steps = 0
    simultaneous_lid_motion = 0.
    prev_lid = float(sim.data.qpos[sim.lid_qadr])
    for _ in range(round(a.duration / sim.model.opt.timestep)):
        sim.step(np.zeros_like(sim.forces))
        force = {n: 0. for n in names}
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            pair = {contact.geom1, contact.geom2}
            label = None
            for side in ('left', 'right'):
                if {next(g for g, s in sim.hand_geoms.items() if s == side),
                    next(g for g, s in sim.pad_geoms.items() if s == side)} <= pair:
                    label = f'{side}_hand'
            if sim.with_brace and (sim.brace_geom in pair or sim.heel_geom in pair):
                other = contact.geom2 if contact.geom1 in (sim.brace_geom, sim.heel_geom) else contact.geom1
                if sim.model.body(sim.model.geom_bodyid[other]).name == 'left_ankle_roll_link':
                    label = 'foot_brace'
            if label is not None:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(sim.model, sim.data, i, wrench)
                force[label] += max(0., wrench[0])
        for n in names:
            force_sum[n] += force[n]
            force_max[n] = max(force_max[n], force[n])
            above[n] += int(force[n] > (5. if n == 'foot_brace' else 1.))
        if force['left_hand'] > 1. and force['right_hand'] > 1. and force['foot_brace'] > 5.:
            simultaneous_steps += 1
            simultaneous_lid_motion += max(0., float(sim.data.qpos[sim.lid_qadr]) - prev_lid)
        prev_lid = float(sim.data.qpos[sim.lid_qadr])
    dt = sim.model.opt.timestep
    result = dict(with_brace=sim.with_brace,
                  duration_s=a.duration,
                  normal_force_mean_N={k: round(v * dt / a.duration, 3) for k, v in force_sum.items()},
                  normal_force_max_N={k: round(v, 3) for k, v in force_max.items()},
                  loaded_contact_s={k: round(v * dt, 3) for k, v in above.items()},
                  three_way_loaded_s=round(simultaneous_steps * dt, 3),
                  lid_motion_during_three_way_loaded_m=round(simultaneous_lid_motion, 5),
                  lid_final_m=round(float(sim.data.qpos[sim.lid_qadr]), 5))
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
