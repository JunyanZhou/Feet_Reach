"""Free-base G1 pushing a fully dynamic box in MuJoCo.

The balance student supplies the leg reference. A task controller moves both
palms to the box and leans the torso. No body is mocap or welded to the world.
The no-box trial replays the exact joint targets recorded with the box.
"""
from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from scripts.mujoco_whole_body_student import Simulation, DEFAULT_BUNDLE, DEFAULT_MODEL


ROOT = Path(__file__).resolve().parents[2]


def smooth(u):
    u = float(np.clip(u, 0., 1.))
    return u * u * (3. - 2. * u)


def scene(with_box, p):
    tree = ET.parse(DEFAULT_MODEL)
    world = tree.find('worldbody')
    palms_only = p.get('contact_mode', 'palms_only') == 'palms_only'
    # In palms_only mode, box channel 2 touches the floor and added palms.
    # Existing robot geoms stay on channel 1 and cannot push the box.
    if palms_only:
        tree.find(".//geom[@name='floor']").set('conaffinity', '3')
    for side in ('left', 'right'):
        wrist = tree.find(f".//body[@name='{side}_wrist_yaw_link']")
        ET.SubElement(wrist, 'geom', name=f'{side}_push_palm', type='sphere',
                      pos='.116 0 0', size=str(p.get('palm_radius', .038)),
                      mass='.04', rgba='.9 .75 .48 1',
                      friction='1.2 .02 .002', solref='.012 1',
                      **({'contype': '2', 'conaffinity': '2'} if palms_only else {}))
    if with_box:
        x, half_x, half_z = p.get('box_x', .57), p.get('box_half_x', .20), p.get('box_half_z', .38)
        box = ET.SubElement(world, 'body', name='push_box',
                            pos=f'{x} 0 {half_z}')
        ET.SubElement(box, 'freejoint', name='box_free')
        mass = p.get('box_mass', 18.)
        half_y = p.get('box_half_y', .36)
        com_z = p.get('box_com_z_offset', 0.)
        if com_z:
            ET.SubElement(box, 'inertial', pos=f'0 0 {com_z}', mass=str(mass),
                          diaginertia=f'{mass*(half_y**2+half_z**2)/3} '
                                      f'{mass*(half_x**2+half_z**2)/3} '
                                      f'{mass*(half_x**2+half_y**2)/3}')
        ET.SubElement(box, 'geom', name='box_collision', type='box',
                      size=f'{half_x} {half_y} {half_z}',
                      mass='0' if com_z else str(mass),
                      priority=str(p.get('box_contact_priority', 2)),
                      friction=f'{p.get("box_friction", .32)} .01 .001',
                      solref='.008 1', rgba='.63 .36 .16 1',
                      **({'contype': '2', 'conaffinity': '0'} if palms_only else {}))
        for z in (.12, 2 * half_z - .12):
            ET.SubElement(box, 'geom', name=f'band_{z:.2f}', type='box',
                          pos=f'0 0 {z-half_z}',
                          size=f'{half_x+.002} {p.get("box_half_y", .36)+.002} .012',
                          mass='0', contype='0', conaffinity='0',
                          rgba='.30 .22 .15 1')
    output = DEFAULT_MODEL.parent / f'g1_push_box_freebase_pid{os.getpid()}_{int(with_box)}.xml'
    tree.write(output)
    return output


class PushSim(Simulation):
    def __init__(self, p, with_box=True, replay=None, seed=None):
        self.p = p
        self.with_box = with_box
        self.replay = replay
        model_path = scene(with_box, p)
        try:
            super().__init__(DEFAULT_BUNDLE, model_path)
        finally:
            model_path.unlink(missing_ok=True)
        self.arm = {}
        self.leg = {}
        self.hand = {}
        self.ankle = {}
        for side in ('left', 'right'):
            names = self.meta['joint_names']
            self.arm[side] = np.array([i for i, n in enumerate(names)
                if n.startswith(side + '_') and any(s in n for s in ('shoulder', 'elbow', 'wrist'))])
            self.leg[side] = np.array([i for i, n in enumerate(names)
                if n.startswith(side + '_') and any(s in n for s in ('hip', 'knee', 'ankle'))])
            self.hand[side] = self.model.body(f'{side}_wrist_yaw_link').id
            self.ankle[side] = self.model.body(f'{side}_ankle_roll_link').id
        self.waist = self.meta['joint_names'].index('waist_pitch_joint')
        self.left_hip = self.meta['joint_names'].index('left_hip_pitch_joint')
        self.right_hip = self.meta['joint_names'].index('right_hip_pitch_joint')
        self.box_id = self.model.body('push_box').id if with_box else -1
        self.box_geom = self.model.geom('box_collision').id if with_box else -1
        self.palm_geoms = {self.model.geom(f'{side}_push_palm').id: side for side in ('left', 'right')}
        self.start_hands = self.hand_positions()
        self.start_box_x = float(self.data.xpos[self.box_id, 0]) if with_box else p.get('box_x', .57)
        self.data.qpos[2] += p.get('root_height_offset', .03)
        if seed is not None:
            rng = np.random.default_rng(seed)
            self.data.qpos[:2] += rng.uniform(-p.get('root_noise_m', .0), p.get('root_noise_m', .0), 2)
            self.data.qpos[self.qadr] += rng.normal(0., p.get('joint_noise_rad', .0), len(self.qadr))
            if with_box:
                address = self.model.jnt_qposadr[self.model.joint('box_free').id]
                width = p.get('box_noise_m', .0)
                self.data.qpos[address:address+2] += rng.uniform(-width, width, 2)
        mujoco.mj_forward(self.model, self.data)
        self.start_hands = self.hand_positions()
        if with_box:
            self.start_box_x = float(self.data.xpos[self.box_id, 0])
        self.start_ankles = {side: self.data.xpos[bid].copy() for side, bid in self.ankle.items()}
        self.min_root_z = float(self.data.qpos[2])
        self.min_torso_z = float(self.data.xpos[self.model.body('torso_link').id, 2])

    def hand_positions(self):
        return {side: self.data.xpos[bid] + self.data.xmat[bid].reshape(3, 3) @ [0.116, 0, 0]
                for side, bid in self.hand.items()}

    def _update_policy(self):
        if self.replay is not None:
            i = min(self.steps // self.decimation, len(self.replay) - 1)
            self.target = self.replay[i].copy()
            return
        super()._update_policy()
        t = self.data.time
        p = self.p
        approach = smooth((t - p.get('approach_start', .7)) / p.get('approach_duration', 1.5))
        push = smooth((t - p.get('push_start', 2.4)) / p.get('push_duration', 2.3))
        lean = smooth((t - p.get('lean_start', 1.2)) / p.get('lean_duration', 1.7))
        # The online teacher tracks the measured box. Replays use its recorded
        # 29 joint targets and therefore cannot adapt separately in ablation.
        x = (float(self.data.xpos[self.box_id, 0]) if self.with_box else self.start_box_x)
        front = x - p.get('box_half_x', .20)
        for side in ('left', 'right'):
            span = self.arm[side]
            point = self.hand_positions()[side]
            goal = np.array([front - p.get('palm_radius', .038) - p.get('approach_gap', .012)
                             + p.get('push_depth', .18) * push,
                             p.get('hand_y', .21) * (1 if side == 'left' else -1),
                             p.get('hand_z', .65)])
            desired = (1 - approach) * self.start_hands[side] + approach * goal
            jacp = np.zeros((3, self.model.nv)); jacr = np.zeros_like(jacp)
            mujoco.mj_jac(self.model, self.data, jacp, jacr, point, self.hand[side])
            J = jacp[:, self.vadr[span]]
            dq = J.T @ np.linalg.solve(J @ J.T + .008 * np.eye(3),
                                        p.get('ik_gain', .8) * (desired - point))
            limits = self.model.jnt_range[self.ids[span]]
            new = np.clip(self.data.qpos[self.qadr[span]] + np.clip(dq, -.09, .09),
                          limits[:, 0] + .01, limits[:, 1] - .01)
            self.target[span] = (1 - approach) * self.target[span] + approach * new
        self.target[self.waist] += lean * p.get('waist_pitch', .35)
        self.target[self.left_hip] += lean * p.get('hip_pitch', 0.)
        self.target[self.right_hip] += lean * p.get('hip_pitch', 0.)
        for side in ('left', 'right'):
            dx = p.get(f'{side}_step_dx', 0.)
            if dx <= 0:
                continue
            start = p.get(f'{side}_step_start', 1.9 if side == 'left' else 3.1)
            if t < start:
                continue
            duration = p.get(f'{side}_step_duration', 1.0)
            u = float(np.clip((t - start) / duration, 0., 1.))
            desired = self.start_ankles[side] + np.array([
                dx * smooth(u), p.get(f'{side}_step_dy', 0.),
                p.get('step_height', .10) * np.sin(np.pi * u)])
            bid = self.ankle[side]
            point = self.data.xpos[bid]
            jacp = np.zeros((3, self.model.nv)); jacr = np.zeros_like(jacp)
            mujoco.mj_jac(self.model, self.data, jacp, jacr, point, bid)
            span = self.leg[side]
            J = jacp[:, self.vadr[span]]
            dq = J.T @ np.linalg.solve(J @ J.T + .01 * np.eye(3),
                                        p.get('step_gain', .8) * (desired - point))
            self.target[span] += np.clip(dq, -.08, .08)

    def step(self, desired):
        super().step(desired)
        self.min_root_z = min(self.min_root_z, float(self.data.qpos[2]))
        self.min_torso_z = min(self.min_torso_z, float(self.data.xpos[self.model.body('torso_link').id, 2]))

    def contact_sides(self):
        sides = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self.box_geom and c.geom2 in self.palm_geoms:
                sides.add(self.palm_geoms[c.geom2])
            if c.geom2 == self.box_geom and c.geom1 in self.palm_geoms:
                sides.add(self.palm_geoms[c.geom1])
        return sides


def run(p, with_box, duration, replay=None, seed=None):
    s = PushSim(p, with_box, replay, seed)
    n = round(duration / s.model.opt.timestep)
    skip = round(.02 / s.model.opt.timestep)
    frames = []
    targets = []
    contact_time = {'left': 0., 'right': 0., 'both': 0.}
    peak_force = 0.
    palm_impulse = 0.
    non_palm_impulse = 0.
    non_palm_contact_steps = 0
    floor_geom = s.model.geom('floor').id
    for step in range(n):
        if step % s.decimation == 0 and replay is None:
            # target is updated inside step()
            pass
        s.step(np.zeros_like(s.forces))
        if step % s.decimation == 0:
            targets.append(s.target.copy())
        sides = s.contact_sides()
        for side in ('left', 'right'):
            contact_time[side] += s.model.opt.timestep * (side in sides)
        contact_time['both'] += s.model.opt.timestep * (len(sides) == 2)
        for ci in range(s.data.ncon):
            c = s.data.contact[ci]
            if s.box_geom in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == s.box_geom else c.geom1
                if other != floor_geom:
                    f = np.zeros(6); mujoco.mj_contactForce(s.model, s.data, ci, f)
                    impulse = float(abs(f[0])) * s.model.opt.timestep
                    if other in s.palm_geoms:
                        palm_impulse += impulse
                        peak_force = max(peak_force, float(abs(f[0])))
                    else:
                        non_palm_impulse += impulse
                        non_palm_contact_steps += 1
        if step % skip == 0:
            box_qpos = (s.data.qpos[s.model.jnt_qposadr[s.model.joint('box_free').id]:][:7].copy()
                        if with_box else np.full(7, np.nan))
            frames.append((s.data.time, s.data.qpos.copy(),
                           float(s.data.xpos[s.box_id, 0]) if with_box else np.nan,
                           list(sides), box_qpos,
                           s.data.xpos[s.model.body('torso_link').id].copy(),
                           s.data.xpos[s.model.body('left_ankle_roll_link').id].copy(),
                           s.data.xpos[s.model.body('right_ankle_roll_link').id].copy()))
        if s.first_fall is not None and p.get('stop_on_fall', True):
            break
    shift = float(s.data.xpos[s.box_id, 0] - s.start_box_x) if with_box else None
    # Inclination of the box's local up axis, excluding harmless yaw rotation.
    box_tilt = (float(np.max(np.arccos(np.clip(1 - 2 * np.sum(
                    np.square(np.array([f[4][4:6] for f in frames])), axis=1), -1., 1.))))
                if with_box else None)
    box_height_error = (float(np.max(np.abs(np.array([f[4][2] for f in frames]) - p.get('box_half_z', .38))))
                        if with_box else None)
    box_motion_both = (float(sum(max(0., frames[i][2] - frames[i-1][2])
                                 for i in range(1, len(frames)) if len(frames[i][3]) == 2))
                       if with_box else None)
    metrics = dict(with_box=with_box, duration_s=float(s.data.time),
                   box_displacement_x_m=shift, first_fall_s=s.first_fall,
                   final_root_z_m=float(s.data.qpos[2]), min_root_z_m=s.min_root_z,
                   min_torso_z_m=s.min_torso_z,
                   max_box_tilt_rad=box_tilt, max_box_height_error_m=box_height_error,
                   box_motion_during_both_hands_m=box_motion_both,
                   palm_contact_s=contact_time, peak_palm_force_N=peak_force,
                   palm_normal_impulse_Ns=palm_impulse,
                   non_palm_normal_impulse_Ns=non_palm_impulse,
                   non_palm_box_contact_count=non_palm_contact_steps,
                   successful_push=bool(with_box and shift is not None and shift >= p.get('threshold_m', .15)
                                        and s.first_fall is None and contact_time['both'] >= .3
                                        and box_motion_both >= .04
                                        and box_tilt < p.get('max_box_tilt_rad', .25)
                                        and box_height_error < p.get('max_box_height_error_m', .05)))
    return s, metrics, dict(time=np.array([f[0] for f in frames]),
                            qpos=np.array([f[1] for f in frames]),
                            box_x=np.array([f[2] for f in frames]),
                            contact=np.array([','.join(f[3]) for f in frames]),
                            box_qpos=np.array([f[4] for f in frames]),
                            torso_pos=np.array([f[5] for f in frames]),
                            left_ankle_pos=np.array([f[6] for f in frames]),
                            right_ankle_pos=np.array([f[7] for f in frames])), np.asarray(targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--duration', type=float, default=5.5)
    ap.add_argument('--seed', type=int)
    a = ap.parse_args()
    p = json.loads(a.params.read_text()) if a.params else {}
    p = p.get('params', p)
    a.output.mkdir(parents=True, exist_ok=True)
    _, teacher, tr, targets = run(p, True, a.duration, seed=a.seed)
    np.savez_compressed(a.output / 'teacher.npz', **tr, target=targets)
    _, with_box, br, br_targets = run(p, True, a.duration, replay=targets, seed=a.seed)
    np.savez_compressed(a.output / 'with_box_replay.npz', **br, target=br_targets)
    _, without_box, nr, nr_targets = run(p, False, a.duration, replay=targets, seed=a.seed)
    np.savez_compressed(a.output / 'without_box_replay.npz', **nr, target=nr_targets)
    shared = min(len(br_targets), len(nr_targets))
    report = dict(params=p, teacher=teacher, with_box_same_targets=with_box,
                  without_box_same_targets=without_box,
                  shared_target_ticks=shared,
                  maximum_target_difference=float(np.max(np.abs(br_targets[:shared]-nr_targets[:shared]))))
    (a.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
