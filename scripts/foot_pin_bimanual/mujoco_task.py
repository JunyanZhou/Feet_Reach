"""Free-base G1 and a foot-braced, jointed sliding lid in MuJoCo.

The controller is the existing balance student plus Cartesian arm IK.  The
lid and base remain fully dynamic; only the *reference points* are scripted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.mujoco_whole_body_student import Simulation, DEFAULT_BUNDLE, DEFAULT_MODEL


def build_model(with_brace: bool, params: dict) -> Path:
    tree = ET.parse(DEFAULT_MODEL)
    world = tree.find("worldbody")
    for side in ("left", "right"):
        wrist = tree.find(f".//body[@name='{side}_wrist_yaw_link']")
        ET.SubElement(wrist, "geom", name=f"{side}_palm_contact", type="sphere",
                      pos="0.116 0 0", size="0.031", mass="0.04",
                      rgba="0.92 0.75 0.48 1", friction="1.5 0.05 0.005")
    # The brace contacts the left shoe, while the cabinet alone contacts the
    # floor.  This keeps the brace from becoming an accidental ground anchor.
    left_foot = tree.find(".//body[@name='left_ankle_roll_link']")
    for geom in left_foot.findall("geom"):
        if geom.get("contype") != "0":
            geom.set("contype", "3")
            geom.set("conaffinity", "3")
    base = ET.SubElement(world, "body", name="pin_base", pos="0.55 0 0.35")
    ET.SubElement(base, "freejoint", name="base_free")
    base_mass = params.get("base_mass", 6.0)
    inertia_scale = base_mass / 6.
    ET.SubElement(base, "inertial", pos="0 0 0", mass=str(base_mass),
                  diaginertia=f"{.14*inertia_scale} {.11*inertia_scale} {.10*inertia_scale}")
    ET.SubElement(base, "geom", name="base_cabinet", type="box",
                  pos="0 0 -0.055", size="0.11 0.17 0.295", mass="0",
                  priority=str(params.get("base_priority", 0)),
                  rgba="0.14 0.27 0.31 1",
                  friction=f"{params.get('base_friction', 1.5)} 0.02 0.002")
    # A separate, low front lip gives the left toe something to press against.
    # It is the sole geometry removed in the no-brace ablation.
    if with_brace and params.get("support_mode") == "brake":
        pedal = ET.SubElement(base, "body", name="foot_brake", pos="-0.46 0.10 -0.326")
        ET.SubElement(pedal, "joint", name="foot_brake_slide", type="slide",
                      axis="0 0 1", range="-0.035 0", stiffness="450", damping="12")
        ET.SubElement(pedal, "inertial", pos="0 0 0", mass="0.5",
                      diaginertia="0.004 0.004 0.004")
        ET.SubElement(pedal, "geom", name="brake_pedal", type="box",
                      pos="0 0 0", size="0.20 0.17 0.009", mass="0",
                      contype="2", conaffinity="3", priority="3",
                      rgba="0.90 0.63 0.15 1", friction="1.6 0.03 0.003")
        ET.SubElement(base, "geom", name="brake_beam", type="box",
                      pos="-0.23 0.19 -0.31", size="0.24 0.018 0.014",
                      mass="0", contype="0", conaffinity="0",
                      rgba="0.90 0.63 0.15 1")
    elif with_brace and params.get("support_mode") == "plate":
        ET.SubElement(base, "geom", name="support_foot_plate", type="box",
                      pos="-0.46 0.10 -0.347", size="0.20 0.18 0.005",
                      mass="0", contype="2", conaffinity="2",
                      rgba="0.90 0.63 0.15 1", friction="1.5 0.03 0.003")
        ET.SubElement(base, "geom", name="plate_beam", type="box",
                      pos="-0.23 0.19 -0.32", size="0.24 0.018 0.014",
                      mass="0", contype="0", conaffinity="0",
                      rgba="0.90 0.63 0.15 1")
    elif with_brace:
        brace_contact = {}
        if params.get('brace_solref') is not None:
            brace_contact['solref'] = f"{params['brace_solref']} 1"
        if params.get('brace_solimp') is not None:
            brace_contact['solimp'] = params['brace_solimp']
        ET.SubElement(base, "geom", name="foot_brace", type="box",
                      pos=f"{params.get('toe_wall_x', -.370)} 0.10 -0.305",
                      size=f"{params.get('toe_wall_width', .040)} {params.get('toe_wall_y_size', .090)} 0.045",
                      mass="0", contype="2", conaffinity="2",
                      rgba="0.9 0.63 0.15 1", friction="1.5 0.03 0.003",
                      **brace_contact)
        ET.SubElement(base, "geom", name="heel_cuff", type="box",
                      pos=f"{params.get('heel_wall_x', -.590)} 0.10 -0.305",
                      size=f"{params.get('heel_wall_width', .020)} {params.get('heel_wall_y_size', .090)} 0.045",
                      mass="0", contype="2", conaffinity="2",
                      rgba="0.9 0.63 0.15 1", friction="1.5 0.03 0.003",
                      **brace_contact)
        ET.SubElement(base, "geom", name="brace_beam", type="box",
                      pos="-0.31 0.19 -0.275", size="0.29 0.012 0.018",
                      mass="0", contype="0", conaffinity="0",
                      rgba="0.9 0.63 0.15 1", friction="1.5 0.03 0.003")
    pull_drawer = params.get("task_mode") == "pull_drawer"
    lid = ET.SubElement(base, "body", name="sliding_lid",
                        pos=f"{params.get('lid_x', -.19)} 0 0.343")
    ET.SubElement(lid, "joint", name="lid_slide", type="slide",
                  axis="-1 0 0" if pull_drawer else "1 0 0",
                  range="0 0.19", damping="1.0",
                  frictionloss=str(params.get("lid_friction", 8.0)),
                  stiffness=str(params.get("lid_stiffness", 0.0)), springref="0")
    ET.SubElement(lid, "inertial", pos="0 0 0", mass="0.35",
                  diaginertia="0.002 0.002 0.002")
    ET.SubElement(lid, "geom", name="lid_panel", type="box", pos="0 0 0",
                  size="0.018 0.16 0.045", mass="0", rgba="0.94 0.32 0.08 1",
                  friction="1.2 0.03 0.003")
    if pull_drawer:
        ET.SubElement(lid, "geom", name="drawer_floor", type="box",
                      pos="0.10 0 -0.064", size="0.125 0.14 0.012",
                      mass="0", contype="0", conaffinity="0",
                      rgba="0.94 0.40 0.10 1")
        for side, y in (("left", .13), ("right", -.13)):
            ET.SubElement(lid, "geom", name=f"drawer_{side}_rail", type="box",
                          pos=f"0.10 {y} -0.034", size="0.125 0.01 0.04",
                          mass="0", contype="0", conaffinity="0",
                          rgba="0.94 0.40 0.10 1")
    # Two pads on the front face let each palm contribute to opening the lid.
    for side, y, x in (("left", 0.095, -0.06), ("right", -0.095, -0.025)):
        if pull_drawer:
            x = -.155
        x = params.get(f"{side}_pad_x", x)
        depth = params.get(f"{side}_pad_depth", .012 if pull_drawer else .012)
        width = params.get(f"{side}_pad_width", .040 if pull_drawer else .048)
        height = params.get(f"{side}_pad_height", .045 if pull_drawer else .039)
        ET.SubElement(lid, "geom", name=f"{side}_push_pad", type="box",
                      pos=f"{x} {y} 0", size=f"{depth} {width} {height}", mass="0",
                      rgba="1 0.53 0.14 1", friction="1.6 0.03 0.003")
    mode = params.get("support_mode", "stirrup")
    # Each process loads its XML immediately after generating it. Keeping a
    # process-local filename prevents concurrent searches from overwriting a
    # different mass/friction/handle layout between write and model load.
    output = DEFAULT_MODEL.parent / f"g1_foot_pin_{mode}_pid{os.getpid()}_{'brace' if with_brace else 'no_brace'}.xml"
    tree.write(output)
    return output


def smooth(t: float) -> float:
    u = np.clip(t, 0., 1.)
    return float(u * u * (3. - 2. * u))


class TaskSimulation(Simulation):
    def __init__(self, with_brace: bool, params: dict, seed: int | None = None,
                 randomize: bool = False):
        self.params = params
        self.with_brace = with_brace
        super().__init__(DEFAULT_BUNDLE, build_model(with_brace, params))
        self.residual_action = np.zeros(8)
        replay_path = params.get("target_replay_file")
        if replay_path:
            replay_path = Path(replay_path)
            if not replay_path.is_absolute():
                replay_path = ROOT / replay_path
        self.target_replay = np.load(replay_path) if replay_path else None
        names = self.meta['joint_names']
        self.arm_idx = {side: np.asarray([i for i, name in enumerate(names)
                         if name.startswith(side + '_') and
                         any(part in name for part in ('shoulder', 'elbow', 'wrist'))], dtype=int)
                        for side in ('left', 'right')}
        self.leg_idx = {side: np.asarray([i for i, name in enumerate(names)
                         if name.startswith(side + '_') and
                         any(part in name for part in ('hip', 'knee', 'ankle'))], dtype=int)
                        for side in ('left', 'right')}
        assert all(len(self.arm_idx[s]) == 7 and len(self.leg_idx[s]) == 6
                   for s in ('left', 'right'))
        self.lid_id = self.model.joint("lid_slide").id
        self.lid_qadr = self.model.jnt_qposadr[self.lid_id]
        self.base_id = self.model.body("pin_base").id
        self.lid_body_id = self.model.body("sliding_lid").id
        self.hand_ids = {side: self.model.body(f"{side}_wrist_yaw_link").id
                         for side in ("left", "right")}
        self.hand_geoms = {self.model.geom(f"{side}_palm_contact").id: side
                           for side in ("left", "right")}
        self.pad_geoms = {self.model.geom(f"{side}_push_pad").id: side
                          for side in ("left", "right")}
        self.pad_ids = {side: self.model.geom(f"{side}_push_pad").id
                        for side in ("left", "right")}
        self.brace_geom = (self.model.geom("brake_pedal").id if params.get("support_mode") == "brake"
                           else self.model.geom("support_foot_plate").id if params.get("support_mode") == "plate"
                           else self.model.geom("foot_brace").id) if with_brace else -1
        self.heel_geom = (self.model.geom("heel_cuff").id if with_brace and
                          params.get("support_mode", "stirrup") not in ("plate", "brake") else -1)
        foot_body = self.model.body("left_ankle_roll_link").id
        toe_geoms = [g for g in range(self.model.ngeom)
                     if self.model.geom_bodyid[g] == foot_body and
                     self.model.geom(g).type == mujoco.mjtGeom.mjGEOM_SPHERE]
        self.toe_geom = toe_geoms[2]
        self.foot_body = foot_body
        right_foot_body = self.model.body("right_ankle_roll_link").id
        right_toes = [g for g in range(self.model.ngeom)
                      if self.model.geom_bodyid[g] == right_foot_body and
                      self.model.geom(g).type == mujoco.mjtGeom.mjGEOM_SPHERE]
        self.right_toe_geom = right_toes[2]
        self.right_foot_body = right_foot_body
        if params.get("root_height_offset", 0.):
            self.data.qpos[2] += params["root_height_offset"]
            mujoco.mj_forward(self.model, self.data)
        if randomize:
            rng = np.random.default_rng(seed)
            root_noise = params.get("root_noise_m", .01)
            joint_noise = params.get("joint_noise_rad", .01)
            base_noise = params.get("base_noise_m", .015)
            self.data.qpos[:2] += rng.uniform(-root_noise, root_noise, 2)
            noise = rng.normal(0., joint_noise, len(self.qadr))
            joint_limits = self.model.jnt_range[self.ids]
            self.data.qpos[self.qadr] = np.clip(self.data.qpos[self.qadr] + noise,
                                                joint_limits[:, 0] + .01,
                                                joint_limits[:, 1] - .01)
            base_adr = self.model.jnt_qposadr[self.model.joint("base_free").id]
            self.data.qpos[base_adr:base_adr + 2] += rng.uniform(-base_noise, base_noise, 2)
            mujoco.mj_forward(self.model, self.data)
        self.base_initial = self.data.xpos[self.base_id].copy()
        self.fixed_foot_goal = self.data.geom_xpos[self.toe_geom].copy()
        self.fixed_foot_goal[2] = .006
        self.fixed_right_foot_goal = self.data.geom_xpos[self.right_toe_geom].copy()
        self.fixed_right_foot_goal[2] = .006
        initial_wall = self.data.xpos[self.base_id] + self.data.xmat[self.base_id].reshape(3, 3) @ np.array([
            self.params.get("toe_wall_x", -.370), .10, -.305])
        self.fixed_pin_goal = np.array([
            initial_wall[0] - self.params.get("toe_wall_width", .040) - .004,
            initial_wall[1] + self.params.get("foot_target_y", -.02), .007])
        self.initial_hands = None
        self.arm_start = self.data.qpos[self.qadr].copy()
        self.arm_targets = None

    def hand_positions(self):
        return {side: self.data.xpos[bid] + self.data.xmat[bid].reshape(3, 3) @ [0.116, 0, 0]
                for side, bid in self.hand_ids.items()}

    def _update_policy(self):
        if self.target_replay is not None:
            t = self.data.time
            # The trace stores state after stepping; its target was computed
            # one physics step earlier at the policy tick.
            times = self.target_replay['time'] - self.model.opt.timestep
            targets = self.target_replay['target']
            i = min(max(int(np.searchsorted(times, t)), 1), len(times) - 1)
            weight = np.clip((t - times[i - 1]) / max(times[i] - times[i - 1], 1e-9), 0., 1.)
            self.target = (1 - weight) * targets[i - 1] + weight * targets[i]
            root_gain = self.params.get('replay_root_gain', 0.)
            if root_gain:
                reference_root = ((1 - weight) * self.target_replay['root_pos'][i - 1] +
                                  weight * self.target_replay['root_pos'][i])
                root_error = np.clip(self.data.qpos[:2] - reference_root[:2],
                                     -self.params.get('replay_root_clip_m', .08),
                                     self.params.get('replay_root_clip_m', .08))
                span = self.leg_idx['right']
                jacp = np.zeros((3, self.model.nv))
                jacr = np.zeros((3, self.model.nv))
                mujoco.mj_jac(self.model, self.data, jacp, jacr,
                               self.data.geom_xpos[self.right_toe_geom],
                               self.right_foot_body)
                J = jacp[:, self.vadr[span]]
                correction = J.T @ np.linalg.solve(J @ J.T + .01 * np.eye(3),
                                                    [*root_error, 0.])
                self.target[span] += root_gain * correction
            if self.params.get('replay_feedback', False):
                if self.initial_hands is None:
                    self.initial_hands = {k: v.copy() for k, v in self.hand_positions().items()}
                self._update_foot_pin(t)
                self._cartesian_arms(t, self.params.get('approach_start', 1.5))
            return
        super()._update_policy()
        t = self.data.time
        approach_start = self.params.get("approach_start", 1.5)
        if t < approach_start:
            self._update_foot_pin(t)
            self._update_stance(t)
            return
        if self.initial_hands is None:
            self.initial_hands = {k: v.copy() for k, v in self.hand_positions().items()}
        if self.params.get("control", "joint") == "joint":
            self._update_joint_arms(t, approach_start)
            self._update_foot_pin(t)
            self._update_stance(t)
            return
        if self.params.get("control") == "hybrid":
            self._update_joint_arms(t, approach_start)
            self._track_lid(t)
            self._update_foot_pin(t)
            self._update_stance(t)
            return
        if self.params.get("control") == "cartesian":
            self._cartesian_arms(t, approach_start)
            self._update_foot_pin(t)
            self._update_stance(t)
            return
        if self.params.get("control") == "drawer_pull":
            self._drawer_pull_arms(t, approach_start)
            self._update_foot_pin(t)
            self._update_stance(t)
            return
        approach = smooth((t - approach_start) / self.params.get("approach_duration", 1.3))
        push = smooth((t - self.params.get("push_start", 3.1)) / self.params.get("push_duration", 1.6))
        for side, span, y in (("left", self.arm_idx['left'], 0.095),
                               ("right", self.arm_idx['right'], -0.095)):
            # Palm sphere should touch the near face of its pad.  References
            # follow the *measured* base position so no object is teleported.
            target = self.data.xpos[self.base_id] + np.array([
                -0.19 - 0.025 - 0.012 - 0.031 - 0.002 + self.params.get("push_depth", 0.18) * push,
                y, 0.343])
            desired = (1 - approach) * self.initial_hands[side] + approach * target
            point = self.hand_positions()[side]
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jacp, jacr, point, self.hand_ids[side])
            columns = self.vadr[span]
            J = jacp[:, columns]
            error = desired - point
            delta = J.T @ np.linalg.solve(J @ J.T + 0.005 * np.eye(3),
                                         self.params.get("ik_gain", 0.38) * error)
            delta = np.clip(delta, -0.065, 0.065)
            q = self.data.qpos[self.qadr[span]]
            limits = self.model.jnt_range[self.ids[span]]
            ik_target = np.clip(q + delta, limits[:, 0] + .01, limits[:, 1] - .01)
            self.target[span] = (1 - approach) * self.target[span] + approach * ik_target
        self._update_foot_pin(t)
        self._update_stance(t)

    def _update_foot_pin(self, t):
        gain = self.params.get("foot_gain", 0.)
        if gain <= 0:
            return
        weight = smooth((t - self.params.get("foot_start", 1.2)) /
                        self.params.get("foot_ramp", .6))
        if weight <= 0:
            return
        point = self.data.geom_xpos[self.toe_geom]
        # Use the same object-relative virtual toe target in *both* ablations;
        # no-brace deletes collision geometry only, never a control command.
        if self.params.get("support_mode") == "plate":
            desired = self.fixed_foot_goal.copy()
        elif self.params.get("foot_goal_mode") == "world":
            desired = self.fixed_pin_goal.copy()
        else:
            base_rot = self.data.xmat[self.base_id].reshape(3, 3)
            wall = self.data.xpos[self.base_id] + base_rot @ np.array([
                self.params.get("toe_wall_x", -.370), .10, -.305])
            desired = np.array([wall[0] - self.params.get("toe_wall_width", .040) - .004,
                                wall[1] + self.params.get("foot_target_y", -.02),
                                max(.007, point[2])])
        desired[:2] += self.residual_action[6:8] * self.params.get('residual_foot_scale_m', .05)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, point, self.foot_body)
        span = self.leg_idx['left']
        J = jacp[:, self.vadr[span]]
        delta = J.T @ np.linalg.solve(J @ J.T + .01 * np.eye(3),
                                     gain * weight * (desired - point))
        self.target[span] += np.clip(delta, -.05, .05)

    def _update_stance(self, t):
        gain = self.params.get("stance_gain", 0.)
        if gain <= 0:
            return
        weight = smooth((t - self.params.get("stance_start", .4)) /
                        self.params.get("stance_ramp", .6))
        if weight <= 0:
            return
        point = self.data.geom_xpos[self.right_toe_geom]
        desired = self.fixed_right_foot_goal.copy()
        root_gain = self.params.get("root_track_gain", 0.)
        if root_gain:
            # A stance-foot joint correction supplies horizontal COM feedback.
            # The toe stays a physical MuJoCo contact, with no applied wrench.
            knots = np.asarray(self.params.get("root_reference", [
                [0., 0., 0.], [.5, .01, .07], [1., -.03, .09],
                [1.5, -.10, -.01], [2., -.09, -.03], [2.5, .01, 0.],
                [3., .15, .02], [3.5, .16, .04], [4., .33, .12]]))
            reference = np.array([np.interp(t, knots[:, 0], knots[:, j])
                                  for j in (1, 2)])
            error = np.clip(self.data.qpos[:2] - reference, -.12, .12)
            desired[:2] += root_gain * error
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, point, self.right_foot_body)
        span = self.leg_idx['right']
        J = jacp[:, self.vadr[span]]
        delta = J.T @ np.linalg.solve(J @ J.T + .01 * np.eye(3),
                                     gain * weight * (desired - point))
        self.target[span] += np.clip(delta, -.05, .05)

    def _solve_arm_pose(self, side: str, target: np.ndarray, q_seed: np.ndarray):
        trial = mujoco.MjData(self.model)
        trial.qpos[:] = self.data.qpos
        span = self.arm_idx[side]
        bid = self.hand_ids[side]
        columns = self.vadr[span]
        limits = self.model.jnt_range[self.ids[span]]
        q = q_seed.copy()
        for _ in range(80):
            trial.qpos[self.qadr[span]] = q
            mujoco.mj_forward(self.model, trial)
            point = trial.xpos[bid] + trial.xmat[bid].reshape(3, 3) @ [0.116, 0, 0]
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, trial, jacp, jacr, point, bid)
            J = jacp[:, columns]
            dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(3), .6 * (target - point))
            q = np.clip(q + np.clip(dq, -.08, .08), limits[:, 0] + .01, limits[:, 1] - .01)
        return q

    def _update_joint_arms(self, t, approach_start):
        if self.arm_targets is None:
            self.arm_targets = {}
            for side, span, y, x in (("left", self.arm_idx['left'], .095, -.295),
                                      ("right", self.arm_idx['right'], -.095, -.26)):
                q_seed = self.data.qpos[self.qadr[span]].copy()
                near = self.data.xpos[self.base_id] + np.array([
                    x + self.params.get(f"{side}_near_adjust",
                                        self.params.get("near_adjust", 0.)), y,
                    self.params.get("target_height", .343)])
                far = near + np.array([self.params.get("push_depth", .15), 0, 0])
                q_near = self._solve_arm_pose(side, near, q_seed)
                q_far = self._solve_arm_pose(side, far, q_near)
                self.arm_targets[side] = (q_near, q_far)
        approach = smooth((t - approach_start) / self.params.get("approach_duration", 1.2))
        push = smooth((t - self.params.get("push_start", 3.0)) / self.params.get("push_duration", 1.3))
        for side, span in (("left", self.arm_idx['left']),
                           ("right", self.arm_idx['right'])):
            q_near, q_far = self.arm_targets[side]
            planned = (1 - push) * q_near + push * q_far
            self.target[span] = (1 - approach) * self.target[span] + approach * planned

    def _track_lid(self, t):
        start = self.params.get("hybrid_start", 2.25)
        if t < start:
            return
        weight = smooth((t - start) / self.params.get("hybrid_ramp", .3))
        for side, span in (("left", self.arm_idx['left']),
                           ("right", self.arm_idx['right'])):
            gid = self.pad_ids[side]
            normal = self.data.geom_xmat[gid].reshape(3, 3)[:, 0]
            pad = self.data.geom_xpos[gid]
            point = self.hand_positions()[side]
            desired = pad - normal * (self.model.geom_size[gid, 0] + .031 -
                                      self.params.get("push_penetration", .02))
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jacp, jacr, point, self.hand_ids[side])
            J = jacp[:, self.vadr[span]]
            delta = J.T @ np.linalg.solve(J @ J.T + .008 * np.eye(3),
                                         self.params.get("hybrid_gain", .45) * (desired - point))
            limits = self.model.jnt_range[self.ids[span]]
            ik_target = np.clip(self.data.qpos[self.qadr[span]] + np.clip(delta, -.08, .08),
                                limits[:, 0] + .01, limits[:, 1] - .01)
            self.target[span] = (1 - weight) * self.target[span] + weight * ik_target

    def _cartesian_arms(self, t, start):
        approach = smooth((t - start) / self.params.get("approach_duration", 1.2))
        push = smooth((t - self.params.get("push_start", 2.4)) /
                      self.params.get("push_duration", 1.2))
        gap = self.params.get("approach_gap", .012)
        penetration = self.params.get("push_penetration", .04)
        for side in ("left", "right"):
            span = self.arm_idx[side]
            gid = self.pad_ids[side]
            normal = self.data.geom_xmat[gid].reshape(3, 3)[:, 0]
            pad = self.data.geom_xpos[gid]
            point = self.hand_positions()[side]
            desired_contact = pad - normal * (self.model.geom_size[gid, 0] + .031 +
                                              gap * (1 - push) - penetration * push)
            desired = (1 - approach) * self.initial_hands[side] + approach * desired_contact
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jacp, jacr, point, self.hand_ids[side])
            J = jacp[:, self.vadr[span]]
            delta = J.T @ np.linalg.solve(J @ J.T + .008 * np.eye(3),
                                         self.params.get("cartesian_gain", .5) * (desired - point))
            limits = self.model.jnt_range[self.ids[span]]
            ik_target = np.clip(self.data.qpos[self.qadr[span]] + np.clip(delta, -.09, .09),
                                limits[:, 0] + .01, limits[:, 1] - .01)
            self.target[span] = (1 - approach) * self.target[span] + approach * ik_target

    def _drawer_pull_arms(self, t, start):
        approach = smooth((t - start) / self.params.get("approach_duration", 1.0))
        grasp = smooth((t - self.params.get("grasp_start", 2.5)) /
                       self.params.get("grasp_duration", .6))
        pull = smooth((t - self.params.get("pull_start", 3.15)) /
                      self.params.get("pull_duration", 1.0))
        for side in ("left", "right"):
            span = self.arm_idx[side]
            gid = self.pad_ids[side]
            pad = self.data.geom_xpos[gid]
            normal = self.data.geom_xmat[gid].reshape(3, 3)[:, 0]
            lateral = self.data.geom_xmat[gid].reshape(3, 3)[:, 1]
            point = self.hand_positions()[side]
            backoff = self.model.geom_size[gid, 0] + .031 + self.params.get("grasp_gap", .008)
            side_offset = self.params.get("side_offset", .14) * (1 if side == "left" else -1)
            side_goal = pad + normal * backoff + lateral * side_offset
            hook_goal = pad + normal * (backoff - self.params.get("pull_depth", .20) * pull)
            desired = (1 - approach) * self.initial_hands[side] + approach * (
                (1 - grasp) * side_goal + grasp * hook_goal)
            residual_span = slice(0, 3) if side == 'left' else slice(3, 6)
            desired += approach * self.residual_action[residual_span] * self.params.get(
                'residual_hand_scale_m', .08)
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jacp, jacr, point, self.hand_ids[side])
            J = jacp[:, self.vadr[span]]
            delta = J.T @ np.linalg.solve(J @ J.T + .008 * np.eye(3),
                                         self.params.get("cartesian_gain", .8) * (desired - point))
            limits = self.model.jnt_range[self.ids[span]]
            ik_target = np.clip(self.data.qpos[self.qadr[span]] + np.clip(delta, -.09, .09),
                                limits[:, 0] + .01, limits[:, 1] - .01)
            self.target[span] = (1 - approach) * self.target[span] + approach * ik_target

    def contacts(self):
        result = {"left_hand": False, "right_hand": False, "foot_brace": False}
        for k in range(self.data.ncon):
            c = self.data.contact[k]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if (a in self.hand_geoms and b in self.pad_geoms and
                        self.hand_geoms[a] == self.pad_geoms[b]):
                    result[f"{self.hand_geoms[a]}_hand"] = True
                if b in (self.brace_geom, self.heel_geom) and self.model.body(self.model.geom_bodyid[a]).name == "left_ankle_roll_link":
                    result["foot_brace"] = True
        return result

    def snapshot(self):
        return dict(time=np.asarray(self.data.time), qpos=self.data.qpos.copy(),
                    qvel=self.data.qvel.copy(), history=self.history.copy(),
                    actions=self.actions.copy(), target=self.target.copy(),
                    forces=self.forces.copy(), substep_q=np.asarray(self.substep_q).copy(),
                    steps=np.asarray(self.steps),
                    initial_hands=np.stack([self.initial_hands[s] for s in ('left', 'right')])
                    if self.initial_hands is not None else np.zeros((2, 3)),
                    fixed_pin_goal=self.fixed_pin_goal.copy(),
                    fixed_foot_goal=self.fixed_foot_goal.copy(),
                    fixed_right_foot_goal=self.fixed_right_foot_goal.copy())

    def restore(self, saved, seed: int | None = None, randomize: bool = False):
        self.data.qpos[:] = saved['qpos']
        self.data.qvel[:] = saved['qvel']
        self.data.time = float(saved['time'])
        self.history[:] = saved['history']
        self.actions[:] = saved['actions']
        self.target[:] = saved['target']
        self.forces[:] = saved['forces']
        self.substep_q = [row.copy() for row in saved['substep_q']]
        self.steps = int(saved['steps'])
        stored_hands = saved['initial_hands'].copy()
        self.initial_hands = (None if not np.any(stored_hands) else
                              dict(zip(('left', 'right'), stored_hands)))
        self.fixed_pin_goal = saved['fixed_pin_goal'].copy()
        self.fixed_foot_goal = saved['fixed_foot_goal'].copy()
        self.fixed_right_foot_goal = saved['fixed_right_foot_goal'].copy()
        if randomize:
            rng = np.random.default_rng(seed)
            self.data.qpos[:2] += rng.uniform(-self.params.get('root_noise_m', .01),
                                               self.params.get('root_noise_m', .01), 2)
            joint_sigma = self.params.get('joint_noise_rad', .01)
            limits = self.model.jnt_range[self.ids]
            self.data.qpos[self.qadr] = np.clip(
                self.data.qpos[self.qadr] + rng.normal(0, joint_sigma, len(self.qadr)),
                limits[:, 0] + .01, limits[:, 1] - .01)
            base_adr = self.model.jnt_qposadr[self.model.joint('base_free').id]
            radius = self.params.get('base_noise_m', .015)
            self.data.qpos[base_adr:base_adr+2] += rng.uniform(-radius, radius, 2)
        mujoco.mj_forward(self.model, self.data)
        self.base_initial = self.data.xpos[self.base_id].copy()
        if self.params.get('reset_world_foot_goal_at_snapshot', False):
            self.fixed_pin_goal = self.data.geom_xpos[self.toe_geom].copy()
            self.fixed_pin_goal[2] = .007
        self.first_fall = None


def run(with_brace: bool, params: dict, duration: float, output: Path | None = None,
        record: bool = False, seed: int | None = None, randomize: bool = False,
        start_state: Path | None = None):
    sim = TaskSimulation(with_brace, params, seed=seed,
                         randomize=randomize and start_state is None)
    if start_state is not None:
        sim.restore(np.load(start_state), seed=seed, randomize=randomize)
    steps = round((duration - sim.data.time) / sim.model.opt.timestep)
    step_skip = round(.02 / sim.model.opt.timestep)
    rows = []
    contact_counts = {k: 0 for k in ("left_hand", "right_hand", "foot_brace", "both_hands", "three_way")}
    for i in range(steps):
        sim.step(np.zeros_like(sim.forces))
        c = sim.contacts()
        c["both_hands"] = c["left_hand"] and c["right_hand"]
        c["three_way"] = c["both_hands"] and c["foot_brace"]
        for k in contact_counts:
            contact_counts[k] += int(c[k])
        if i % step_skip == 0:
            rows.append((sim.data.time, sim.data.qpos[:3].copy(),
                         sim.data.qpos[3:7].copy(), sim.data.qpos[sim.qadr].copy(),
                         sim.data.xpos[sim.base_id].copy(),
                         float(sim.data.qpos[sim.lid_qadr]),
                         sim.hand_positions()["left"].copy(),
                         sim.hand_positions()["right"].copy(),
                         c.copy(), sim.data.qpos.copy(), sim.target.copy()))
        if sim.first_fall is not None and params.get('stop_on_fall', True):
            break
    dt = sim.model.opt.timestep
    base_drift = np.linalg.norm(sim.data.xpos[sim.base_id] - sim.base_initial)
    base_max_drift = max(np.linalg.norm(r[4] - sim.base_initial) for r in rows)
    valid_lid = [r[5] for r in rows if np.linalg.norm(r[4] - sim.base_initial) < .05 and r[1][2] > .55]
    lid_steps = np.diff([r[5] for r in rows])
    active_three_way = [i for i, r in enumerate(rows[1:], start=1)
                        if r[8]["three_way"] and lid_steps[i - 1] > .0002]
    three_way_motion = sum(max(0, lid_steps[i - 1]) for i in active_three_way)
    metrics = dict(with_brace=with_brace, seed=seed, randomize=randomize,
                   duration_s=float(sim.data.time),
                   lid_open_m=float(sim.data.qpos[sim.lid_qadr]),
                   maximum_lid_open_m=float(max(r[5] for r in rows)),
                   max_stable_lid_open_m=float(max(valid_lid) if valid_lid else 0),
                   three_way_motion_m=float(three_way_motion),
                   three_way_active_contact_s=float(len(active_three_way) * .02),
                   base_drift_m=float(base_drift), root_min_height_m=float(min(r[1][2] for r in rows)),
                   base_max_drift_m=float(base_max_drift),
                   contact_s={k: round(v * dt, 4) for k, v in contact_counts.items()},
                   fell=sim.first_fall is not None,
                   first_fall_s=sim.first_fall)
    open_threshold = params.get("open_threshold_m", .15)
    metrics["open_threshold_m"] = open_threshold
    mechanical = (metrics["lid_open_m"] >= open_threshold and
                  metrics["base_max_drift_m"] < .05 and
                  metrics["root_min_height_m"] > .55)
    metrics["success"] = bool(mechanical and
                              metrics["contact_s"]["three_way"] >= .15 and
                              metrics["three_way_active_contact_s"] >= .10 and
                              metrics["three_way_motion_m"] >= .004)
    metrics["strict_success"] = bool(metrics["lid_open_m"] >= .15 and
                                     metrics["base_max_drift_m"] < .05 and
                                     metrics["root_min_height_m"] > .55 and
                                     metrics["contact_s"]["three_way"] >= .35 and
                                     metrics["three_way_active_contact_s"] >= .10 and
                                     metrics["three_way_motion_m"] >= .005)
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        if record:
            np.savez_compressed(output / "rollout.npz",
                time=np.asarray([r[0] for r in rows]), root_pos=np.asarray([r[1] for r in rows]),
                root_quat=np.asarray([r[2] for r in rows]), joint_pos=np.asarray([r[3] for r in rows]),
                base_pos=np.asarray([r[4] for r in rows]), lid_open=np.asarray([r[5] for r in rows]),
                left_hand_pos=np.asarray([r[6] for r in rows]),
                right_hand_pos=np.asarray([r[7] for r in rows]),
                contact=np.asarray([list(r[8].values()) for r in rows], dtype=np.int8),
                qpos=np.asarray([r[9] for r in rows]),
                target=np.asarray([r[10] for r in rows]))
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--without-brace", action="store_true")
    p.add_argument("--duration", type=float, default=6.5)
    p.add_argument("--output", type=Path)
    p.add_argument("--record", action="store_true")
    p.add_argument("--params", type=str, default="{}")
    p.add_argument("--params-file", type=Path)
    p.add_argument("--seed", type=int)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--start-state", type=Path)
    args = p.parse_args()
    params = json.loads(args.params)
    if args.params_file is not None:
        saved = json.loads(args.params_file.read_text())
        params = saved.get("params", saved)
    print(json.dumps(run(not args.without_brace, params, args.duration,
                         args.output, args.record, args.seed, args.randomize,
                         args.start_state), indent=2))


if __name__ == "__main__":
    main()
