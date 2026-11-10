"""Foot reaching with explicit root, hand, and both-sole position targets.

This is the wall-supported first stage of the unified controller. Targets are
world-fixed internally and expressed in the live root frame for the student.
It deliberately keeps the existing motion reference for joint supervision;
nonzero target jitter must be trained and evaluated before claiming independent
root/hand/stance-foot control.
"""

import torch

from active_adaptation.envs.mdp import observation, reward
from active_adaptation.envs.mdp.commands.foot_reach import WallFootReachCommand
from active_adaptation.utils.math import quat_apply, quat_apply_inverse
from active_adaptation.utils.symmetry import SymmetryTransform


class UnifiedSupportReachCommand(WallFootReachCommand):
    _GOAL_NAMES = ("root", "left_hand", "right_foot")

    def __init__(self, env, *args, goal_jitter_m=None, goal_offset_range_m=None,
                 goal_jitter_one_goal=True, aux_support_force=None,
                 contact_compliance=None, support_release_cycle=None,
                 landing_cycle=None, anchor_init_prob=None, **kwargs):
        super().__init__(env, *args, **kwargs)
        if anchor_init_prob is not None:
            self.zero_init_prob = float(anchor_init_prob)
            if not 0 <= self.zero_init_prob <= 1:
                raise ValueError("anchor_init_prob must be within [0,1]")
        self.goal_jitter_one_goal = bool(goal_jitter_one_goal)
        jitter = goal_jitter_m or {}
        self.goal_jitter_m = {}
        self.goal_offset_ranges_w = {}
        for name in self._GOAL_NAMES:
            width = torch.as_tensor(jitter.get(name, (0., 0., 0.)),
                                    device=self.device, dtype=torch.float32)
            if width.shape != (3,) or not torch.isfinite(width).all() or (width < 0).any():
                raise ValueError(f"goal_jitter_m.{name} must be three finite nonnegative widths")
            self.goal_jitter_m[name] = width
            if goal_offset_range_m and name in goal_offset_range_m:
                bounds = torch.as_tensor(goal_offset_range_m[name],
                                         device=self.device, dtype=torch.float32)
                if (bounds.shape != (2, 3) or not torch.isfinite(bounds).all()
                        or (bounds[0] > bounds[1]).any()):
                    raise ValueError(f"goal_offset_range_m.{name} must be finite [min_xyz,max_xyz]")
                self.goal_offset_ranges_w[name] = bounds
        self.goal_offsets_w = {
            name: torch.zeros(self.num_envs, 3, device=self.device)
            for name in self._GOAL_NAMES
        }
        self.goal_overrides_w = {
            name: torch.zeros(self.num_envs, 3, device=self.device)
            for name in self._GOAL_NAMES
        }
        self.goal_override_valid = {
            name: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for name in self._GOAL_NAMES
        }
        self.left_foot_goal_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.left_foot_override_valid = torch.zeros(self.num_envs, dtype=torch.bool,
                                                    device=self.device)
        landing_cfg = dict(landing_cycle or {})
        self.landing_cycle_enabled = bool(landing_cfg.pop("enabled", False))
        self.landing_settle_steps = int(landing_cfg.pop("settle_steps", 50))
        self.landing_reach_steps = int(landing_cfg.pop("reach_steps", 100))
        self.landing_transfer_steps = int(landing_cfg.pop("transfer_steps", 50))
        self.landing_root_shift_y_m = float(landing_cfg.pop("root_shift_y_m", 0.))
        self.landing_target_xyz = torch.as_tensor(
            landing_cfg.pop("target_xyz", (0., .06, 0.)),
            dtype=torch.float32, device=self.device)
        self.landing_load_target_N = float(landing_cfg.pop("load_target_N", 80.))
        self.landing_peak_soft_N = float(landing_cfg.pop("peak_soft_N", 300.))
        if (landing_cfg or self.landing_settle_steps < 0 or self.landing_reach_steps < 1
                or self.landing_transfer_steps < 1
                or not -.1 <= self.landing_root_shift_y_m <= .1
                or self.landing_target_xyz.shape != (3,)
                or not torch.isfinite(self.landing_target_xyz).all()
                or self.landing_load_target_N <= 0 or self.landing_peak_soft_N <= 0):
            raise ValueError(f"Invalid landing_cycle settings: {landing_cfg}")
        self.landing_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.landing_start_goal_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.landing_root_start_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.landing_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.landing_foot_peak_N = torch.zeros(self.num_envs, device=self.device)
        cycle_cfg = dict(support_release_cycle or {})
        self.release_cycle_enabled = bool(cycle_cfg.pop("enabled", False))
        self.release_cycle_lengths = tuple(int(cycle_cfg.pop(name, default)) for name, default in (
            ("ramp_out_steps", 100), ("hold_steps", 50),
            ("ramp_in_steps", 100), ("rest_steps", 0)))
        if cycle_cfg or any(value < 0 for value in self.release_cycle_lengths) or (
                min(self.release_cycle_lengths[:3]) == 0 and self.release_cycle_enabled):
            raise ValueError("Invalid support_release_cycle phase lengths or extra fields")
        self.release_cycle_step = torch.zeros(self.num_envs, dtype=torch.long,
                                              device=self.device)
        self.release_goal_amplitude_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.hand_contact_sensor = env.scene["ee_contact_forces"]
        self.foot_contact_sensor = env.scene["contact_forces"]
        self.hand_sensor_id = self._one_body(self.hand_contact_sensor, "left_hand_mimic")
        self.left_foot_sensor_id = self._one_body(self.foot_contact_sensor, "left_ankle_roll_link")
        self.right_foot_sensor_id = self._one_body(self.foot_contact_sensor, "right_ankle_roll_link")
        from active_adaptation.utils.foot_force import FootForceRamp
        force_cfg = dict(aux_support_force or {})
        self.aux_force_enabled = bool(force_cfg.pop("enabled", False))
        self.aux_force_train_only = bool(force_cfg.pop("train_only", True))
        body_names = force_cfg.pop("body_names", ("left_hand_mimic", "torso_link", "pelvis"))
        if not body_names or len(set(body_names)) != len(body_names):
            raise ValueError("aux_support_force.body_names must be unique and nonempty")
        self.aux_body_ids = torch.as_tensor(
            [self.asset.body_names.index(name) for name in body_names],
            dtype=torch.long, device=self.device)
        self.aux_point_radius_m = float(force_cfg.pop("application_radius_m", .02))
        if not 0. <= self.aux_point_radius_m <= .1:
            raise ValueError("aux_support_force.application_radius_m must be in [0, 0.1]")
        self.aux_force_ramp = FootForceRamp(self.num_envs, self.device, **force_cfg)
        self.aux_body_id = torch.full((self.num_envs,), int(self.aux_body_ids[0]),
                                      dtype=torch.long, device=self.device)
        self.aux_point_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._rows = torch.arange(self.num_envs, device=self.device)
        contact_cfg = dict(contact_compliance or {})
        self.contact_compliance_enabled = bool(contact_cfg.pop("enabled", False))
        if self.contact_compliance_enabled:
            from active_adaptation.utils.gentle_contact import ContactEvents
            self.wall_contact_sensor = env.scene["ee_wall_contact_forces"]
            self.contact_force_cap_N = float(contact_cfg.pop("force_cap_N", 80.))
            self.contact_peak_soft_N = float(contact_cfg.pop("peak_soft_N", 100.))
            self.contact_peak_scale_N = float(contact_cfg.pop("peak_scale_N", 100.))
            if (self.contact_force_cap_N <= 0 or self.contact_peak_soft_N < 0
                    or self.contact_peak_scale_N <= 0 or contact_cfg):
                raise ValueError(f"Invalid contact_compliance settings: {contact_cfg}")
            self.contact_events = ContactEvents(self.num_envs, self.device,
                                                env.physics_dt, window=.1, release=.1)
            self.contact_event_step = torch.zeros(self.num_envs, device=self.device)
            self.contact_sustain_step = torch.zeros_like(self.contact_event_step)
            self.contact_peak_N = torch.zeros_like(self.contact_event_step)

    @staticmethod
    def _one_body(sensor, name):
        ids, found = sensor.find_bodies(name)
        if len(ids) != 1:
            raise ValueError(f"Expected one contact sensor body {name}, got {found}")
        return ids[0]

    def reset(self, env_ids):
        super().reset(env_ids)
        if hasattr(self, "aux_force_ramp"):
            self.aux_force_ramp.reset(env_ids)
            self.aux_point_b[env_ids] = 0.
        if hasattr(self, "left_foot_override_valid"):
            self.left_foot_override_valid[env_ids] = False
        if hasattr(self, "landing_step"):
            self.landing_step[env_ids] = 0
            self.landing_active[env_ids] = False
            self.landing_foot_peak_N[env_ids] = 0.
            if self.landing_cycle_enabled and self.env.training:
                self.landing_start_goal_w[env_ids] = self._reference_world("left_foot")[env_ids]
                self.landing_root_start_w[env_ids] = self._reference_world("root")[env_ids]
                self.left_foot_goal_w[env_ids] = self.landing_start_goal_w[env_ids]
                self.left_foot_override_valid[env_ids] = True
        if hasattr(self, "contact_events"):
            self.contact_events.reset(env_ids)
            # The reference starts with the hand supported; only a fresh
            # contact after sustained separation counts as a recontact event.
            self.contact_events.armed[env_ids] = False
            self.contact_events.age[env_ids] = self.contact_events.window_steps
            self.contact_event_step[env_ids] = 0.
            self.contact_sustain_step[env_ids] = 0.
            self.contact_peak_N[env_ids] = 0.
        # Motion references remain the zero-offset, physically checked anchor.
        # Jitter is opt-in and is sampled once per episode, never every step.
        if hasattr(self, "goal_offsets_w"):
            enabled_goals = [i for i, name in enumerate(self._GOAL_NAMES)
                             if (bool(self.goal_jitter_m[name].any())
                                 or (name in self.goal_offset_ranges_w
                                     and bool((self.goal_offset_ranges_w[name][0]
                                               != self.goal_offset_ranges_w[name][1]).any())))]
            active_goal = None
            if self.env.training and self.goal_jitter_one_goal and enabled_goals:
                choices = torch.as_tensor(enabled_goals, device=self.device)
                active_goal = choices[torch.randint(len(choices), (len(env_ids),),
                                                    device=self.device)]
            for index, name in enumerate(self._GOAL_NAMES):
                self.goal_override_valid[name][env_ids] = False
                width = self.goal_jitter_m[name]
                if self.env.training and name in self.goal_offset_ranges_w:
                    bounds = self.goal_offset_ranges_w[name]
                    self.goal_offsets_w[name][env_ids] = (
                        bounds[0] + torch.rand(len(env_ids), 3, device=self.device)
                        * (bounds[1]-bounds[0]))
                else:
                    self.goal_offsets_w[name][env_ids] = (
                        (2. * torch.rand(len(env_ids), 3, device=self.device)-1.)*width
                        if self.env.training else 0.)
                if active_goal is not None:
                    self.goal_offsets_w[name][env_ids] *= (active_goal == index)[:, None]
            if self.release_cycle_enabled:
                if "left_hand" not in self.goal_offset_ranges_w:
                    raise ValueError("Support release cycle requires a left-hand offset range")
                self.release_goal_amplitude_w[env_ids] = self.goal_offsets_w["left_hand"][env_ids]
                self.goal_offsets_w["left_hand"][env_ids] = 0.
                self.release_cycle_step[env_ids] = 0

    def step(self, substep):
        if substep == 0 and self.landing_cycle_enabled and self.env.training:
            u = ((self.landing_step-self.landing_settle_steps).float()
                 /self.landing_reach_steps).clamp(0., 1.)
            fraction = u*u*u*(10.-15.*u+6.*u*u)
            landing = self.env.scene.env_origins+self.landing_target_xyz
            self.left_foot_goal_w.copy_(
                (1.-fraction[:, None])*self.landing_start_goal_w
                +fraction[:, None]*landing)
            self.left_foot_override_valid.fill_(True)
            if self.landing_root_shift_y_m:
                transfer_u = ((self.landing_step-self.landing_settle_steps
                               -self.landing_reach_steps).float()
                              /self.landing_transfer_steps).clamp(0., 1.)
                transfer_fraction = (transfer_u**3
                                     *(10.-15.*transfer_u+6.*transfer_u.square()))
                root_goal = self.landing_root_start_w.clone()
                root_goal[:, 1] += self.landing_root_shift_y_m*transfer_fraction
                self.goal_overrides_w["root"].copy_(root_goal)
                self.goal_override_valid["root"].fill_(True)
            self.landing_active.copy_(self.landing_step >=
                                      self.landing_settle_steps+self.landing_reach_steps)
            self.landing_step += 1
            self.landing_foot_peak_N.zero_()
        if substep == 0 and self.release_cycle_enabled and self.env.training:
            out, hold, back, rest = self.release_cycle_lengths
            elapsed = self.release_cycle_step % (out+hold+back+rest)
            outward = (elapsed.float()/out).clamp(0., 1.)
            inward = ((elapsed.float()-out-hold)/back).clamp(0., 1.)
            outward = outward*outward*(3.-2.*outward)
            inward = inward*inward*(3.-2.*inward)
            fraction = torch.where(elapsed < out, outward,
                                   torch.where(elapsed < out+hold, torch.ones_like(inward),
                                               torch.where(elapsed < out+hold+back,
                                                           1.-inward, 0.)))
            self.goal_offsets_w["left_hand"].copy_(
                self.release_goal_amplitude_w*fraction[:, None])
            self.release_cycle_step += 1
        super().step(substep)
        if substep == 0 and self.contact_compliance_enabled:
            self.contact_event_step.zero_()
            self.contact_sustain_step.zero_()
            self.contact_peak_N.zero_()
        enabled = self.aux_force_enabled and (self.env.training or not self.aux_force_train_only)
        if not enabled:
            self.aux_force_ramp.force.zero_()
            return
        self.force_apply_world = True
        if substep == 0:
            previous_phase = self.aux_force_ramp.phase.clone()
            self.aux_force_ramp.advance()
            entering = ((previous_phase != 1) & (self.aux_force_ramp.phase == 1)).nonzero().flatten()
            if len(entering):
                choice = torch.randint(len(self.aux_body_ids), (len(entering),), device=self.device)
                self.aux_body_id[entering] = self.aux_body_ids[choice]
                self.aux_point_b[entering] = (
                    (2. * torch.rand(len(entering), 3, device=self.device) - 1.)
                    * self.aux_point_radius_m)
            magnitude = self.aux_force_ramp.force.norm(dim=-1)
            self.env.extra["force/aux_mean_N"] = magnitude.mean().item()
            self.env.extra["force/aux_max_N"] = magnitude.max().item()
        ids = self.aux_body_id
        self.force_apply_buffer[self._rows, ids] += self.aux_force_ramp.force
        data = self.asset.data
        self.position_apply_buffer[self._rows, ids] = (
            data.body_pos_w[self._rows, ids]
            + quat_apply(data.body_quat_w[self._rows, ids], self.aux_point_b))
        self.asset.has_external_wrench = False

    def post_step(self, substep):
        if self.landing_cycle_enabled:
            forces = self.foot_contact_sensor.data.net_forces_w
            if forces is not None:
                normal = forces[:, self.left_foot_sensor_id, 2].clamp_min(0.)
                self.landing_foot_peak_N.copy_(torch.maximum(self.landing_foot_peak_N,
                                                              normal))
        if not self.contact_compliance_enabled:
            return
        matrix = self.wall_contact_sensor.data.force_matrix_w
        if matrix is None:
            return
        force = matrix.reshape(self.num_envs, -1, 3).sum(dim=1).norm(dim=-1)
        present = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        impulse, peak, rise = self.contact_events.advance(
            force, present, impulse_free=2., impulse_ref=3.1,
            impulse_scale=.5, impulse_cap=1.,
            peak_soft=80., peak_scale=60., peak_weight=.5, peak_cap=1.,
            rate_soft=500., rate_scale=5000., rate_cap=.1)
        self.contact_event_step += impulse + peak + rise
        self.contact_sustain_step += (
            ((force-self.contact_force_cap_N).clamp_min(0.)
             /self.contact_force_cap_N).square().clamp(max=1.)
            /self.env.decimation)
        self.contact_peak_N.copy_(torch.maximum(self.contact_peak_N, force))
        if substep == self.env.decimation-1:
            self.env.extra["contact/wall_peak_N"] = self.contact_peak_N.max().item()

    @reward
    def wall_recontact_cost(self):
        if not self.contact_compliance_enabled:
            return torch.zeros(self.num_envs, 1, device=self.device)
        return -self.contact_event_step[:, None]

    @reward
    def landing_support_reward(self):
        """Reward sustained two-foot load after the commanded foot has landed."""
        if not self.landing_cycle_enabled:
            return torch.zeros(self.num_envs, 1, device=self.device)
        forces = self.foot_contact_sensor.data.net_forces_w
        if forces is None:
            return torch.zeros(self.num_envs, 1, device=self.device)
        left = (forces[:, self.left_foot_sensor_id, 2].clamp_min(0.)
                /self.landing_load_target_N).clamp(max=1.)
        right = (forces[:, self.right_foot_sensor_id, 2].clamp_min(0.)
                 /250.).clamp(max=1.)
        return (self.landing_active.float()*left*right)[:, None]

    @reward
    def landing_impact_cost(self):
        if not self.landing_cycle_enabled:
            return torch.zeros(self.num_envs, 1, device=self.device)
        excess = ((self.landing_foot_peak_N-self.landing_peak_soft_N).clamp_min(0.)
                  /self.landing_peak_soft_N).clamp(max=2.)
        return -(excess.square())[:, None]

    @reward
    def wall_force_cap_cost(self):
        if not self.contact_compliance_enabled:
            return torch.zeros(self.num_envs, 1, device=self.device)
        return -self.contact_sustain_step[:, None]

    @reward
    def wall_contact_away_cost(self):
        """Penalize wall load only when the commanded hand target is withdrawn."""
        if not self.contact_compliance_enabled:
            return torch.zeros(self.num_envs, 1, device=self.device)
        release = ((self._reference_world("left_hand")[:, 1]
                    - self.goal_world("left_hand")[:, 1]).clamp(0., .06)/.06)
        return -(release*self.contact_peak_N.clamp(max=160.)/80.)[:, None]

    @reward
    def wall_no_contact_cost(self):
        """Penalize wall load in a reference phase explicitly marked hand-off-wall."""
        if not self.contact_compliance_enabled:
            return torch.zeros(self.num_envs, 1, device=self.device)
        return -(self.contact_peak_N.clamp(max=160.)/80.)[:, None]

    @reward
    def wall_peak_excess_cost(self):
        """Physics-rate cost for large contact spikes in any support phase."""
        if not self.contact_compliance_enabled:
            return torch.zeros(self.num_envs, 1, device=self.device)
        excess = ((self.contact_peak_N-self.contact_peak_soft_N).clamp_min(0.)
                  /self.contact_peak_scale_N).clamp(max=3.)
        return -excess.square()[:, None]

    def set_goal_world(self, name, target):
        """Override one world-fixed target for student rollout; None restores reference."""
        if name == "left_foot":
            valid = self.left_foot_override_valid
            buffer = self.left_foot_goal_w
        elif name in self._GOAL_NAMES:
            valid = self.goal_override_valid[name]
            buffer = self.goal_overrides_w[name]
        else:
            raise KeyError(f"Unknown goal {name!r}; expected left_foot or {self._GOAL_NAMES}")
        if target is None:
            valid.zero_()
            return
        value = torch.as_tensor(target, dtype=torch.float32, device=self.device)
        if value.shape == (3,):
            value = value.expand(self.num_envs, -1)
        if value.shape != (self.num_envs, 3) or not torch.isfinite(value).all():
            raise ValueError("World target must be finite xyz or [num_envs, 3]")
        buffer.copy_(value)
        valid.fill_(True)

    def target_sole_world(self):
        reference_or_root_relative = super().target_sole_world()
        return torch.where(self.left_foot_override_valid[:, None],
                           self.left_foot_goal_w, reference_or_root_relative)

    def _reference_world(self, name):
        if name == "left_foot":
            return super().target_sole_world()
        origins = self.env.scene.env_origins
        if name == "root":
            reference = self.reward_root_pos_w
        elif name == "left_hand":
            reference = self._motion.body_pos_w[:, 0, self.hand_motion] + origins
        elif name == "right_foot":
            index = self.right_motion
            reference = (
                self._motion.body_pos_w[:, 0, index]
                + quat_apply(self._motion.body_quat_w[:, 0, index],
                             self.sole_offset.expand(self.num_envs, -1))
                + origins
            )
        else:
            raise KeyError(name)
        return reference

    def goal_world(self, name):
        if name == "left_foot":
            return self.target_sole_world()
        target = self._reference_world(name) + self.goal_offsets_w[name]
        return torch.where(self.goal_override_valid[name][:, None],
                           self.goal_overrides_w[name], target)

    def _goal_b(self, name):
        data = self.asset.data
        return quat_apply_inverse(data.root_quat_w,
                                  self.goal_world(name) - data.root_pos_w)

    def _goal_offset_b(self, name):
        return quat_apply_inverse(self.asset.data.root_quat_w,
                                  self.goal_world(name) - self._reference_world(name))

    @observation
    def target_root_pos_b(self):
        return self._goal_b("root")

    def target_root_pos_b_sym(self):
        return SymmetryTransform(torch.arange(3), [1., 1., 1.])

    @observation
    def target_left_hand_pos_b(self):
        return self._goal_b("left_hand")

    def target_left_hand_pos_b_sym(self):
        return SymmetryTransform(torch.arange(3), [1., 1., 1.])

    @observation
    def target_right_foot_pos_b(self):
        return self._goal_b("right_foot")

    def target_right_foot_pos_b_sym(self):
        return SymmetryTransform(torch.arange(3), [1., 1., 1.])

    @observation
    def goal_offset_root_b(self):
        return self._goal_offset_b("root")

    def goal_offset_root_b_sym(self):
        return SymmetryTransform(torch.arange(3), [1., 1., 1.])

    @observation
    def goal_offset_left_hand_b(self):
        return self._goal_offset_b("left_hand")

    def goal_offset_left_hand_b_sym(self):
        return SymmetryTransform(torch.arange(3), [1., 1., 1.])

    @observation
    def goal_offset_right_foot_b(self):
        return self._goal_offset_b("right_foot")

    def goal_offset_right_foot_b_sym(self):
        return SymmetryTransform(torch.arange(3), [1., 1., 1.])

    @observation
    def support_contact_loads(self):
        """Public measured loads: left hand, right stance foot, left swing foot."""
        hand = self.hand_contact_sensor.data.net_forces_w
        feet = self.foot_contact_sensor.data.net_forces_w
        if hand is None or feet is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        return torch.stack((
            hand[:, self.hand_sensor_id].norm(dim=-1) / 100.,
            feet[:, self.right_foot_sensor_id].norm(dim=-1) / 400.,
            feet[:, self.left_foot_sensor_id].norm(dim=-1) / 400.,
        ), dim=-1)

    def support_contact_loads_sym(self):
        return SymmetryTransform(torch.arange(3), [1., 1., 1.])

    @reward
    def root_pos_tracking(self):
        error = (self.asset.data.root_pos_w - self.goal_world("root")).norm(dim=-1, keepdim=True)
        return torch.exp(-error / .08)

    @reward
    def wall_hand_tracking(self):
        error = (self.asset.data.body_pos_w[:, self.hand_asset]
                 - self.goal_world("left_hand")).norm(dim=-1, keepdim=True)
        return torch.exp(-error / .025)

    @reward
    def support_foot_tracking(self):
        data = self.asset.data
        sole = data.body_pos_w[:, self.right_asset] + quat_apply(
            data.body_quat_w[:, self.right_asset],
            self.sole_offset.expand(self.num_envs, -1))
        error = (sole - self.goal_world("right_foot")).norm(dim=-1, keepdim=True)
        return torch.exp(-error / .025)
