"""Extend the unified foot/hand controller with an explicit second-hand goal.

This is the interface foundation for foot-pinned, two-hand manipulation. The
right-hand goal is public policy input; joint actions remain policy outputs.
"""
import torch

from active_adaptation.envs.mdp import observation, reward
from active_adaptation.utils.math import quat_apply_inverse
from active_adaptation.utils.symmetry import SymmetryTransform
from .unified_support_reach import UnifiedSupportReachCommand


class BimanualFootPinCommand(UnifiedSupportReachCommand):
    _GOAL_NAMES = ("root", "left_hand", "right_foot", "right_hand")

    def __init__(self, env, *args, base_initial_pos=(.42, 0., .4),
                 base_half_x=.1, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.base_initial_pos = torch.as_tensor(base_initial_pos, device=self.device)
        self.base_half_x = float(base_half_x)
        self.right_hand_motion = self.dataset.body_names.index("right_hand_mimic")
        self.right_hand_asset = self.asset.body_names.index("right_hand_mimic")
        self.right_hand_sensor_id = self._one_body(self.hand_contact_sensor, "right_hand_mimic")
        self.pin_base = env.scene["pin_base"] if "pin_base" in env.scene.keys() else None
        self.extract_part = env.scene["extract_part"] if "extract_part" in env.scene.keys() else None
        self.previous_relative_x = torch.zeros(self.num_envs, device=self.device)
        self.initial_relative_x = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids):
        super().reset(env_ids)
        if (hasattr(self, "previous_relative_x") and self.pin_base is not None
                and self.extract_part is not None):
            self.previous_relative_x[env_ids] = (
                self.extract_part.data.root_pos_w[env_ids, 0]
                - self.pin_base.data.root_pos_w[env_ids, 0])
            self.initial_relative_x[env_ids] = self.previous_relative_x[env_ids]

    def _reference_world(self, name):
        if name == "right_hand":
            return (self._motion.body_pos_w[:, 0, self.right_hand_motion]
                    + self.env.scene.env_origins)
        return super()._reference_world(name)

    @observation
    def target_right_hand_pos_b(self):
        target = self.goal_world("right_hand")
        data = self.asset.data
        return quat_apply_inverse(data.root_quat_w, target - data.root_pos_w)

    def target_right_hand_pos_b_sym(self):
        return SymmetryTransform(torch.arange(3), [1., 1., 1.])

    @reward
    def right_hand_tracking(self):
        error = (self.asset.data.body_pos_w[:, self.right_hand_asset]
                 - self.goal_world("right_hand")).norm(dim=-1, keepdim=True)
        return torch.exp(-error / .15)

    @reward
    def left_hand_distance(self):
        error = (self.asset.data.body_pos_w[:, self.hand_asset]
                 - self.goal_world("left_hand")).norm(dim=-1, keepdim=True)
        return -error.clamp(max=1.)

    @reward
    def right_hand_distance(self):
        error = (self.asset.data.body_pos_w[:, self.right_hand_asset]
                 - self.goal_world("right_hand")).norm(dim=-1, keepdim=True)
        return -error.clamp(max=1.)

    @reward
    def pin_foot_contact(self):
        force = self.foot_contact_sensor.data.net_forces_w
        if force is None or self.pin_base is None:
            return torch.zeros(self.num_envs, 1, device=self.device)
        active = (self.t.float() * self.env.step_dt >= 1.).float()
        magnitude = force[:, self.left_foot_sensor_id].norm(dim=-1)
        foot = self.asset.data.body_pos_w[:, self.left_asset]
        base = self.pin_base.data.root_pos_w
        near_front = ((foot[:, 0] - (base[:, 0] - self.base_half_x)).abs() < .13)
        near_width = ((foot[:, 1] - base[:, 1]).abs() < .18)
        above_ground = (foot[:, 2] > .1).float()
        return (active * near_front.float() * near_width.float() * above_ground
                * (magnitude / 150.).clamp(max=1.))[:, None]

    @reward
    def bilateral_hand_contact(self):
        force = self.hand_contact_sensor.data.net_forces_w
        if force is None or self.extract_part is None:
            return torch.zeros(self.num_envs, 1, device=self.device)
        active = (self.t.float() * self.env.step_dt >= 3.).float()
        left = (force[:, self.hand_sensor_id].norm(dim=-1) / 40.).clamp(max=1.)
        right = (force[:, self.right_hand_sensor_id].norm(dim=-1) / 40.).clamp(max=1.)
        part = self.extract_part.data.root_pos_w
        left_near = ((self.asset.data.body_pos_w[:, self.hand_asset] - part).norm(dim=-1)
                     < .17).float()
        right_near = ((self.asset.data.body_pos_w[:, self.right_hand_asset] - part).norm(dim=-1)
                      < .17).float()
        return (active * left_near * right_near * torch.minimum(left, right))[:, None]

    @reward
    def early_left_support(self):
        force = self.hand_contact_sensor.data.net_forces_w
        if force is None:
            return torch.zeros(self.num_envs, 1, device=self.device)
        active = (self.t.float() * self.env.step_dt < 3.).float()
        hand = self.asset.data.body_pos_w[:, self.hand_asset]
        near_rail = ((hand[:, 1] > .17) & (hand[:, 2] > 1.)).float()
        magnitude = (force[:, self.hand_sensor_id].norm(dim=-1) / 80.).clamp(max=1.)
        return (active * near_rail * magnitude)[:, None]

    @reward
    def right_hand_part_contact(self):
        force = self.hand_contact_sensor.data.net_forces_w
        if force is None or self.extract_part is None:
            return torch.zeros(self.num_envs, 1, device=self.device)
        active = (self.t.float() * self.env.step_dt >= 2.5).float()
        near_part = ((self.asset.data.body_pos_w[:, self.right_hand_asset]
                      - self.extract_part.data.root_pos_w).norm(dim=-1) < .17).float()
        magnitude = (force[:, self.right_hand_sensor_id].norm(dim=-1) / 50.).clamp(max=1.)
        return (active * near_part * magnitude)[:, None]

    @reward
    def base_stillness(self):
        if self.pin_base is None:
            return torch.zeros(self.num_envs, 1, device=self.device)
        expected = self.env.scene.env_origins + self.base_initial_pos
        error = (self.pin_base.data.root_pos_w - expected).norm(dim=-1)
        active = (self.t.float() * self.env.step_dt >= 3.).float()
        return (active * torch.exp(-error / .05))[:, None]

    @reward
    def part_outward_progress(self):
        if self.extract_part is None:
            return torch.zeros(self.num_envs, 1, device=self.device)
        relative_x = self.extract_part.data.root_pos_w[:, 0] - self.pin_base.data.root_pos_w[:, 0]
        delta = (self.previous_relative_x - relative_x).clamp(-.02, .02)
        self.previous_relative_x.copy_(relative_x)
        active = ((self.t.float() * self.env.step_dt >= 3.5)
                  & (self.t.float() * self.env.step_dt <= 7.5)).float()
        return (active * delta / .02)[:, None]

    @reward
    def part_extracted(self):
        if self.extract_part is None:
            return torch.zeros(self.num_envs, 1, device=self.device)
        relative_x = self.extract_part.data.root_pos_w[:, 0] - self.pin_base.data.root_pos_w[:, 0]
        active = (self.t.float() * self.env.step_dt >= 5.).float()
        return (active * ((self.initial_relative_x - relative_x) > .15).float())[:, None]
