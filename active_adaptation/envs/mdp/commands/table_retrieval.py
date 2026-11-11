"""Task rewards for a physically simulated table-supported foot retrieval.

The actor retains the existing unified public observation/action interface.
Object pose and task phase are used only by the scripted goal sequence and
training reward; they are not sent to the student policy.
"""
import torch
from active_adaptation.envs.mdp import reward
from active_adaptation.utils.math import quat_apply
from .unified_support_reach import UnifiedSupportReachCommand


class TableRetrievalCommand(UnifiedSupportReachCommand):
    def __init__(self, env, *args, hand_support_window=(2.5, 13.5),
                 stance_window=(4.0, 13.5), progress_window=(7.5, 13.5),
                 retrieved_start=10.0, require_supported_progress=False,
                 min_support_force_N=5.0, min_stance_force_N=100.0,
                 min_root_height_m=0.55, hand_target_down_m=0.0, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.target_object = env.scene["target_object"]
        self.previous_object_x = torch.zeros(self.num_envs, device=self.device)
        self.hand_support_window = hand_support_window
        self.stance_window = stance_window
        self.progress_window = progress_window
        self.retrieved_start = retrieved_start
        self.require_supported_progress = require_supported_progress
        self.min_support_force_N = min_support_force_N
        self.min_stance_force_N = min_stance_force_N
        self.min_root_height_m = min_root_height_m
        self.hand_target_down_m = float(hand_target_down_m)
        if not 0 <= self.hand_target_down_m <= 0.08:
            raise ValueError("hand_target_down_m must be within [0, 0.08]")

    def _support_mask(self):
        if not self.require_supported_progress:
            return torch.ones(self.num_envs, device=self.device)
        hand = self.hand_contact_sensor.data.net_forces_w
        feet = self.foot_contact_sensor.data.net_forces_w
        if hand is None or feet is None:
            return torch.zeros(self.num_envs, device=self.device)
        return ((self.asset.data.root_pos_w[:, 2] > self.min_root_height_m)
                & (hand[:, self.hand_sensor_id, 2] > self.min_support_force_N)
                & (feet[:, self.right_foot_sensor_id, 2] > self.min_stance_force_N)).float()

    def reset(self, env_ids):
        super().reset(env_ids)
        if self.hand_target_down_m:
            self.goal_offsets_w["left_hand"][env_ids, 2] -= self.hand_target_down_m
        if hasattr(self, "previous_object_x"):
            self.previous_object_x[env_ids] = self.target_object.data.root_pos_w[env_ids, 0]

    def _phase_s(self):
        return self.t.float() * self.env.step_dt

    @reward
    def swing_foot_distance(self):
        current = self.asset.data.body_pos_w[:, self.left_asset] + quat_apply(
            self.asset.data.body_quat_w[:, self.left_asset],
            self.sole_offset.expand(self.num_envs, -1))
        return -(self.target_sole_world() - current).norm(dim=-1, keepdim=True)

    @reward
    def table_hand_support(self):
        forces = self.hand_contact_sensor.data.net_forces_w
        if forces is None:
            return torch.zeros(self.num_envs, 1, device=self.device)
        magnitude = (forces[:, self.hand_sensor_id, 2].clamp_min(0.)
                     if self.require_supported_progress else
                     forces[:, self.hand_sensor_id].norm(dim=-1))
        phase = self._phase_s()
        active = ((phase >= self.hand_support_window[0]) & (phase <= self.hand_support_window[1])).float()
        support = (magnitude / 40.).clamp(max=1.)
        excess = ((magnitude - 120.).clamp_min(0.) / 80.).clamp(max=2.)
        return (active * (support - excess))[:, None]

    @reward
    def right_stance_load(self):
        forces = self.foot_contact_sensor.data.net_forces_w
        if forces is None:
            return torch.zeros(self.num_envs, 1, device=self.device)
        vertical = forces[:, self.right_foot_sensor_id, 2].clamp_min(0.)
        phase = self._phase_s()
        active = ((phase >= self.stance_window[0]) & (phase <= self.stance_window[1])).float()
        return (active * (vertical / 230.).clamp(max=1.))[:, None]

    @reward
    def object_outward_progress(self):
        x = self.target_object.data.root_pos_w[:, 0]
        delta = (self.previous_object_x - x).clamp(-.02, .02)
        self.previous_object_x.copy_(x)
        phase = self._phase_s()
        active = ((phase >= self.progress_window[0]) & (phase <= self.progress_window[1])).float()
        return (active * self._support_mask() * delta / .02)[:, None]

    @reward
    def object_retrieved(self):
        x = self.target_object.data.root_pos_w[:, 0] - self.env.scene.env_origins[:, 0]
        phase = self._phase_s()
        active = (phase >= self.retrieved_start).float()
        threshold = float(getattr(self.env.cfg, "retrieval_boundary_x", .28)) - float(getattr(self.env.cfg, "object_half_x", .045))
        return (active * self._support_mask() * (x < threshold).float())[:, None]
