"""Small state-dependent foot residual used with a trained balance policy.

The object pose supplies a high-level contact target. The frozen student still
produces all joint actions; these corrections are learned from Isaac rollouts.
"""

import torch


ACTION_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_ankle_roll_joint",
)


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return x*x*(3.0-2.0*x)


def contact_correction(
    params: torch.Tensor,
    foot_pos_w: torch.Tensor,
    object_pos_w: torch.Tensor,
    initial_object_x_w: torch.Tensor,
    time_s: float,
    version: int = 2,
) -> torch.Tensor:
    """Return [hip pitch, hip roll, hip yaw, ankle roll] action corrections.

    Params per environment: lateral foot offset in metres, then gains for hip
    roll, hip yaw, hip pitch, and ankle roll. Version 2 tracks a foot position
    6 cm behind the moving block so contact persists while it moves sideways.
    """
    if params.ndim == 1:
        params = params.unsqueeze(0).expand(foot_pos_w.shape[0], -1)
    if params.shape != (foot_pos_w.shape[0], 5):
        raise ValueError("contact feedback parameters must have shape [num_envs, 5]")
    lateral_error = ((object_pos_w[:, 1] + params[:, 0] - foot_pos_w[:, 1]) / .10)
    lateral_error = lateral_error.clamp(-1.5, 1.5)
    if version == 1:
        pitch_error = ((initial_object_x_w - object_pos_w[:, 0]) / .10).clamp(0.0, 2.0)
        behind = _smoothstep((foot_pos_w[:, 0] - object_pos_w[:, 0] - .015) / .04)
    elif version in (2, 3):
        target_gap_x = .06 if version == 2 else .045
        pitch_error = ((object_pos_w[:, 0] + target_gap_x - foot_pos_w[:, 0]) / .10)
        pitch_error = pitch_error.clamp(-1.5, 1.5)
        behind = _smoothstep((foot_pos_w[:, 0] - object_pos_w[:, 0] + .03) / .04)
    else:
        raise ValueError("unsupported contact feedback version")
    phase_start, phase_duration = (7.6, .25) if version == 3 else (5.2, .5)
    phase = max(0.0, min(1.0, (time_s - phase_start) / phase_duration))
    phase = phase*phase*(3.0-2.0*phase)
    gate = phase*behind
    return torch.stack((
        (params[:, 3]*pitch_error*gate).clamp(-1.5, 1.5),
        (params[:, 1]*lateral_error*gate).clamp(-1.5, 1.5),
        (params[:, 2]*lateral_error*gate).clamp(-1.5, 1.5),
        (params[:, 4]*lateral_error*gate).clamp(-.45, .45),
    ), dim=-1)
