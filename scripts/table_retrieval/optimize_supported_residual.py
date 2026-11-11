"""CEM search for a small learned leg residual around a trained balance student.

All candidate returns come from free-base Isaac physics.  The frozen student
chooses every joint action; the searched residual modifies four swing-leg
actions during three goal phases and is saved as policy parameters.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MEMPATH", str(ROOT / "dataset"))
os.environ.setdefault("ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE", "1")

import hydra
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from isaaclab.app import AppLauncher
from omegaconf import OmegaConf
import active_adaptation.learning

LEG_JOINTS = ["left_hip_pitch_joint", "left_hip_roll_joint",
              "left_knee_joint", "left_ankle_pitch_joint"]


def smooth(x):
    x = max(0.0, min(1.0, x))
    return x*x*(3.0-2.0*x)


def phase_weights(t):
    return (smooth(t/.30)*(1-smooth((t-1.5)/.5)),
            smooth((t-1.2)/.5)*(1-smooth((t-4.2)/.5)),
            smooth((t-3.9)/.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--population", type=int, default=64)
    p.add_argument("--replicas", type=int, default=1,
                   help="Evaluate each candidate in several parallel environments")
    p.add_argument("--generations", type=int, default=12)
    p.add_argument("--steps", type=int, default=450)
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--init", type=Path, default=None,
                   help="Start search near a previous best.json")
    p.add_argument("--fixed-base", type=Path, default=None,
                   help="Freeze a learned three-phase reach and optimize two late phases")
    p.add_argument("--fixed-five", type=Path, default=None,
                   help="Freeze five learned phases and optimize a final pull phase")
    p.add_argument("--fixed-five-seven", type=Path, default=None,
                   help="Freeze five phases and optimize both late phases 6 and 7")
    p.add_argument("--fixed-six", type=Path, default=None,
                   help="Freeze six learned phases and optimize a second pull phase")
    p.add_argument("--extra-hip-yaw", action="store_true",
                   help="Search left hip yaw in the final phase; requires --fixed-six")
    p.add_argument("--wide", action="store_true",
                   help="Use wider action range and exploration for late pull search")
    p.add_argument("--grid-pull", action="store_true",
                   help="Sweep late hip pitch and roll; requires 64 environments and --init")
    p.add_argument("--grid-pitch-range", type=float, nargs=2, default=[-.2, 2.5])
    p.add_argument("--grid-roll-range", type=float, nargs=2, default=[-.2, 2.5])
    p.add_argument("--grid-index-a", type=int, default=4)
    p.add_argument("--grid-index-b", type=int, default=5)
    p.add_argument("--object-x", type=float, default=None)
    p.add_argument("--object-height", type=float, default=None)
    p.add_argument("--object-y", type=float, default=None)
    p.add_argument("--object-width-x", type=float, default=None)
    p.add_argument("--object-width-y", type=float, default=None)
    p.add_argument("--object-mass", type=float, default=None)
    p.add_argument("--object-friction", type=float, default=None)
    args = p.parse_args()
    if args.population < 8:
        raise ValueError("population must be at least 8")
    if args.replicas < 1 or args.population % args.replicas or args.population//args.replicas < 8:
        raise ValueError("replicas must divide population with at least eight candidates")
    if args.grid_pull and args.replicas != 1:
        raise ValueError("grid-pull requires replicas=1")
    if sum(x is not None for x in (args.fixed_base, args.fixed_five,
                                   args.fixed_five_seven, args.fixed_six)) > 1:
        raise ValueError("Use only one frozen prefix")
    if args.extra_hip_yaw and args.fixed_six is None and args.fixed_five_seven is None:
        raise ValueError("--extra-hip-yaw requires --fixed-six or --fixed-five-seven")
    if args.grid_pull and (args.population != 64 or args.init is None or args.fixed_base is None):
        raise ValueError("--grid-pull requires --population 64, --init and --fixed-base")
    if args.grid_pull and (not 0 <= args.grid_index_a < 8 or not 0 <= args.grid_index_b < 8
                           or args.grid_index_a == args.grid_index_b):
        raise ValueError("Grid indices must be distinct and in 0..7")
    args.output.mkdir(parents=True, exist_ok=True)
    with hydra.initialize_config_dir(config_dir=str(ROOT / "cfg"), version_base=None):
        cfg = hydra.compose("train", overrides=[
            "task=G1/G1_table_retrieval_supported_rl", "+exp=finetune"])
    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint_path = str(args.checkpoint.resolve())
    cfg.task.num_envs = args.population
    cfg.task.max_episode_length = args.steps + 20
    cfg.task.randomize_episode_length = False
    cfg.task.termination = {}
    cfg.task.command.dataset.mem_paths = ["table_retrieval_supported/train"]
    cfg.task.command.anchor_init_prob = 1.0
    cfg.task.command.student_train = True
    cfg.task.command.foot_force.enabled = False
    for obj in cfg.task.objects:
        if obj.name == "target_object":
            if args.object_x is not None:
                obj.pos[0] = args.object_x
            if args.object_y is not None:
                obj.pos[1] = args.object_y
            if args.object_width_x is not None:
                obj.size[0] = args.object_width_x
            if args.object_width_y is not None:
                obj.size[1] = args.object_width_y
            if args.object_height is not None:
                obj.size[2] = args.object_height
                obj.pos[2] = args.object_height/2 + .001
            if args.object_mass is not None:
                obj.mass = args.object_mass
            if args.object_friction is not None:
                obj.friction = args.object_friction
    cfg.algo.symmetry_augmentation = False
    cfg.vecnorm = "eval"
    cfg.wandb.mode = "disabled"
    cfg.seed = args.seed
    cfg.app = {"headless": True, "enable_cameras": False}
    torch.manual_seed(args.seed)
    AppLauncher(headless=True, device="cuda:0").app
    exit_code = 0
    try:
        from scripts.utils.helpers import make_env_policy
        env, agent, vecnorm, _ = make_env_policy(cfg)
        env.eval(); env.base_env.eval(); agent.eval()
        env.set_seed(args.seed)
        base = env.base_env
        cmd = base.command_manager
        cmd.zero_init_prob = 1.0
        actor = agent.get_rollout_policy("eval")
        normalizer = vecnorm.to_observation_norm()
        normalizer.eval()
        leg_joints = LEG_JOINTS + (["left_hip_yaw_joint"] if args.extra_hip_yaw else [])
        n_joints = len(leg_joints)
        indices = [base.action_manager.joint_names.index(name) for name in leg_joints]
        base_params = None
        if args.fixed_base is not None:
            base_payload = json.loads(args.fixed_base.read_text())
            base_params = torch.tensor(base_payload["parameters"], device=base.device)
            if base_params.shape != (3, 4):
                raise ValueError("fixed-base must contain three phases for four leg joints")
        if args.fixed_five is not None:
            base_payload = json.loads(args.fixed_five.read_text())
            base_params = torch.tensor(base_payload["parameters"], device=base.device)
            if base_params.shape != (5, 4):
                raise ValueError("fixed-five must contain five phases for four leg joints")
        if args.fixed_five_seven is not None:
            base_payload = json.loads(args.fixed_five_seven.read_text())
            base_params = torch.tensor(base_payload["parameters"], device=base.device)
            if base_params.shape not in ((5, 4), (5, 5)):
                raise ValueError("fixed-five-seven must contain five phases for four or five leg joints")
            if args.extra_hip_yaw and base_params.shape == (5, 4):
                base_params = torch.nn.functional.pad(base_params, (0, 1))
            if base_params.shape[1] != n_joints:
                raise ValueError("fixed-five-seven joint count must match optimized joints")
        if args.fixed_six is not None:
            base_payload = json.loads(args.fixed_six.read_text())
            base_params = torch.tensor(base_payload["parameters"], device=base.device)
            if base_params.shape not in ((6, 4), (6, 5)):
                raise ValueError("fixed-six must contain six phases for four or five leg joints")
            if args.extra_hip_yaw and base_params.shape == (6, 4):
                base_params = torch.nn.functional.pad(base_params, (0, 1))
            if base_params.shape[1] != n_joints:
                raise ValueError("fixed-six joint count must match optimized joints")
        param_dim = (2*n_joints if args.fixed_five_seven is not None else
                     n_joints if base_params is not None and base_params.shape[0] in (5, 6) else
                     8 if base_params is not None else 12)
        mean = torch.zeros(param_dim, device=base.device)
        std = torch.full((param_dim,), .8 if args.wide else .45, device=base.device)
        if args.init is not None:
            prior = json.loads(args.init.read_text())
            prior_params = torch.tensor(prior["parameters"], device=base.device)
            if prior_params.shape == (4, 4):
                prior_params = prior_params[:3]
            if base_params is not None and prior_params.shape == (5, 4):
                prior_params = prior_params[3:] if base_params.shape[0] == 3 else prior_params
            if base_params is not None and base_params.shape[0] == 5 and prior_params.shape == (6, 4):
                prior_params = prior_params[5:]
            if base_params is not None and base_params.shape[0] == 6 and prior_params.shape == (7, n_joints):
                prior_params = prior_params[6:]
            if args.fixed_five_seven is not None and prior_params.shape == (7, n_joints):
                prior_params = prior_params[5:]
            if prior_params.numel() != param_dim:
                raise ValueError("--init parameters do not match optimized phases")
            mean = prior_params.flatten()
            std.fill_(.55 if args.wide else .30)
        best_score = -1e9
        best_params = None
        history = []
        candidate_count = args.population//args.replicas
        elite_n = min(candidate_count, max(4, candidate_count//8))
        origin_x = base.scene.env_origins[:, 0]
        object_asset = base.scene["target_object"]
        hand_sensor = base.scene["ee_contact_forces"]
        hand_id = hand_sensor.find_bodies("left_hand_mimic")[0][0]
        stance_sensor = base.scene["contact_forces"]

        for gen in range(args.generations):
            candidate_params = mean + torch.randn(candidate_count, param_dim, device=base.device)*std
            candidate_params = candidate_params.clamp(-2.5 if args.wide else -1.5,
                                                      2.5 if args.wide else 1.5)
            candidate_params[0] = mean
            if best_params is not None:
                candidate_params[1] = best_params
            params = candidate_params.repeat_interleave(args.replicas, dim=0)
            if args.grid_pull:
                params = mean.expand(args.population, -1).clone()
                params[:, args.grid_index_a] = torch.linspace(*args.grid_pitch_range, 8, device=base.device).repeat_interleave(8)
                params[:, args.grid_index_b] = torch.linspace(*args.grid_roll_range, 8, device=base.device).repeat(8)
                candidate_params = params
            env.reset()
            # Custom dynamic objects retain their last simulated pose across
            # env.reset(); restore the configured pose for every CEM generation.
            object_state = object_asset.data.default_root_state.clone()
            object_state[:, :3] += base.scene.env_origins
            object_asset.write_root_state_to_sim(object_state)
            base.episode_length_buf.zero_()
            x0 = object_asset.data.root_pos_w[:, 0].clone()
            root_min = torch.full((args.population,), 10., device=base.device)
            foot_best = torch.full((args.population,), -10., device=base.device)
            safe_foot_best = torch.full((args.population,), -10., device=base.device)
            foot_lift = torch.zeros(args.population, device=base.device)
            lifted_sum = torch.zeros(args.population, device=base.device)
            clearance_sum = torch.zeros(args.population, device=base.device)
            foot_behind_best = torch.zeros(args.population, device=base.device)
            object_best = torch.zeros(args.population, device=base.device)
            object_inward_peak = torch.zeros(args.population, device=base.device)
            foot_track_sum = torch.zeros(args.population, device=base.device)
            support_steps = torch.zeros(args.population, device=base.device)
            supported_push_steps = torch.zeros(args.population, device=base.device)
            previous_object_x = x0.clone()
            foot_x_at_pull_start = torch.zeros(args.population, device=base.device)
            foot_x_at_end = torch.zeros(args.population, device=base.device)
            foot_y_at_end = torch.zeros(args.population, device=base.device)
            late_support_steps = torch.zeros(args.population, device=base.device)
            late_alignment_sum = torch.zeros(args.population, device=base.device)
            with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                for i in range(args.steps):
                    raw = TensorDict({}, batch_size=[args.population], device=base.device)
                    base._compute_observation(raw)
                    carry = normalizer(raw).select("policy")
                    carry["is_init"] = torch.full((args.population, 1), i == 0,
                                                   device=base.device, dtype=torch.bool)
                    actor(carry)
                    w = torch.tensor(phase_weights(i*base.step_dt), device=base.device)
                    if base_params is None:
                        residual = (params.reshape(args.population, 3, 4)*w[None, :, None]).sum(1)
                    elif base_params.shape[0] == 3:
                        late = torch.tensor((smooth((i*base.step_dt-3.5)/.4)
                                             *(1-smooth((i*base.step_dt-5.6)/.4)),
                                             smooth((i*base.step_dt-5.3)/.4)),
                                            device=base.device)
                        residual = (base_params*w[:, None]).sum(0)[None, :]
                        residual = residual + (params.reshape(args.population, 2, 4)
                                               *late[None, :, None]).sum(1)
                    else:
                        late = torch.tensor((smooth((i*base.step_dt-3.5)/.4)
                                             *(1-smooth((i*base.step_dt-5.6)/.4)),
                                             smooth((i*base.step_dt-5.3)/.4)),
                                            device=base.device)
                        frozen = (base_params[:3]*w[:, None]).sum(0)
                        frozen = frozen + (base_params[3:5]*late[:, None]).sum(0)
                        if args.fixed_five_seven is not None:
                            late_two = torch.tensor((smooth((i*base.step_dt-6.1)/.4),
                                                     smooth((i*base.step_dt-7.2)/.5)),
                                                    device=base.device)
                            residual = frozen[None, :] + (
                                params.reshape(args.population, 2, n_joints)
                                * late_two[None, :, None]).sum(1)
                        else:
                            if base_params.shape[0] == 6:
                                frozen = frozen + base_params[5]*smooth((i*base.step_dt-6.1)/.4)
                            final_weight = (smooth((i*base.step_dt-6.1)/.4)
                                            if base_params.shape[0] == 5 else
                                            smooth((i*base.step_dt-7.2)/.5))
                            residual = frozen[None, :] + params.reshape(args.population, n_joints)*final_weight
                    carry["action"][:, indices] += residual
                    base.step(carry)
                    robot = cmd.asset.data
                    root_z = robot.root_pos_w[:, 2]
                    root_min = torch.minimum(root_min, root_z)
                    hand_z = hand_sensor.data.net_forces_w[:, hand_id, 2]
                    stance_z = stance_sensor.data.net_forces_w[:, cmd.right_foot_sensor_id, 2]
                    supported = (root_z > (.50 if base_params is not None else .55)) & (hand_z > 5.) & (stance_z > 100.)
                    support_steps += supported.float()
                    foot = robot.body_pos_w[:, cmd.left_asset]
                    object_pos = object_asset.data.root_pos_w
                    if i >= 310:
                        relative = foot-object_pos
                        lateral_alignment = (1.-relative[:, 1].abs()/.16).clamp(0., 1.)
                        behind_alignment = (1.-(relative[:, 0]-.06).abs()/.16).clamp(0., 1.)
                        late_alignment_sum += lateral_alignment*behind_alignment*supported.float()
                    if 50 <= i < 200:
                        lifted_sum += (((foot[:, 2]-.05)/.10).clamp(0., 1.)
                                       *supported.float())
                        lateral_clearance = ((foot[:, 1]-object_pos[:, 1]-.05)/.10)
                        vertical_clearance = ((foot[:, 2]-object_pos[:, 2]-.07)/.10)
                        clearance_sum += (torch.maximum(lateral_clearance,
                                                        vertical_clearance).clamp(0., 1.)
                                          * (foot[:, 0]-origin_x > .20).float()
                                          * supported.float())
                    if i*base.step_dt < 5.0:
                        target_foot = cmd._reference_world("left_foot")
                        foot_track_sum += (torch.exp(
                            -(foot-target_foot).norm(dim=-1)/.15)*supported.float())
                    foot_x = foot[:, 0] - origin_x
                    if i == min(args.steps - 1, 270):
                        foot_x_at_pull_start.copy_(foot_x)
                    if i == args.steps - 1:
                        foot_x_at_end.copy_(foot_x)
                        foot_y_at_end.copy_(foot[:, 1]-base.scene.env_origins[:, 1])
                    if i >= 270:
                        late_support_steps += supported.float()
                    foot_best = torch.maximum(foot_best,
                                              torch.where(supported, foot_x, -10.))
                    if i < 250:
                        safe_clearance = (((foot[:, 1]-object_pos[:, 1]) > .11)
                                          | ((foot[:, 2]-object_pos[:, 2]) > .11))
                        safe_foot_best = torch.maximum(
                            safe_foot_best,
                            torch.where(supported & safe_clearance, foot_x, -10.))
                    foot_lift = torch.maximum(foot_lift,
                                              torch.where(supported, foot[:, 2], 0.))
                    object_x = object_asset.data.root_pos_w[:, 0]
                    behind = object_pos + torch.tensor([.075, 0., .055], device=base.device)
                    behind_distance = (foot-behind).norm(dim=-1)
                    if 175 <= i < 300:
                        foot_behind_best = torch.maximum(
                            foot_behind_best,
                            torch.where(supported & (object_x < x0+.005),
                                        (1.-behind_distance/.25).clamp(0., 1.), 0.))
                    near_object = (((foot[:, 0]-object_x).abs() < .18)
                                   & ((foot[:, 1]-object_pos[:, 1]).abs() < .15)
                                   & ((foot[:, 2]-object_pos[:, 2]).abs() < .20))
                    object_best = torch.maximum(object_best,
                                                torch.where(supported & near_object,
                                                            x0-object_x, 0.))
                    object_inward_peak = torch.maximum(object_inward_peak,
                                                       (object_x-x0).clamp_min(0.))
                    supported_push_steps += (supported & near_object &
                                             (object_x < previous_object_x-.001)).float()
                    previous_object_x.copy_(object_x)
            quat = object_asset.data.root_quat_w
            qw, qx, qy, qz = (quat[:, j] for j in range(4))
            hx = (args.object_width_x or .09)/2
            hy = (args.object_width_y or .09)/2
            hz = (args.object_height or .08)/2
            back_local_x = (object_asset.data.root_pos_w[:, 0]-origin_x
                            + (1-2*(qy*qy+qz*qz)).abs()*hx
                            + (2*(qx*qy-qz*qw)).abs()*hy
                            + (2*(qx*qz+qy*qw)).abs()*hz)
            score = (200.*object_best.clamp(0., .35)
                     + 5.*(foot_best-.07).clamp(0., .45)
                     + 25.*(safe_foot_best-.07).clamp(0., .45)
                     + 8.*foot_behind_best
                     + 4.*lifted_sum/150.
                     + 8.*clearance_sum/150.
                     + 3.*foot_track_sum/max(1, min(args.steps, 250))
                     + 2.*(foot_lift-.04).clamp(0., .15)
                     + 2.*support_steps/args.steps
                     + 3.*supported_push_steps/args.steps
                     - 7.*(root_min < .50).float()
                     - 3.*(root_min < .40).float()
                     - 300.*object_inward_peak.clamp(0., .30)
                     - .08*params.square().sum(-1))
            if base_params is not None:
                half_x = args.object_width_x/2 if args.object_width_x is not None else .045
                retrieval_distance = args.object_x-(.28-half_x) if args.object_x is not None else .115
                pull_motion = (foot_x_at_pull_start-foot_x_at_end).clamp(0., .35)
                score = (score + 650.*object_best.clamp(0., .35)
                         + 25.*pull_motion
                         + 12.*(object_best > retrieval_distance).float()
                         + 3.*late_support_steps/max(1, args.steps-270)
                         - 400.*object_inward_peak.clamp(0., .30))
                valid = ((root_min > .52)
                         & (late_support_steps > .70*max(1, args.steps-270))
                         & (support_steps > .70*args.steps))
                score = torch.where(valid, score, torch.full_like(score, -1000.))
                if base_params.shape[0] in (5, 6):
                    final_outward = x0-object_asset.data.root_pos_w[:, 0]
                    score += torch.where(valid, 500.*final_outward.clamp(0., .35), 0.)
                    score += torch.where(valid, 18.*(final_outward > (x0-origin_x-.28)).float(), 0.)
                    score += torch.where(valid, 60.*(final_outward > retrieval_distance).float(), 0.)
                    score += torch.where(valid, 100.*((.30-back_local_x)/.03).clamp(-1., 1.5), 0.)
                    score += torch.where(valid, 100.*(back_local_x < .28).float(), 0.)
                    score += torch.where(valid, 15.*late_alignment_sum/max(1, args.steps-310), 0.)
            candidate_score_all = score.reshape(candidate_count, args.replicas).mean(1)
            top = torch.topk(candidate_score_all, elite_n)
            elites = candidate_params[top.indices]
            mean = .3*mean + .7*elites.mean(0)
            min_std = .18 if args.wide else .05
            std = (.25*std + .75*elites.std(0).clamp_min(min_std)).clamp(min_std, 1.0 if args.wide else .8)
            best_candidate_idx = int(candidate_score_all.argmax())
            gen_best = best_candidate_idx*args.replicas + int(
                score.reshape(candidate_count, args.replicas)[best_candidate_idx].argmax())
            candidate_score = float(candidate_score_all[best_candidate_idx])
            if candidate_score > best_score:
                best_score = candidate_score
                best_params = candidate_params[best_candidate_idx].clone()
                optimized_phases = (2 if base_params is not None and
                                    (base_params.shape[0] == 3 or args.fixed_five_seven is not None)
                                    else 1)
                saved_params = (best_params.reshape(3, 4) if base_params is None else
                                torch.cat((base_params, best_params.reshape(optimized_phases, n_joints)), dim=0))
                record = {"source_checkpoint": str(args.checkpoint.resolve()),
                          "joint_names": leg_joints,
                          "phase_times_s": ([1.5, 4.2] if base_params is None else
                                            [1.5, 4.2, 3.5, 5.6] if base_params.shape[0] == 3 else
                                            [1.5, 4.2, 3.5, 5.6, 6.1, 7.2] if args.fixed_five_seven is not None else
                                            [1.5, 4.2, 3.5, 5.6, 6.1] if base_params.shape[0] == 5 else
                                            [1.5, 4.2, 3.5, 5.6, 6.1, 7.2]),
                          "parameters": saved_params.cpu().tolist(),
                          "replicas": args.replicas,
                          "candidate_mean_score": candidate_score,
                          "score": best_score, "generation": gen,
                          "object_outward_m": float(object_best[gen_best]),
                          "object_final_outward_m": float((x0-object_asset.data.root_pos_w[:, 0])[gen_best]),
                          "object_final_backmost_x_m": float(back_local_x[gen_best]),
                          "object_inward_peak_m": float(object_inward_peak[gen_best]),
                          "foot_best_x_m": float(foot_best[gen_best]),
                          "safe_foot_best_x_m": float(safe_foot_best[gen_best]),
                          "foot_best_z_m": float(foot_lift[gen_best]),
                          "lifted_fraction": float(lifted_sum[gen_best]/150.),
                          "clearance_fraction": float(clearance_sum[gen_best]/150.),
                          "foot_behind_score": float(foot_behind_best[gen_best]),
                          "root_min_height_m": float(root_min[gen_best]),
                          "support_duration_s": float(support_steps[gen_best]*base.step_dt),
                          "foot_pull_motion_m": float((foot_x_at_pull_start-foot_x_at_end)[gen_best]),
                          "supported_push_duration_s": float(supported_push_steps[gen_best]*base.step_dt)}
                (args.output/"best.json").write_text(json.dumps(record, indent=2)+"\n")
            final_outward = x0-object_asset.data.root_pos_w[:, 0]
            retrieval_distance = (args.object_x-(.28-(args.object_width_x or .09)/2)
                                  if args.object_x is not None else .115)
            retrieved_valid = ((score > -999.) & (back_local_x < .28))
            if retrieved_valid.any():
                success_idx = int(torch.where(retrieved_valid, score,
                                              torch.full_like(score, -1e9)).argmax())
                success_params = (params[success_idx].reshape(3, 4) if base_params is None else
                                  torch.cat((base_params, params[success_idx].reshape(optimized_phases, n_joints)), dim=0))
                success_record = {"source_checkpoint": str(args.checkpoint.resolve()),
                                  "joint_names": leg_joints,
                                  "parameters": success_params.cpu().tolist(),
                                  "generation": gen,
                                  "object_outward_m": float(object_best[success_idx]),
                                  "object_final_outward_m": float(final_outward[success_idx]),
                                  "object_final_backmost_x_m": float(back_local_x[success_idx]),
                                  "root_min_height_m": float(root_min[success_idx]),
                                  "support_duration_s": float(support_steps[success_idx]*base.step_dt)}
                (args.output/f"success_gen{gen:02d}.json").write_text(
                    json.dumps(success_record, indent=2)+"\n")
            row = {"generation": gen, "best_score": best_score,
                   "generation_score": candidate_score,
                   "generation_object_outward_m": float(object_best[gen_best]),
                   "generation_object_final_outward_m": float((x0-object_asset.data.root_pos_w[:, 0])[gen_best]),
                   "generation_object_final_backmost_x_m": float(back_local_x[gen_best]),
                   "generation_object_inward_peak_m": float(object_inward_peak[gen_best]),
                   "generation_foot_best_x_m": float(foot_best[gen_best]),
                   "generation_safe_foot_best_x_m": float(safe_foot_best[gen_best]),
                   "generation_lifted_fraction": float(lifted_sum[gen_best]/150.),
                   "generation_clearance_fraction": float(clearance_sum[gen_best]/150.),
                   "generation_foot_behind_score": float(foot_behind_best[gen_best]),
                   "generation_root_min_m": float(root_min[gen_best]),
                   "generation_foot_pull_motion_m": float((foot_x_at_pull_start-foot_x_at_end)[gen_best]),
                   "population_valid": int((score > -999.).sum()),
                   "population_retrieved_valid": int(retrieved_valid.sum()),
                   "population_stable": int((root_min > .55).sum()),
                   "population_retrieved": int(((x0-object_asset.data.root_pos_w[:,0]) > .12).sum())}
            if args.grid_pull:
                grid = []
                for j in range(args.population):
                    grid.append({"env_id": j, "late_hip_pitch": float(params[j, 4]),
                                 "late_hip_roll": float(params[j, 5]),
                                 "grid_index_a": args.grid_index_a,
                                 "grid_index_b": args.grid_index_b,
                                 "grid_value_a": float(params[j, args.grid_index_a]),
                                 "grid_value_b": float(params[j, args.grid_index_b]),
                                 "score": float(score[j]), "object_best_m": float(object_best[j]),
                                 "object_final_outward_m": float(final_outward[j]),
                                 "root_min_m": float(root_min[j]),
                                 "support_duration_s": float(support_steps[j]*base.step_dt),
                                 "foot_final_y_m": float(foot_y_at_end[j]),
                                 "foot_pull_motion_m": float((foot_x_at_pull_start-foot_x_at_end)[j])})
                (args.output/"grid_results.json").write_text(json.dumps(grid, indent=2)+"\n")
            history.append(row)
            (args.output/"history.json").write_text(json.dumps(history, indent=2)+"\n")
            print("CEM "+json.dumps(row), flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    main()
