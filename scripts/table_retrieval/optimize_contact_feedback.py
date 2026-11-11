"""Train a small object-relative foot correction around the supported student.

The frozen neural policy controls every joint. A learned residual follows the
moving object's lateral position and extends the pull while Isaac simulates the
free-base robot, table, and dynamic block. Each CEM candidate is evaluated in
multiple physics environments before it can become the best checkpoint.
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
import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from isaaclab.app import AppLauncher
from omegaconf import OmegaConf
import active_adaptation.learning
from scripts.table_retrieval.contact_feedback import ACTION_JOINTS, contact_correction


def smooth(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x*x*(3.0-2.0*x)


def phase_weights(t: float, device: torch.device) -> torch.Tensor:
    return torch.tensor((
        smooth(t/.30)*(1-smooth((t-1.5)/.5)),
        smooth((t-1.2)/.5)*(1-smooth((t-4.2)/.5)),
        smooth((t-3.9)/.5),
        smooth((t-3.5)/.4)*(1-smooth((t-5.6)/.4)),
        smooth((t-5.3)/.4),
        smooth((t-6.1)/.4),
        smooth((t-7.2)/.5),
    ), device=device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--base-residual", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--population", type=int, default=64)
    p.add_argument("--replicas", type=int, default=4)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--steps", type=int, default=450)
    p.add_argument("--seed", type=int, default=61)
    p.add_argument("--object-x", type=float, default=.345)
    p.add_argument("--object-y", type=float, default=.08)
    p.add_argument("--object-width-x", type=float, default=.09)
    p.add_argument("--version", type=int, choices=(2, 3), default=3)
    p.add_argument("--object-height", type=float, default=.12)
    p.add_argument("--object-mass", type=float, default=.08)
    p.add_argument("--object-friction", type=float, default=.15)
    args = p.parse_args()
    if args.population % args.replicas or args.population//args.replicas < 8:
        raise ValueError("population must contain at least eight complete replica groups")
    args.output.mkdir(parents=True, exist_ok=True)
    base_payload = json.loads(args.base_residual.read_text())
    joint_names = base_payload["joint_names"]
    if len(base_payload["parameters"]) != 7 or len(joint_names) != 5:
        raise ValueError("base-residual must contain seven phases and five leg joints")

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
            obj.pos[0] = args.object_x
            obj.pos[1] = args.object_y
            obj.size[0] = args.object_width_x
            obj.size[2] = args.object_height
            obj.pos[2] = args.object_height/2 + .001
            obj.mass = args.object_mass
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
        device = base.device
        residual = torch.tensor(base_payload["parameters"], device=device)
        residual_indices = [base.action_manager.joint_names.index(n) for n in joint_names]
        feedback_indices = [base.action_manager.joint_names.index(n) for n in ACTION_JOINTS]
        object_asset = base.scene["target_object"]
        hand_sensor = base.scene["ee_contact_forces"]
        hand_id = hand_sensor.find_bodies("left_hand_mimic")[0][0]
        stance_sensor = base.scene["contact_forces"]
        origin_x = base.scene.env_origins[:, 0]
        candidates = args.population//args.replicas
        elite_n = max(4, candidates//4)
        mean = torch.tensor([.06, 0., 0., 0., 0.], device=device)
        std = torch.tensor([.02, .35, .35, .30, .20], device=device)
        best_score = -1e9
        best_params = None
        history = []

        for gen in range(args.generations):
            candidate_params = mean + torch.randn(candidates, 5, device=device)*std
            candidate_params[:, 0].clamp_(0.0, .12)
            candidate_params[:, 1:4].clamp_(-1.5, 1.5)
            candidate_params[:, 4].clamp_(-.8, .8)
            candidate_params[0] = mean
            if best_params is not None:
                candidate_params[1] = best_params
            params = candidate_params.repeat_interleave(args.replicas, dim=0)
            env.reset()
            # The custom dynamic object is not restored by this environment's
            # reset hook. Restore its pose and velocity before every CEM round.
            object_state = object_asset.data.default_root_state.clone()
            object_state[:, :3] += base.scene.env_origins
            object_asset.write_root_state_to_sim(object_state)
            base.episode_length_buf.zero_()
            x0 = object_asset.data.root_pos_w[:, 0].clone()
            root_min = torch.full((args.population,), 10., device=device)
            support_steps = torch.zeros(args.population, device=device)
            late_support_steps = torch.zeros(args.population, device=device)
            supported_outward = torch.zeros(args.population, device=device)
            inward_peak = torch.zeros(args.population, device=device)
            supported_motion_steps = torch.zeros(args.population, device=device)
            lateral_error_sum = torch.zeros(args.population, device=device)
            previous_x = x0.clone()
            with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                for i in range(args.steps):
                    raw = TensorDict({}, batch_size=[args.population], device=device)
                    base._compute_observation(raw)
                    carry = normalizer(raw).select("policy")
                    carry["is_init"] = torch.full(
                        (args.population, 1), i == 0, device=device, dtype=torch.bool)
                    actor(carry)
                    t = i*base.step_dt
                    carry["action"][:, residual_indices] += phase_weights(t, device) @ residual
                    foot_before = cmd.asset.data.body_pos_w[:, cmd.left_asset]
                    carry["action"][:, feedback_indices] += contact_correction(
                        params, foot_before, object_asset.data.root_pos_w, x0, t,
                        version=args.version)
                    base.step(carry)
                    robot = cmd.asset.data
                    foot = robot.body_pos_w[:, cmd.left_asset]
                    object_pos = object_asset.data.root_pos_w
                    object_x = object_pos[:, 0]
                    root_z = robot.root_pos_w[:, 2]
                    root_min = torch.minimum(root_min, root_z)
                    hand_z = hand_sensor.data.net_forces_w[:, hand_id, 2]
                    stance_z = stance_sensor.data.net_forces_w[:, cmd.right_foot_sensor_id, 2]
                    supported = (root_z > .50) & (hand_z > 5.) & (stance_z > 100.)
                    support_steps += supported.float()
                    if i >= 260:
                        late_support_steps += supported.float()
                        lateral_error_sum += ((foot[:, 1]-object_pos[:, 1]).abs()/.16).clamp(0., 1.)
                    near_foot = ((foot-object_pos).abs() < torch.tensor([.18, .15, .20], device=device)).all(-1)
                    supported_outward = torch.maximum(
                        supported_outward, torch.where(supported & near_foot, x0-object_x, 0.))
                    inward_peak = torch.maximum(inward_peak, (object_x-x0).clamp_min(0.))
                    supported_motion_steps += (supported & near_foot &
                                               (object_x < previous_x-.0001)).float()
                    previous_x.copy_(object_x)
            final_outward = x0-object_asset.data.root_pos_w[:, 0]
            center_out = (final_outward > x0-origin_x-.28)
            quat = object_asset.data.root_quat_w
            qw, qx, qy, qz = (quat[:, j] for j in range(4))
            half_extent_x = ((1-2*(qy*qy+qz*qz)).abs()*args.object_width_x/2
                             + (2*(qx*qy-qz*qw)).abs()*.09/2
                             + (2*(qx*qz+qy*qw)).abs()*args.object_height/2)
            fully_out = (object_asset.data.root_pos_w[:, 0]+half_extent_x < origin_x+.28)
            valid = ((root_min > .52) & (support_steps > .7*args.steps)
                     & (late_support_steps > .7*max(1, args.steps-260)))
            score = (550.*final_outward.clamp(0., .25)
                     + 200.*supported_outward.clamp(0., .25)
                     + 35.*center_out.float() + 80.*fully_out.float()
                     + 10.*support_steps/args.steps
                     + 5.*supported_motion_steps/args.steps
                     - 300.*inward_peak.clamp(0., .25)
                     - 3.*lateral_error_sum/max(1, args.steps-260)
                     - .3*params[:, 1:].square().sum(-1))
            score = torch.where(valid, score, torch.full_like(score, -500.))
            group_scores = score.reshape(candidates, args.replicas)
            ordered = group_scores.sort(dim=1).values
            # The best single contact in a group can be a rare collision.
            # Select for the weaker three of four replicas as well.
            group_score = (.25*group_scores.mean(1)
                           + .75*ordered[:, :max(1, args.replicas-1)].mean(1))
            elite_idx = torch.topk(group_score, elite_n).indices
            elites = candidate_params[elite_idx]
            mean = .35*mean + .65*elites.mean(0)
            floor = torch.tensor([.008, .08, .08, .08, .05], device=device)
            ceiling = torch.tensor([.04, .9, .9, .8, .5], device=device)
            std = (.3*std + .7*elites.std(0).clamp_min(floor)).clamp(floor, ceiling)
            group_best = int(group_score.argmax())
            representative = group_best*args.replicas + int(
                group_scores[group_best].argmax())
            candidate_score = float(group_score[group_best])
            if candidate_score > best_score:
                best_score = candidate_score
                best_params = candidate_params[group_best].clone()
                saved = dict(base_payload)
                saved["contact_feedback"] = {
                    "version": args.version,
                    "parameters": best_params.cpu().tolist(),
                    "action_joint_names": ACTION_JOINTS,
                }
                saved["feedback_search"] = {
                    "generation": gen, "replicas": args.replicas,
                    "group_score": candidate_score,
                    "representative_outward_m": float(final_outward[representative]),
                    "representative_root_min_m": float(root_min[representative]),
                    "object_x_m": args.object_x, "object_y_m": args.object_y,
                    "object_width_x_m": args.object_width_x,
                    "object_mass_kg": args.object_mass,
                    "object_friction": args.object_friction,
                }
                (args.output/"best.json").write_text(json.dumps(saved, indent=2)+"\n")
            row = {
                "generation": gen, "best_group_score": best_score,
                "generation_group_score": candidate_score,
                "group_best_center_out": int(center_out.reshape(candidates, args.replicas)[group_best].sum()),
                "group_best_full_out": int(fully_out.reshape(candidates, args.replicas)[group_best].sum()),
                "group_best_valid": int(valid.reshape(candidates, args.replicas)[group_best].sum()),
                "population_center_out_valid": int((center_out & valid).sum()),
                "population_full_out_valid": int((fully_out & valid).sum()),
                "representative_outward_m": float(final_outward[representative]),
                "representative_root_min_m": float(root_min[representative]),
                "group_final_outward_m": final_outward.reshape(candidates, args.replicas)[group_best].cpu().tolist(),
                "group_start_local_x_m": (x0-origin_x).reshape(candidates, args.replicas)[group_best].cpu().tolist(),
                "group_scores": group_scores[group_best].cpu().tolist(),
            }
            history.append(row)
            (args.output/"history.json").write_text(json.dumps(history, indent=2)+"\n")
            print("FEEDBACK_CEM "+json.dumps(row), flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    main()
