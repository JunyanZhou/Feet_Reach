"""Distill a physics-selected leg residual into the student policy network.

The teacher is the saved whole-body student plus CEM-trained left-leg actions.
Only public policy observations are used as student input. This script collects
teacher actions in free-base Isaac physics, selects supported retrievals, and
fits the network to those actions. Evaluation must omit the residual JSON.
"""
import argparse
import copy
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
import torch.nn.functional as F
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from isaaclab.app import AppLauncher
from omegaconf import OmegaConf
import active_adaptation.learning


def smooth(x):
    x = max(0.0, min(1.0, x))
    return x*x*(3.0-2.0*x)


def phase_weights(t, device):
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
    p.add_argument("--residual-json", type=Path, required=True)
    p.add_argument("--student-checkpoint", type=Path, default=None,
                   help="Continue distillation on states reached by this student")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=450)
    p.add_argument("--fit-steps", type=int, default=1200)
    p.add_argument("--leg-loss-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--object-x", type=float, default=.315)
    p.add_argument("--object-y", type=float, default=.08)
    p.add_argument("--object-width-x", type=float, default=.06)
    p.add_argument("--object-height", type=float, default=.12)
    p.add_argument("--object-mass", type=float, default=.08)
    p.add_argument("--object-friction", type=float, default=.15)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    residual_data = json.loads(args.residual_json.read_text())
    if len(residual_data["parameters"]) != 7:
        raise ValueError("Expected seven saved action phases")

    with hydra.initialize_config_dir(config_dir=str(ROOT / "cfg"), version_base=None):
        cfg = hydra.compose("train", overrides=[
            "task=G1/G1_table_retrieval_supported_rl", "+exp=finetune"])
    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint_path = str(args.checkpoint.resolve())
    cfg.task.num_envs = args.num_envs
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
        teacher_actor = agent.get_rollout_policy("eval")
        if args.student_checkpoint is not None:
            student_agent = copy.deepcopy(agent)
            student_state = torch.load(args.student_checkpoint, map_location="cpu",
                                       weights_only=False)["policy"]
            student_agent.load_state_dict(student_state)
            student_agent.eval()
            actor = student_agent.get_rollout_policy("eval")
        else:
            student_agent = agent
            actor = teacher_actor
        normalizer = vecnorm.to_observation_norm()
        normalizer.eval()
        residual = torch.tensor(residual_data["parameters"], device=base.device)
        indices = [base.action_manager.joint_names.index(name)
                   for name in residual_data["joint_names"]]
        env.reset()
        base.episode_length_buf.zero_()
        obj = base.scene["target_object"]
        origin_x = base.scene.env_origins[:, 0]
        initial_x = obj.data.root_pos_w[:, 0].clone()
        hand_sensor = base.scene["ee_contact_forces"]
        hand_id = hand_sensor.find_bodies("left_hand_mimic")[0][0]
        stance_sensor = base.scene["contact_forces"]
        root_min = torch.full((args.num_envs,), 10., device=base.device)
        support_steps = torch.zeros(args.num_envs, device=base.device)
        observations, labels = [], []
        teacher_actor.eval()
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            for i in range(args.steps):
                raw = TensorDict({}, batch_size=[args.num_envs], device=base.device)
                base._compute_observation(raw)
                carry = normalizer(raw).select("policy")
                carry["is_init"] = torch.full((args.num_envs, 1), i == 0,
                                              device=base.device, dtype=torch.bool)
                teacher_actor(carry)
                carry["action"][:, indices] += phase_weights(i*base.step_dt, base.device) @ residual
                observations.append(carry["policy"].detach().cpu().clone())
                labels.append(carry["action"].detach().cpu().clone())
                base.step(carry)
                root_z = cmd.asset.data.root_pos_w[:, 2]
                root_min = torch.minimum(root_min, root_z)
                hand_z = hand_sensor.data.net_forces_w[:, hand_id, 2]
                stance_z = stance_sensor.data.net_forces_w[:, cmd.right_foot_sensor_id, 2]
                support_steps += ((root_z > .50) & (hand_z > 5.) & (stance_z > 100.)).float()
        final_x = obj.data.root_pos_w[:, 0]
        selected = ((final_x < origin_x + .28) & (root_min > .50)
                    & (support_steps*base.step_dt > 5.))
        if selected.sum() < 8:
            raise RuntimeError(f"Only {int(selected.sum())} successful teacher rollouts")
        selected_mean_outward = float((initial_x-final_x)[selected].mean())
        obs = torch.stack(observations, 0)[:, selected.cpu()].reshape(-1, observations[0].shape[-1]).to(base.device)
        target = torch.stack(labels, 0)[:, selected.cpu()].reshape(-1, labels[0].shape[-1]).to(base.device)
        teacher_sample_count = len(obs)
        student_sample_count = 0
        if args.student_checkpoint is not None:
            env.reset()
            object_state = obj.data.default_root_state.clone()
            object_state[:, :3] += base.scene.env_origins
            obj.write_root_state_to_sim(object_state)
            base.episode_length_buf.zero_()
            student_observations, student_labels, student_valid = [], [], []
            actor.eval()
            with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                for i in range(args.steps):
                    raw = TensorDict({}, batch_size=[args.num_envs], device=base.device)
                    base._compute_observation(raw)
                    carry = normalizer(raw).select("policy")
                    carry["is_init"] = torch.full((args.num_envs, 1), i == 0,
                                                  device=base.device, dtype=torch.bool)
                    teacher_carry = carry.clone()
                    teacher_actor(teacher_carry)
                    teacher_carry["action"][:, indices] += phase_weights(
                        i*base.step_dt, base.device) @ residual
                    actor(carry)
                    student_observations.append(carry["policy"].detach().cpu().clone())
                    student_labels.append(teacher_carry["action"].detach().cpu().clone())
                    student_valid.append((cmd.asset.data.root_pos_w[:, 2] > .50).detach().cpu().clone())
                    base.step(carry)
            valid_mask = torch.stack(student_valid, 0)
            sx = torch.stack(student_observations, 0)[valid_mask].to(base.device)
            sy = torch.stack(student_labels, 0)[valid_mask].to(base.device)
            student_sample_count = len(sx)
            obs = torch.cat((obs, sx), 0)
            target = torch.cat((target, sy), 0)
            print(f"DISTILL_STUDENT samples={student_sample_count}", flush=True)
        report = {"teacher_checkpoint": str(args.checkpoint.resolve()),
                  "residual_json": str(args.residual_json.resolve()),
                  "teacher_center_crossing": int(selected.sum()),
                  "teacher_num_envs": args.num_envs,
                  "selected_samples": len(obs),
                  "teacher_samples": teacher_sample_count,
                  "student_samples": student_sample_count,
                  "student_checkpoint": (str(args.student_checkpoint.resolve())
                                         if args.student_checkpoint else None),
                  "selected_mean_outward_m": selected_mean_outward,
                  "fit_losses": []}
        print("DISTILL_TEACHER " + json.dumps({k: v for k, v in report.items()
                                                if k != "fit_losses"}), flush=True)
        actor.train()
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
        joint_weights = torch.ones(target.shape[-1], device=base.device)
        joint_weights[indices] = args.leg_loss_weight
        for i in range(args.fit_steps):
            ids = torch.randint(len(obs), (min(256, len(obs)),), device=base.device)
            td = TensorDict({"policy": obs[ids],
                             "is_init": torch.zeros(len(ids), 1, device=base.device,
                                                    dtype=torch.bool)},
                            batch_size=[len(ids)], device=base.device)
            with set_exploration_type(ExplorationType.MODE):
                actor(td)
            loss = (((td["loc"]-target[ids])**2)*joint_weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.)
            optimizer.step()
            if i % 100 == 0 or i == args.fit_steps-1:
                row = [i, float(loss.detach())]
                report["fit_losses"].append(row)
                leg_mse = float(F.mse_loss(td["loc"][:, indices], target[ids][:, indices]).detach())
                print(f"DISTILL_FIT step={i} loss={row[1]:.6f} leg_mse={leg_mse:.6f}", flush=True)
            if i in (299, 599, 1199, args.fit_steps-1):
                source = torch.load(args.student_checkpoint or args.checkpoint,
                                    map_location="cpu", weights_only=False)
                source["policy"] = student_agent.state_dict()
                source["distillation"] = {"teacher": report["teacher_checkpoint"],
                                         "residual": report["residual_json"],
                                         "selected_samples": len(obs),
                                         "fit_step": i+1,
                                         "loss": float(loss.detach())}
                output = args.output / f"distilled_{i+1}.pt"
                torch.save(source, output)
                print(f"DISTILL_CHECKPOINT {output}", flush=True)
        (args.output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    except BaseException:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    main()
