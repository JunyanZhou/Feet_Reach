"""Isaac smoke check of the migrated four-target foot-reaching student."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE", "1")
os.environ.setdefault("MEMPATH", str(ROOT / "dataset"))

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
import active_adaptation.learning
from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aux-force", action="store_true",
                        help="Also exercise randomized multibody force application")
    parser.add_argument("--contact-compliance", action="store_true",
                        help="Exercise wall-filtered contact events and force costs")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "artifacts/unified_support_reach/v2_migrated.pt")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(OmegaConf.to_container(state["cfg"], resolve=True))
    OmegaConf.set_struct(cfg, False)
    cfg.algo.geometry_points = 0
    cfg.checkpoint_path = str(checkpoint)
    cfg.task.num_envs = 1
    cfg.task.command.student_train = True
    if args.aux_force:
        cfg.task.command.foot_force.enabled = False
        cfg.task.command.aux_support_force = {
            "enabled": True, "train_only": False,
            "body_names": ["left_hand_mimic", "torso_link", "pelvis"],
            "application_radius_m": .02, "force_range": [5., 10.],
            "ramp_up_range": [1, 1], "hold_range": [2, 2],
            "ramp_down_range": [1, 1], "rest_range": [1, 1],
            "zero_prob": 0.,
        }
    if args.contact_compliance:
        cfg.task.proactive_contact = True
        cfg.task.command.contact_compliance = {"enabled": True, "force_cap_N": 80.}
        cfg.task.reward.contact = {
            "wall_recontact_cost": {"weight": 1.},
            "wall_force_cap_cost": {"weight": .25},
        }
    cfg.vecnorm = "eval"
    cfg.wandb.mode = "disabled"
    cfg.app = {"headless": True, "enable_cameras": False}
    app = AppLauncher(headless=True, device="cuda:0").app
    try:
        from isaaclab.sim import SimulationContext
        from scripts.utils.helpers import make_env_policy
        env, agent, vecnorm, _ = make_env_policy(cfg)
        env.eval()
        env.base_env.eval()
        agent.eval()
        td = env.reset()
        cmd = env.base_env.command_manager
        policy_dim = int(state["vecnorm"]["_extra_state"]["policy_sum"].numel())
        assert td["policy"].shape == (1, policy_dim), td["policy"].shape
        actor = agent.get_rollout_policy("eval")
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            normalizer = vecnorm.to_observation_norm()
            raw_a = TensorDict({}, batch_size=[1], device=env.device)
            env.base_env._compute_observation(raw_a)
            baseline = normalizer(raw_a)
            baseline["is_init"] = td["is_init"].clone()
            a = actor(baseline.select("policy", "is_init").clone())["action"].clone()
            # Change all three new goals while keeping old proprioception and
            # the original left-foot target exactly unchanged.
            for name, delta in (
                ("root", (0.02, 0., 0.)),
                ("left_hand", (0., 0., 0.02)),
                ("right_foot", (0.01, 0., 0.)),
            ):
                target = cmd.goal_world(name) + torch.tensor(delta, device=cmd.device)
                cmd.set_goal_world(name, target)
            raw = TensorDict({}, batch_size=[1], device=env.device)
            env.base_env._compute_observation(raw)
            normalized = normalizer(raw)
            assert torch.equal(normalized["policy"][:, :241], baseline["policy"][:, :241]), (
                (normalized["policy"][:, :241] - baseline["policy"][:, :241]).abs().max().item())
            assert not torch.equal(normalized["policy"][:, 241:250], baseline["policy"][:, 241:250])
            normalized["is_init"] = td["is_init"].clone()
            b = actor(normalized.select("policy", "is_init").clone())["action"].clone()
            max_action_change = (a-b).abs().max().item()
            assert max_action_change < 1e-6, max_action_change
            env.base_env.step(normalized.set("action", b))
            assert torch.isfinite(cmd.asset.data.root_state_w).all()
            if args.aux_force:
                force_sizes = []
                body_ids = set()
                for _ in range(12):
                    env.base_env.step(normalized.set("action", b))
                    force_sizes.append(cmd.aux_force_ramp.force.norm(dim=-1).max().item())
                    if force_sizes[-1] > 0.:
                        body_ids.add(int(cmd.aux_body_id[0]))
                    assert torch.isfinite(cmd.asset.data.root_state_w).all()
                assert max(force_sizes) >= 5., force_sizes
                assert body_ids.issubset(set(cmd.aux_body_ids.tolist()))
            if args.contact_compliance:
                contact_peaks = []
                cap_costs = []
                for _ in range(12):
                    env.base_env.step(normalized.set("action", b))
                    contact_peaks.append(cmd.contact_peak_N.max().item())
                    cap_costs.append(cmd.wall_force_cap_cost().min().item())
                assert max(contact_peaks) > 1., contact_peaks
                assert torch.isfinite(cmd.wall_recontact_cost()).all()
                assert torch.isfinite(cmd.wall_force_cap_cost()).all()
                assert min(cap_costs) < 0., cap_costs
        result = {
            "checkpoint": str(checkpoint),
            "policy_dim": int(td["policy"].shape[-1]),
            "action_dim": int(a.shape[-1]),
            "zero_initialized_goal_action_change_max": max_action_change,
            "new_goal_observation_changed": True,
            "physics_step_finite": True,
        }
        if args.aux_force:
            result.update(aux_force_max_N=max(force_sizes),
                          aux_force_body_ids=sorted(body_ids))
        if args.contact_compliance:
            result.update(wall_contact_peak_N=max(contact_peaks),
                          wall_force_cap_cost_min=min(cap_costs),
                          contact_cost_finite=True)
        out = args.output or ROOT / "artifacts/unified_support_reach" / (
            "smoke_contact.json" if args.contact_compliance else
            "smoke_aux_force.json" if args.aux_force else "smoke.json")
        out.write_text(json.dumps(result, indent=2) + "\n")
        print("UNIFIED_SUPPORT_SMOKE_OK", json.dumps(result), flush=True)
        SimulationContext.instance()._disable_app_control_on_stop_handle = True
        env.close()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        app.close()


if __name__ == "__main__":
    main()
