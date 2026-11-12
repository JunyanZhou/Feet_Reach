"""PPO curriculum for an 8D Cartesian residual on the physical drawer task.

The pretrained balance student remains active. PPO only changes two palm
references and the bracing foot reference; all forces come from MuJoCo contacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from scripts.foot_pin_bimanual.mujoco_task import TaskSimulation


DURATION = 4.6


class DrawerEnv:
    def __init__(self, params: dict, rng: np.random.Generator, with_brace: bool = True,
                 start_state: Path | None = None, phase: str = 'full',
                 episode_end: float = DURATION):
        self.original = params
        self.rng = rng
        self.with_brace = with_brace
        self.start_state = start_state
        self.phase = phase
        self.episode_end = episode_end
        self.sim = None

    def reset(self, noise_scale: float):
        seed = int(self.rng.integers(0, 2**31 - 1))
        p = dict(self.original)
        p['root_noise_m'] = .01 * noise_scale
        p['joint_noise_rad'] = .01 * noise_scale
        p['base_noise_m'] = .015 * noise_scale
        p['base_friction'] *= self.rng.uniform(1 - .1 * noise_scale,
                                                1 + .1 * noise_scale)
        p['lid_friction'] *= self.rng.uniform(1 - .1 * noise_scale,
                                               1 + .1 * noise_scale)
        self.sim = TaskSimulation(self.with_brace, p, seed=seed,
                                  randomize=noise_scale > 0 and self.start_state is None)
        if self.start_state is not None:
            self.sim.restore(np.load(self.start_state), seed=seed,
                             randomize=noise_scale > 0)
        self.last_lid = float(self.sim.data.qpos[self.sim.lid_qadr])
        self.base_max_drift = 0.
        self.root_min_z = float(self.sim.data.qpos[2])
        self.three_way_s = 0.
        self.three_way_motion = 0.
        self.both_hands_s = 0.
        self.foot_contact_s = 0.
        return self.observe()

    def observe(self):
        s = self.sim
        base = s.data.xpos[s.base_id]
        hand = s.hand_positions()
        pad = {side: s.data.geom_xpos[gid] for side, gid in s.pad_ids.items()}
        base_dof = s.model.jnt_dofadr[s.model.joint('base_free').id]
        c = s.contacts()
        brace_point = (s.data.geom_xpos[s.brace_geom] if s.with_brace
                       else s.data.xpos[s.base_id] + s.data.xmat[s.base_id].reshape(3, 3) @
                       [s.params.get('toe_wall_x', -.37), .10, -.305])
        return np.concatenate((
            [s.data.time / DURATION],
            (s.data.qpos[:3] - [0, 0, .74]) / [.4, .4, .2],
            s.data.qpos[3:7],
            s.data.qvel[:6] / 3.,
            (base - s.base_initial) / .2,
            s.data.qvel[base_dof:base_dof + 6] / 3.,
            [s.data.qpos[s.lid_qadr] / .19,
             s.data.qvel[s.model.jnt_dofadr[s.lid_id]]],
            (hand['left'] - pad['left']) / .3,
            (hand['right'] - pad['right']) / .3,
            (s.data.geom_xpos[s.toe_geom] - brace_point) / .2,
            [float(c['left_hand']), float(c['right_hand']), float(c['foot_brace'])],
            (s.data.qpos[s.qadr] - s.q0) / 2.,
            s.data.qvel[s.vadr] / 6.,
        )).astype(np.float32)

    def step(self, action):
        s = self.sim
        s.residual_action[:] = np.clip(action, -1., 1.)
        for _ in range(s.decimation):
            s.step(np.zeros_like(s.forces))
            if s.first_fall is not None:
                break
        c = s.contacts()
        both = c['left_hand'] and c['right_hand']
        three = both and c['foot_brace']
        lid = float(s.data.qpos[s.lid_qadr])
        delta_lid = lid - self.last_lid
        self.last_lid = lid
        drift = float(np.linalg.norm(s.data.xpos[s.base_id] - s.base_initial))
        self.base_max_drift = max(self.base_max_drift, drift)
        self.root_min_z = min(self.root_min_z, float(s.data.qpos[2]))
        if three:
            self.three_way_s += s.meta['policy_dt']
            self.three_way_motion += max(0., delta_lid)
        if both:
            self.both_hands_s += s.meta['policy_dt']
        if c['foot_brace']:
            self.foot_contact_s += s.meta['policy_dt']
        hand = s.hand_positions()
        distances = [np.linalg.norm(hand[side] - s.data.geom_xpos[s.pad_ids[side]])
                     for side in ('left', 'right')]
        hand_reward = sum(np.exp(-d / .09) for d in distances)
        if self.phase == 'approach':
            reward = (.09 * hand_reward + .035 * c['foot_brace'] +
                      .18 * both + .20 * three -
                      1.5 * max(0., drift - .02) -
                      .6 * max(0., .7 - float(s.data.qpos[2])) -
                      .001 * float(np.dot(action, action)))
        else:
            reward = (0.035 * hand_reward + .025 * c['foot_brace'] +
                      .08 * both + .12 * three +
                      100 * max(0., delta_lid) + 20 * min(0., delta_lid) -
                      .8 * max(0., drift - .02) -
                      .6 * max(0., .7 - float(s.data.qpos[2])) -
                      .001 * float(np.dot(action, action)))
        done = s.first_fall is not None or s.data.time >= self.episode_end - 1e-6
        if self.phase == 'approach':
            success = (distances[0] < .13 and distances[1] < .10 and
                       self.foot_contact_s >= 1. and
                       self.base_max_drift < .05 and self.root_min_z > .55)
        else:
            success = (lid >= s.params.get('open_threshold_m', .12) and
                       self.base_max_drift < .05 and self.root_min_z > .55 and
                       self.three_way_s >= .15 and self.three_way_motion >= .01)
        if done:
            reward += (0 if self.phase == 'approach' else 20 * lid) + (25 if success else 0)
            reward -= 15 * max(0., self.base_max_drift - .05)
            if s.first_fall is not None:
                reward -= 10
        info = dict(success=success, lid=lid, drift=self.base_max_drift,
                    three_way_s=self.three_way_s, both_hands_s=self.both_hands_s,
                    foot_contact_s=self.foot_contact_s,
                    fell=s.first_fall is not None)
        return self.observe(), float(reward), done, info


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, 128), nn.Tanh(),
                                  nn.Linear(128, 128), nn.Tanh())
        self.actor = nn.Linear(128, 8)
        self.critic = nn.Linear(128, 1)
        self.logstd = nn.Parameter(torch.full((8,), -1.7))
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)

    def forward(self, obs):
        h = self.body(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    def distribution(self, obs):
        mean, value = self(obs)
        return Normal(mean, self.logstd.exp()), value


def noise_curriculum(iteration):
    if iteration < 10:
        return 0.
    if iteration < 25:
        return .05
    if iteration < 45:
        return .1
    if iteration < 70:
        return .25
    if iteration < 95:
        return .5
    return 1.


def train(args):
    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    source = json.loads(args.params.read_text())
    params = source.get('params', source)
    envs = [DrawerEnv(params, rng, start_state=args.start_state,
                      phase=args.phase, episode_end=args.episode_end) for _ in range(args.envs)]
    obs = np.stack([env.reset(0.) for env in envs])
    model = ActorCritic(obs.shape[1])
    if args.init_checkpoint is not None:
        saved = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
        if saved['obs_dim'] != obs.shape[1]:
            raise ValueError('Checkpoint observation size does not match the task')
        model.load_state_dict(saved['model'])
    if args.init_logstd is not None:
        with torch.no_grad():
            model.logstd.fill_(args.init_logstd)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    progress = (args.output / 'progress.jsonl').open('w')
    episodes = []
    for iteration in range(args.iterations):
        global_iteration = args.curriculum_offset + iteration
        scale = noise_curriculum(global_iteration)
        batch_obs, batch_actions, batch_logp = [], [], []
        batch_values, batch_rewards, batch_dones = [], [], []
        for _ in range(args.horizon):
            with torch.no_grad():
                dist, value = model.distribution(torch.from_numpy(obs))
                act = dist.sample()
                logp = dist.log_prob(act).sum(-1)
            actions = act.numpy()
            next_obs, rewards, dones = [], [], []
            for j, env in enumerate(envs):
                nxt, reward, done, info = env.step(actions[j])
                if done:
                    episodes.append(info)
                    nxt = env.reset(scale * rng.uniform(.5, 1.))
                next_obs.append(nxt)
                rewards.append(reward)
                dones.append(done)
            batch_obs.append(obs.copy())
            batch_actions.append(actions.copy())
            batch_logp.append(logp.numpy().copy())
            batch_values.append(value.numpy().copy())
            batch_rewards.append(rewards)
            batch_dones.append(dones)
            obs = np.stack(next_obs)
        with torch.no_grad():
            _, last_value = model(torch.from_numpy(obs))
        values = np.asarray(batch_values)
        rewards = np.asarray(batch_rewards)
        dones = np.asarray(batch_dones, dtype=np.float32)
        advantage = np.zeros_like(rewards)
        gae = np.zeros(args.envs)
        for t in reversed(range(args.horizon)):
            next_v = last_value.numpy() if t == args.horizon - 1 else values[t + 1]
            delta = rewards[t] + args.gamma * (1 - dones[t]) * next_v - values[t]
            gae = delta + args.gamma * args.gae_lambda * (1 - dones[t]) * gae
            advantage[t] = gae
        returns = advantage + values
        bobs = torch.from_numpy(np.asarray(batch_obs).reshape(-1, obs.shape[1]))
        bactions = torch.from_numpy(np.asarray(batch_actions).reshape(-1, 8))
        blogp = torch.from_numpy(np.asarray(batch_logp).reshape(-1))
        badv = torch.from_numpy(advantage.reshape(-1).astype(np.float32))
        breturns = torch.from_numpy(returns.reshape(-1).astype(np.float32))
        badv = (badv - badv.mean()) / (badv.std() + 1e-8)
        n = len(bobs)
        for _ in range(args.epochs):
            for ids in torch.randperm(n).split(args.minibatch):
                dist, v = model.distribution(bobs[ids])
                ratio = (dist.log_prob(bactions[ids]).sum(-1) - blogp[ids]).exp()
                surrogate = torch.minimum(ratio * badv[ids],
                                          ratio.clamp(.8, 1.2) * badv[ids])
                loss = (-surrogate.mean() + .5 * (v - breturns[ids]).square().mean()
                        - .002 * dist.entropy().sum(-1).mean())
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), .5)
                optimizer.step()
        recent = episodes[-32:]
        row = dict(iteration=global_iteration + 1, noise_scale=scale,
                   episode_count=len(episodes),
                   recent_success=int(sum(e['success'] for e in recent)),
                   recent_count=len(recent),
                   recent_lid_mean_m=float(np.mean([e['lid'] for e in recent])) if recent else None,
                   recent_drift_mean_m=float(np.mean([e['drift'] for e in recent])) if recent else None)
        progress.write(json.dumps(row) + '\n');progress.flush()
        if (iteration + 1) % 10 == 0:
            torch.save(dict(model=model.state_dict(), obs_dim=obs.shape[1],
                            iteration=global_iteration + 1), args.output / f'checkpoint_{global_iteration + 1:04d}.pt')
            print('TRAIN', json.dumps(row), flush=True)
    torch.save(dict(model=model.state_dict(), obs_dim=obs.shape[1],
                    iteration=args.curriculum_offset + args.iterations), args.output / 'final.pt')
    progress.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=120)
    parser.add_argument('--envs', type=int, default=8)
    parser.add_argument('--horizon', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--minibatch', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--gae-lambda', type=float, default=.95)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--curriculum-offset', type=int, default=0)
    parser.add_argument('--init-logstd', type=float)
    parser.add_argument('--start-state', type=Path)
    parser.add_argument('--phase', choices=['full', 'approach'], default='full')
    parser.add_argument('--episode-end', type=float, default=DURATION)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
