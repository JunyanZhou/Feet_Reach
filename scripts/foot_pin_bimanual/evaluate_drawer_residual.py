"""Held-out paired MuJoCo evaluation of a learned drawer residual policy."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.foot_pin_bimanual.train_drawer_residual import ActorCritic, DrawerEnv


def rollout(model, params, seed, scale, with_brace, record=False,
            start_state=None, phase='full', episode_end=4.6, action_until=None):
    env = DrawerEnv(params, np.random.default_rng(seed), with_brace=with_brace,
                    start_state=start_state, phase=phase, episode_end=episode_end)
    obs = env.reset(scale)
    frames = []
    while True:
        with torch.no_grad():
            action, _ = model(torch.from_numpy(obs[None]))
        command = action[0].numpy() if action_until is None or env.sim.data.time < action_until else np.zeros(8)
        obs, reward, done, info = env.step(command)
        if record:
            s = env.sim
            frames.append((s.data.time, s.data.qpos.copy(),
                           float(s.data.qpos[s.lid_qadr])))
        if done:
            break
    if record:
        return info, dict(time=np.array([f[0] for f in frames]),
                          qpos=np.array([f[1] for f in frames]),
                          lid_open=np.array([f[2] for f in frames]))
    return info, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--params', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=32)
    p.add_argument('--seed-start', type=int, default=10000)
    p.add_argument('--noise-scale', type=float, default=1.)
    p.add_argument('--start-state', type=Path)
    p.add_argument('--phase', choices=['full', 'approach'], default='full')
    p.add_argument('--episode-end', type=float, default=4.6)
    p.add_argument('--action-until', type=float)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    model = ActorCritic(saved['obs_dim'])
    model.load_state_dict(saved['model']);model.eval()
    source = json.loads(a.params.read_text())
    params = source.get('params', source)
    a.output.mkdir(parents=True, exist_ok=True)
    rows = []
    first_success = False
    with (a.output / 'episodes.jsonl').open('w') as out:
        for i in range(a.episodes):
            seed = a.seed_start + i
            with_info, trace = rollout(model, params, seed, a.noise_scale, True,
                                       record=not first_success, start_state=a.start_state,
                                       phase=a.phase, episode_end=a.episode_end,
                                       action_until=a.action_until)
            without_info, no_trace = rollout(model, params, seed, a.noise_scale, False,
                                             record=not first_success, start_state=a.start_state,
                                             phase=a.phase, episode_end=a.episode_end,
                                             action_until=a.action_until)
            if with_info['success'] and not first_success:
                first_success = True
                np.savez_compressed(a.output / 'first_success_with.npz', **trace)
                np.savez_compressed(a.output / 'first_success_without.npz', **no_trace)
            row = dict(seed=seed, with_support=with_info, without_support=without_info)
            rows.append(row)
            out.write(json.dumps(row, default=lambda x: x.item() if isinstance(x, np.generic) else x) + '\n');out.flush()
            if (i + 1) % 8 == 0:
                print('EVAL', i + 1, '/', a.episodes,
                      sum(r['with_support']['success'] for r in rows), flush=True)
    summary = dict(checkpoint=str(a.checkpoint), episodes=a.episodes,
                   start_state=str(a.start_state) if a.start_state else None,
                   phase=a.phase, episode_end=a.episode_end,
                   action_until=a.action_until,
                   seed_start=a.seed_start, noise_scale=a.noise_scale,
                   success_with=int(sum(r['with_support']['success'] for r in rows)),
                   success_without=int(sum(r['without_support']['success'] for r in rows)),
                   lid_mean_with_m=float(np.mean([r['with_support']['lid'] for r in rows])),
                   lid_mean_without_m=float(np.mean([r['without_support']['lid'] for r in rows])),
                   base_drift_mean_with_m=float(np.mean([r['with_support']['drift'] for r in rows])),
                   base_drift_mean_without_m=float(np.mean([r['without_support']['drift'] for r in rows])))
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
