import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import numpy as np
from spectral_braess_core import GridNetwork
from grid_braess_env import CongestedGridEnv
from planning_agents import StandardPrioritizedSweepingAgent, SLPSAgent

def run_proximity_rollout(seed):
    rng = random.Random(seed)
    m, n = 8, 16
    s_bg = (rng.randint(1, 2), rng.randint(2, 4))
    t_bg = (rng.randint(5, 7), rng.randint(10, 13))
    q = round(rng.uniform(2.5, 3.5), 2)

    env_std = CongestedGridEnv(m=m, n=n, s_bg=s_bg, t_bg=t_bg, q=q, p_chord=0.0, noise_std=0.0, seed=seed)
    env_sl = CongestedGridEnv(m=m, n=n, s_bg=s_bg, t_bg=t_bg, q=q, p_chord=0.0, noise_std=0.0, seed=seed)
    env_std._check_and_spawn_chord = lambda: None
    env_sl._check_and_spawn_chord = lambda: None

    grid = env_std.grid
    all_chords = grid.find_all_chords()
    braess_list = all_chords["braess_capable"]
    shortcut_list = all_chords["shortcut"]

    std = StandardPrioritizedSweepingAgent(env_std, backup_budget=30)
    sl = SLPSAgent(env_sl, backup_budget=30, beta_scale=8.0)

    ratio = float(n) / float(m)
    def natural_bellman(agent, s, is_sl=False):
        valid = agent.env.get_valid_actions(s)
        best_v = float("inf")
        best_a = None
        Phi_spec = agent.compute_spectral_congestion_field() if is_sl else None
        beta = agent.beta_scale * (1.0 + agent.slem) if is_sl else 0.0

        for a in valid:
            nxt, cost = agent._get_transition(s, a)
            spec_pen = float(beta * Phi_spec[agent.env.grid.node_to_idx[nxt]]) if is_sl else 0.0
            target = cost + agent.gamma * agent.V[nxt] + spec_pen

            if target < best_v - 1e-4:
                best_v = target
                best_a = a
            elif abs(target - best_v) <= 1e-4:
                curr_nxt, _ = agent._get_transition(s, best_a) if best_a is not None else (s, 0)
                curr_dist = abs(curr_nxt[1] - ratio * curr_nxt[0])
                new_dist = abs(nxt[1] - ratio * nxt[0])
                if new_dist < curr_dist:
                    best_v = target
                    best_a = a
        return best_v, best_a

    s_std = (0, 0)
    s_sl = (0, 0)
    env_std.agent_pos = s_std
    env_sl.agent_pos = s_sl

    traj_std = [s_std]
    traj_sl = [s_sl]
    lat_std = [0.0]
    lat_sl = [0.0]

    # Spawn schedule:
    # Ensure at least 1 Braess chord (V < 0) appears when agent is around step 4-8
    # Ensure at least 1 Shortcut (V > 0) appears when agent is around step 10-15
    braess_spawned = False
    shortcut_spawned = False
    active_chord = None
    chord_type = None

    for step in range(1, 26):
        # Spawn check based on agent's current position s_std
        p_event = 0.45
        if not braess_spawned and 4 <= step <= 10:
            # Look for Braess chord whose origin u is within Delta=(1, 2) of agent
            near_b = [
                c for c in braess_list
                if (s_std[0] <= c[0][0] <= s_std[0] + 1 and s_std[1] <= c[0][1] <= s_std[1] + 2)
                and (s_bg[0] <= c[0][0] <= t_bg[0] and s_bg[1] <= c[0][1] <= t_bg[1])
            ]
            if near_b and (rng.random() < p_event or step >= 8):
                active_chord = rng.choice(near_b)
                chord_type = "braess"
                braess_spawned = True
                # Re-equilibrate
                u_c, v_c, _, _ = active_chord
                f_aug, _, z_star = grid.solve_augmented_equilibrium(u_c, v_c, c=0.0, d=0.0)
                for env, agent in [(env_std, std), (env_sl, sl)]:
                    env.active_chord = active_chord
                    env.chord_type = chord_type
                    env.current_edge_flows = f_aug[:-1]
                    env.current_chord_flow = z_star
                    agent.plan()

        elif braess_spawned and not shortcut_spawned and step >= 11:
            # Look for Shortcut chord (V > 0) near agent that moves toward goal
            cur_pos = s_sl if s_sl != (m, n) else s_std
            near_s = [
                c for c in shortcut_list
                if (cur_pos[0] <= c[0][0] <= cur_pos[0] + 2 and cur_pos[1] <= c[0][1] <= cur_pos[1] + 3)
                and (c[1][0] >= c[0][0] and c[1][1] >= c[0][1]) and c[2] > 0
            ]
            if near_s and (rng.random() < p_event or step >= 15):
                active_chord = rng.choice(near_s)
                chord_type = "shortcut"
                shortcut_spawned = True
                u_c, v_c, _, _ = active_chord
                f_aug, _, z_star = grid.solve_augmented_equilibrium(u_c, v_c, c=0.0, d=0.0)
                for env, agent in [(env_std, std), (env_sl, sl)]:
                    env.active_chord = active_chord
                    env.chord_type = chord_type
                    env.current_edge_flows = f_aug[:-1]
                    env.current_chord_flow = z_star
                    agent.plan()

        # Step agents
        if s_std != (m, n):
            _, a_std = natural_bellman(std, s_std, is_sl=False)
            nxt_std, cost_std, _, _ = env_std.step(a_std)
            traj_std.append(nxt_std)
            lat_std.append(cost_std)
            s_std = nxt_std

        if s_sl != (m, n):
            _, a_sl = natural_bellman(sl, s_sl, is_sl=True)
            nxt_sl, cost_sl, _, _ = env_sl.step(a_sl)
            traj_sl.append(nxt_sl)
            lat_sl.append(cost_sl)
            s_sl = nxt_sl

        if s_std == (m, n) and s_sl == (m, n):
            break

    total_std = sum(lat_std)
    total_sl = sum(lat_sl)
    diff = total_std - total_sl
    pct = (diff / total_std) * 100 if total_std > 0 else 0
    print(f"Seed {seed:4d} | B-Spawned: {braess_spawned} | S-Spawned: {shortcut_spawned} | Std: {total_std:5.2f}s | SL: {total_sl:5.2f}s | SL diff: {diff:+5.2f}s ({pct:+5.1f}%)")
    return diff

print("Testing 10 random seeds with proximity spawning:")
diffs = [run_proximity_rollout(s) for s in [10, 20, 30, 42, 77, 99, 123, 2024, 5555, 9999]]
print(f"Average time saved by SL-PS: {np.mean(diffs):.2f}s")
