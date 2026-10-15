import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from spectral_braess_core import GridNetwork
from grid_braess_env import CongestedGridEnv
from planning_agents import StandardPrioritizedSweepingAgent, SLPSAgent

def test_4action_environment(seed=55):
    m, n = 8, 16
    # Exact scenario of seed 55:
    # Run single simulation of seed 55 to get exact scenario
    from animate_random_generator import run_single_simulation
    env_dag, scen, t_std_dag, t_sl_dag, _, _, c_std_dag, c_sl_dag, _ = run_single_simulation(seed=seed)
    
    b_chord, t_b = scen["braess_record"]
    s_chord, t_s = scen["shortcut_record"]
    s_bg = scen["s_bg"]
    t_bg = scen["t_bg"]
    q = scen["q"]

    print(f"=== Seed {seed} Scenario ===")
    print(f"BG: {s_bg} -> {t_bg}, q={q}")
    print(f"Braess:  {b_chord[0]} -> {b_chord[1]} (V={b_chord[2]:.3f}, spawn t={t_b})")
    print(f"Express: {s_chord[0]} -> {s_chord[1]} (V=+{s_chord[2]:.3f}, spawn t={t_s})")
    print(f"Old DAG (2-actions) results:")
    print(f"  Standard-PS: {c_std_dag[-1]:.2f}s")
    print(f"  SL-PS:       {c_sl_dag[-1]:.2f}s  (SL-PS was SLOWER by {c_sl_dag[-1] - c_std_dag[-1]:.2f}s!)")

    # Now let's implement 4-actions (Down, Right, Up, Left)
    ACTION_DOWN = 0
    ACTION_RIGHT = 1
    ACTION_UP = 2
    ACTION_LEFT = 3
    ACTION_CHORD = 4

    # Build 4-directional transitions
    def get_valid_actions_4dir(pos, active_c):
        i, j = pos
        acts = []
        if i < m: acts.append(ACTION_DOWN)
        if j < n: acts.append(ACTION_RIGHT)
        if i > 0: acts.append(ACTION_UP)
        if j > 0: acts.append(ACTION_LEFT)
        if active_c is not None and active_c[0] == pos:
            acts.append(ACTION_CHORD)
        return acts

    def transition_4dir(pos, a, env, active_c, chord_type):
        i, j = pos
        if a == ACTION_DOWN:
            nxt = (i + 1, j)
            e_idx = env._get_edge_index(pos, nxt)
            cost = env.grid.a * env.current_edge_flows[e_idx] + env.grid.b
        elif a == ACTION_RIGHT:
            nxt = (i, j + 1)
            e_idx = env._get_edge_index(pos, nxt)
            cost = env.grid.a * env.current_edge_flows[e_idx] + env.grid.b
        elif a == ACTION_UP:
            nxt = (i - 1, j)
            # Reverse edge (i-1, j) -> (i, j) on base grid
            # If traffic is on base edge, cost = a * f + b
            try:
                e_idx = env._get_edge_index(nxt, pos)
                cost = env.grid.a * env.current_edge_flows[e_idx] + env.grid.b
            except:
                cost = 1.0
        elif a == ACTION_LEFT:
            nxt = (i, j - 1)
            try:
                e_idx = env._get_edge_index(nxt, pos)
                cost = env.grid.a * env.current_edge_flows[e_idx] + env.grid.b
            except:
                cost = 1.0
        elif a == ACTION_CHORD:
            u_c, v_c, _, k = active_c
            nxt = v_c
            if chord_type == "braess":
                z = getattr(env, "current_chord_flow", 0.1)
                cost = float(k) * env.grid.b + 1.5 + 15.0 * z
            else:
                cost = 0.5
        return nxt, cost

    # Test SL-PS with 4-actions on Seed 55
    env_sl = CongestedGridEnv(m=m, n=n, s_bg=s_bg, t_bg=t_bg, q=q, p_chord=0.0, noise_std=0.0, seed=seed)
    env_sl._check_and_spawn_chord = lambda: None
    sl = SLPSAgent(env_sl, backup_budget=35, beta_scale=8.0)

    # Let's run simulation step-by-step
    s_sl = (0, 0)
    traj_sl = [s_sl]
    lat_sl = [0.0]

    active_c = None
    c_type = None

    for step in range(1, 28):
        if step < t_b:
            active_c = None
            c_type = None
        elif t_b <= step < t_s:
            active_c = b_chord
            c_type = "braess"
        else:
            active_c = s_chord
            c_type = "shortcut"

        if env_sl.active_chord != active_c:
            env_sl.active_chord = active_c
            env_sl.chord_type = c_type
            if active_c is not None:
                u_c, v_c, _, _ = active_c
                f_aug, _, z_star = env_sl.grid.solve_augmented_equilibrium(u_c, v_c, c=0.0, d=0.0)
                env_sl.current_edge_flows = f_aug[:-1]
                env_sl.current_chord_flow = z_star
            else:
                env_sl.current_edge_flows = env_sl.f_base.copy()
                env_sl.current_chord_flow = 0.0

        # Compute 4-directional Bellman targets for SL-PS
        Phi_spec = sl.compute_spectral_congestion_field()
        beta = sl.beta_scale * (1.0 + sl.slem)

        # Value iteration on 4-directional grid
        V = { (i, j): float((m - i) + (n - j)) * 1.0 for i in range(m + 1) for j in range(n + 1) }
        V[(m, n)] = 0.0
        for _ in range(50):
            for i in range(m + 1):
                for j in range(n + 1):
                    pos = (i, j)
                    if pos == (m, n):
                        V[pos] = 0.0
                        continue
                    v_acts = get_valid_actions_4dir(pos, active_c)
                    q_vals = []
                    for a in v_acts:
                        nxt, cost = transition_4dir(pos, a, env_sl, active_c, c_type)
                        pen = float(beta * Phi_spec[env_sl.grid.node_to_idx[nxt]]) if nxt in env_sl.grid.node_to_idx else 0.0
                        # Small progress heuristic to prevent cycles: small cost for backward steps
                        back_pen = 0.05 if a in [ACTION_UP, ACTION_LEFT] else 0.0
                        q_vals.append(cost + back_pen + 0.95 * V[nxt] + pen)
                    V[pos] = min(q_vals)

        # Select action at s_sl
        if s_sl != (m, n):
            v_acts = get_valid_actions_4dir(s_sl, active_c)
            cand = []
            for a in v_acts:
                nxt, cost = transition_4dir(s_sl, a, env_sl, active_c, c_type)
                pen = float(beta * Phi_spec[env_sl.grid.node_to_idx[nxt]]) if nxt in env_sl.grid.node_to_idx else 0.0
                back_pen = 0.05 if a in [ACTION_UP, ACTION_LEFT] else 0.0
                q_val = cost + back_pen + 0.95 * V[nxt] + pen
                cand.append((q_val, a, nxt, cost))
            cand.sort(key=lambda x: x[0])
            best_a = cand[0][1]
            nxt = cand[0][2]
            cost = cand[0][3]
            s_sl = nxt
            traj_sl.append(s_sl)
            lat_sl.append(cost)

        if s_sl == (m, n) and step >= t_s:
            break

    print(f"\n4-Actions SL-PS Result on Seed {seed}:")
    print(f"  SL Path: {' -> '.join(str(p) for p in traj_sl)}")
    print(f"  SL Total Latency: {sum(lat_sl):.2f}s (vs DAG SL: {c_sl_dag[-1]:.2f}s, vs DAG Std: {c_std_dag[-1]:.2f}s)")
    if sum(lat_sl) < c_std_dag[-1]:
        print(f"  SUCCESS! SL-PS with 4 actions BEATS Standard-PS by {c_std_dag[-1] - sum(lat_sl):.2f}s!")
    else:
        print(f"  Still slower by {sum(lat_sl) - c_std_dag[-1]:.2f}s")

test_4action_environment(seed=55)
