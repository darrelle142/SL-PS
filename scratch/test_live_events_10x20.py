import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from grid_braess_env import CongestedGridEnv
from planning_agents import StandardPrioritizedSweepingAgent, SLPSAgent

m, n = 10, 20
s_bg = (3, 6)
t_bg = (7, 14)
q = 2.5

# Braess chord at (3, 9) -> (5, 8)
braess_chord = ((3, 9), (5, 8), -0.0541, 1)

# Benign expressway at (7, 13) -> (9, 17) or (6, 14) -> (9, 18)
expressway_chord = ((6, 14), (9, 18), 1.2, 5)

print(f"Setting up 10x20 test with Dynamic Events...")
print(f"Braess chord: {braess_chord[0]} -> {braess_chord[1]}")
print(f"Expressway:   {expressway_chord[0]} -> {expressway_chord[1]}")

def run_test():
    env_std = CongestedGridEnv(m=m, n=n, s_bg=s_bg, t_bg=t_bg, q=q, p_chord=0.0, noise_std=0.0, seed=42)
    env_sl = CongestedGridEnv(m=m, n=n, s_bg=s_bg, t_bg=t_bg, q=q, p_chord=0.0, noise_std=0.0, seed=42)

    # Disable random chords
    env_std._check_and_spawn_chord = lambda: None
    env_sl._check_and_spawn_chord = lambda: None

    # Custom natural tie-breaking to avoid degenerate border sliding
    def natural_bellman(agent, s, is_sl=False):
        valid = agent.env.get_valid_actions(s)
        best_v = float('inf')
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
                # Tie-break: keep trajectory close to the diagonal line j = 2 * i
                curr_nxt, _ = agent._get_transition(s, best_a) if best_a is not None else (s, 0)
                curr_dist = abs(curr_nxt[1] - 2 * curr_nxt[0])
                new_dist = abs(nxt[1] - 2 * nxt[0])
                if new_dist < curr_dist:
                    best_v = target
                    best_a = a
        return best_v, best_a

    std = StandardPrioritizedSweepingAgent(env_std, backup_budget=35)
    sl = SLPSAgent(env_sl, backup_budget=35, beta_scale=8.0)

    std._compute_bellman_target = lambda s: natural_bellman(std, s, is_sl=False)
    sl._compute_spectral_bellman_target = lambda s, phi, beta: natural_bellman(sl, s, is_sl=True)

    s_std = (0, 0)
    s_sl = (0, 0)
    env_std.agent_pos = s_std
    env_sl.agent_pos = s_sl

    traj_std = [s_std]
    traj_sl = [s_sl]
    lat_std = [0.0]
    lat_sl = [0.0]

    # Pre-plan
    std.plan()
    sl.plan()

    for step in range(35):
        # Dynamic event trigger:
        # Step 0..8: No chord (Base grid)
        # Step 9..16: Braess chord (3, 9) -> (5, 8) opens!
        # Step 17+: Expressway shortcut (6, 14) -> (9, 18) opens!
        if step < 9:
            active_c = None
            c_type = None
        elif 9 <= step < 17:
            active_c = braess_chord
            c_type = "braess"
        else:
            active_c = expressway_chord
            c_type = "shortcut"

        for env, agent in [(env_std, std), (env_sl, sl)]:
            if env.active_chord != active_c:
                env.active_chord = active_c
                env.chord_type = c_type
                if active_c is not None:
                    u_c, v_c, _, _ = active_c
                    f_aug, _, z_star = env.grid.solve_augmented_equilibrium(u_c, v_c, c=0.0, d=0.0)
                    env.current_edge_flows = f_aug[:-1]
                    env.current_chord_flow = z_star
                else:
                    env.current_edge_flows = env.f_base.copy()
                    env.current_chord_flow = 0.0
                agent.plan()

        # Step Std
        if s_std != (m, n):
            _, a_std = natural_bellman(std, s_std, is_sl=False)
            nxt_std, cost_std, _, _ = env_std.step(a_std)
            traj_std.append(nxt_std)
            lat_std.append(cost_std)
            s_std = nxt_std

        # Step SL
        if s_sl != (m, n):
            _, a_sl = natural_bellman(sl, s_sl, is_sl=True)
            nxt_sl, cost_sl, _, _ = env_sl.step(a_sl)
            traj_sl.append(nxt_sl)
            lat_sl.append(cost_sl)
            s_sl = nxt_sl

        print(f"Step {step+1:02d} | Event: {c_type or 'None':8s} | Std @ {str(s_std):8s} (lat={lat_std[-1]:.2f}) | SL @ {str(s_sl):8s} (lat={lat_sl[-1]:.2f})")

        if s_std == (m, n) and s_sl == (m, n):
            break

    print("\n--- Summary ---")
    print(f"Standard-PS total latency: {sum(lat_std):.2f}, steps: {len(traj_std)-1}")
    print(f"  Path: {' -> '.join(str(p) for p in traj_std)}")
    print(f"SL-PS total latency:       {sum(lat_sl):.2f}, steps: {len(traj_sl)-1}")
    print(f"  Path: {' -> '.join(str(p) for p in traj_sl)}")

run_test()
