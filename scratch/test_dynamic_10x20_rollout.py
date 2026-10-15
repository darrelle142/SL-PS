import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from grid_braess_env import CongestedGridEnv
from planning_agents import StandardPrioritizedSweepingAgent, SLPSAgent

m, n = 10, 20
s_bg = (2, 5)
t_bg = (8, 15)
q = 3.0

print(f"Testing dynamic rollout on {m}x{n} grid...")
env = CongestedGridEnv(m=m, n=n, s_bg=s_bg, t_bg=t_bg, q=q, p_chord=0.0, noise_std=0.0, seed=42)

# Prevent automatic random chords, we will trigger them dynamically by schedule
env._check_and_spawn_chord = lambda: None

# Candidate Braess chord at (2, 8) -> (5, 6) or (2, 10) -> (7, 7)
braess_chord = ((2, 8), (5, 6), -0.1098, 1)

# Benign expressway shortcut downstream: e.g. (6, 14) -> (9, 18)
# Let's find a valid shortcut chord
shortcut_candidate = None
for sc in env.shortcut_chords:
    u, v, V_uv, k = sc
    if u[0] >= 6 and u[1] >= 12 and v[0] >= 8 and v[1] >= 16:
        shortcut_candidate = sc
        break
if shortcut_candidate is None:
    shortcut_candidate = env.shortcut_chords[0]
print("Selected Braess chord:", braess_chord)
print("Selected Expressway shortcut:", shortcut_candidate)

std_agent = StandardPrioritizedSweepingAgent(env, backup_budget=35)
sl_agent = SLPSAgent(env, backup_budget=35, beta_scale=8.0)

# We will run a step-by-step synchronized rollout for both agents
# Event schedule:
# t in [0..6]: Base grid, no chord
# t = 7: Braess chord opens! Traffic re-equilibrates.
# t = 14: Braess chord closes / Expressway opens!

def run_synchronized_dynamic_simulation():
    # We will simulate step by step
    # To keep the environment consistent for both agents, each agent runs in its own clone env
    env_std = CongestedGridEnv(m=m, n=n, s_bg=s_bg, t_bg=t_bg, q=q, p_chord=0.0, noise_std=0.0, seed=42)
    env_sl = CongestedGridEnv(m=m, n=n, s_bg=s_bg, t_bg=t_bg, q=q, p_chord=0.0, noise_std=0.0, seed=42)
    env_std._check_and_spawn_chord = lambda: None
    env_sl._check_and_spawn_chord = lambda: None

    std = StandardPrioritizedSweepingAgent(env_std, backup_budget=35)
    sl = SLPSAgent(env_sl, backup_budget=35, beta_scale=8.0)

    # Trigger events at specific step counts
    # Step 0..6: No chord
    # Step 7: Braess chord appears at (2, 8) -> (5, 6)
    # Let's test the trajectory!

    s_std = (0, 0)
    s_sl = (0, 0)
    env_std.agent_pos = s_std
    env_sl.agent_pos = s_sl

    traj_std = [s_std]
    traj_sl = [s_sl]

    lat_history_std = [0.0]
    lat_history_sl = [0.0]

    max_steps = 35
    for t in range(max_steps):
        # Check dynamic event
        active_c = None
        c_type = None
        if 6 <= t < 15:
            active_c = braess_chord
            c_type = "braess"
        elif t >= 15:
            active_c = shortcut_candidate
            c_type = "shortcut"

        # Apply to envs
        for e, ag in [(env_std, std), (env_sl, sl)]:
            if active_c != e.active_chord:
                e.active_chord = active_c
                e.chord_type = c_type
                if active_c is not None:
                    u_c, v_c, _, _ = active_c
                    f_aug, _, z_star = e.grid.solve_augmented_equilibrium(u_c, v_c, c=0.0, d=0.0)
                    e.current_edge_flows = f_aug[:-1]
                    e.current_chord_flow = z_star
                else:
                    e.current_edge_flows = e.f_base.copy()
                    e.current_chord_flow = 0.0
                ag.plan()

        # Step Std
        if s_std != (m, n):
            a_std = std.select_action(s_std, env_std.active_chord)
            s_std, cost_std, _, _ = env_std.step(a_std)
            traj_std.append(s_std)
            lat_history_std.append(cost_std)

        # Step SL
        if s_sl != (m, n):
            a_sl = sl.select_action(s_sl, env_sl.active_chord)
            s_sl, cost_sl, _, _ = env_sl.step(a_sl)
            traj_sl.append(s_sl)
            lat_history_sl.append(cost_sl)

        if s_std == (m, n) and s_sl == (m, n):
            break

    print(f"Finished in {t+1} steps:")
    print(f"Standard-PS total latency: {sum(lat_history_std):.2f}, steps: {len(traj_std)-1}")
    print(f"  Path: {' -> '.join(str(p) for p in traj_std)}")
    print(f"SL-PS total latency:       {sum(lat_history_sl):.2f}, steps: {len(traj_sl)-1}")
    print(f"  Path: {' -> '.join(str(p) for p in traj_sl)}")

run_synchronized_dynamic_simulation()
