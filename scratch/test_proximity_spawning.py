import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import time
import numpy as np
from spectral_braess_core import GridNetwork
from grid_braess_env import CongestedGridEnv
from planning_agents import StandardPrioritizedSweepingAgent, SLPSAgent

def test_proximity(seed=42):
    rng = random.Random(seed)
    m, n = 8, 16
    s_bg = (1, 2)
    t_bg = (6, 12)
    q = 2.8

    grid = GridNetwork(m=m, n=n, s_bg=s_bg, t_bg=t_bg, q=q)
    f_base, c_base = grid.solve_base_equilibrium()
    all_chords = grid.find_all_chords()

    braess_list = all_chords["braess_capable"]
    shortcut_list = all_chords["shortcut"]

    print(f"Total Braess chords: {len(braess_list)}, Shortcuts: {len(shortcut_list)}")

    # Agent is at pos (e.g. (2, 4))
    agent_pos = (2, 4)
    delta_i, delta_j = 1, 2

    # Find chords with u in [agent_pos, agent_pos + delta]
    near_braess = []
    for c in braess_list:
        u, v, V_uv, k = c
        if (agent_pos[0] <= u[0] <= agent_pos[0] + delta_i and
            agent_pos[1] <= u[1] <= agent_pos[1] + delta_j):
            near_braess.append(c)

    near_shortcuts = []
    for c in shortcut_list:
        u, v, V_uv, k = c
        if (agent_pos[0] <= u[0] <= agent_pos[0] + delta_i and
            agent_pos[1] <= u[1] <= agent_pos[1] + delta_j):
            near_shortcuts.append(c)

    print(f"Chords within Delta={(delta_i, delta_j)} of agent at {agent_pos}:")
    print(f"  Near Braess (V < 0): {len(near_braess)}")
    for c in near_braess[:5]:
        print(f"    {c[0]} -> {c[1]}: V_uv={c[2]:.4f}, k={c[3]}")
    print(f"  Near Shortcuts (V > 0): {len(near_shortcuts)}")
    for c in near_shortcuts[:5]:
        print(f"    {c[0]} -> {c[1]}: V_uv={c[2]:.4f}, k={c[3]}")

test_proximity(42)
