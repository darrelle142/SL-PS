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

print(f"Testing Grid 10x20 with BG {s_bg} -> {t_bg}, q={q}...")
env = CongestedGridEnv(m=m, n=n, s_bg=s_bg, t_bg=t_bg, q=q, p_chord=0.0, noise_std=0.0)
f_base, c_base = env.grid.solve_base_equilibrium()

# Find top Braess chords in this setup
chords = env.grid.find_all_chords()
top_chords = []
for c in chords['braess_capable']:
    u, v, V_uv, k = c
    if (s_bg[0] <= u[0] <= t_bg[0] and s_bg[1] <= u[1] <= t_bg[1] and
        s_bg[0] <= v[0] <= t_bg[0] and s_bg[1] <= v[1] <= t_bg[1]):
        f_aug, c_aug, z = env.grid.solve_augmented_equilibrium(u, v, c=0.0, d=0.0)
        dC = c_aug - c_base
        if z > 0.2 and dC > 0:
            top_chords.append((c, z, dC))

top_chords.sort(key=lambda x: x[1], reverse=True)
print(f"Top internal Braess chords for BG {s_bg} -> {t_bg}:")
for c, z, dC in top_chords[:8]:
    print(f"  {c[0]} -> {c[1]}: V_uv={c[2]:.4f}, k={c[3]}, z*={z:.4f}, delta_C={dC:+.4f}")
