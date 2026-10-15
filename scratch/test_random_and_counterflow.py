import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import time
import numpy as np
from spectral_braess_core import GridNetwork
from grid_braess_env import CongestedGridEnv
from planning_agents import StandardPrioritizedSweepingAgent, SLPSAgent

def test_random_sampling(seed=None):
    if seed is None:
        seed = int(time.time() * 1000) % 100000
    rng = random.Random(seed)
    print(f"=== Testing Random Chord Sampling with Seed {seed} ===")

    m, n = 8, 16
    # Randomize background flow inside grid
    # E.g., s_bg and t_bg randomly picked
    s_bg = (rng.randint(1, 3), rng.randint(2, 5))
    t_bg = (rng.randint(m - 3, m - 1), rng.randint(n - 5, n - 2))
    q = rng.uniform(2.0, 3.5)

    print(f"Random BG flow: {s_bg} -> {t_bg}, q={q:.2f}")
    grid = GridNetwork(m=m, n=n, s_bg=s_bg, t_bg=t_bg, q=q)
    f_base, c_base = grid.solve_base_equilibrium()

    all_chords = grid.find_all_chords()
    braess_list = all_chords["braess_capable"]
    shortcut_list = all_chords["shortcut"]

    print(f"Available Braess chords (V < 0): {len(braess_list)}")
    print(f"Available Shortcuts (V > 0): {len(shortcut_list)}")

    # Filter internal Braess chords with true paradox (z* > 0.1, delta_C > 0)
    valid_braess = []
    # Sample up to 40 candidate Braess chords to check
    sample_b = rng.sample(braess_list, min(40, len(braess_list)))
    for c in sample_b:
        u, v, V_uv, k = c
        if (s_bg[0] <= u[0] <= t_bg[0] and s_bg[1] <= u[1] <= t_bg[1] and
            s_bg[0] <= v[0] <= t_bg[0] and s_bg[1] <= v[1] <= t_bg[1]):
            f_aug, c_aug, z = grid.solve_augmented_equilibrium(u, v, c=0.0, d=0.0)
            if z > 0.1 and c_aug - c_base > 0:
                valid_braess.append((c, z, c_aug - c_base))

    # Pick 1 random guaranteed Braess chord
    chosen_braess = None
    if valid_braess:
        chosen_braess = rng.choice(valid_braess)
        print(f"Guaranteed V < 0 Chord: {chosen_braess[0][0]} -> {chosen_braess[0][1]}, V_uv={chosen_braess[0][2]:.4f}, z*={chosen_braess[1]:.3f}, delta_C={chosen_braess[2]:+.4f}")
    else:
        # Fallback to the strongest in braess_list
        chosen_braess = (braess_list[0], 0.3, 0.05)
        print(f"Fallback V < 0 Chord: {chosen_braess[0][0]} -> {chosen_braess[0][1]}")

    # Pick 1 random guaranteed Shortcut chord (V > 0) downstream
    # We want a shortcut where u is after the Braess chord or downstream
    downstream_shortcuts = [sc for sc in shortcut_list if sc[0][0] >= chosen_braess[0][0][0] and sc[2] > 0]
    chosen_shortcut = rng.choice(downstream_shortcuts if downstream_shortcuts else shortcut_list)
    print(f"Guaranteed V > 0 Chord: {chosen_shortcut[0]} -> {chosen_shortcut[1]}, V_uv=+{chosen_shortcut[2]:.4f}, k={chosen_shortcut[3]}")

    return chosen_braess, chosen_shortcut

test_random_sampling(42)
test_random_sampling(1234)
