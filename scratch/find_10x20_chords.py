import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
from spectral_braess_core import GridNetwork

m, n = 10, 20
s_bg = (2, 5)
t_bg = (8, 15)
q = 3.0

print(f"Initializing Grid {m}x{n} with BG flow {s_bg} -> {t_bg}, q={q}...")
t0 = time.time()
grid = GridNetwork(m=m, n=n, s_bg=s_bg, t_bg=t_bg, q=q)
chords = grid.find_all_chords()
print(f"Found {len(chords['braess_capable'])} total Braess chords in {time.time()-t0:.2f}s.")

internal_braess = []
for c in chords['braess_capable']:
    u, v, V_uv, k = c
    if (s_bg[0] <= u[0] <= t_bg[0] and s_bg[1] <= u[1] <= t_bg[1] and
        s_bg[0] <= v[0] <= t_bg[0] and s_bg[1] <= v[1] <= t_bg[1]):
        internal_braess.append(c)

print(f"Internal Braess chords: {len(internal_braess)}")

f_base, c_base = grid.solve_base_equilibrium()
print(f"Base equilibrium cost: {c_base:.4f}")

top_chords = []
# Test a sample of internal chords
for idx, c in enumerate(internal_braess[:50]):
    u, v, V_uv, k = c
    f_aug, c_aug, z_star = grid.solve_augmented_equilibrium(u, v, c=0.0, d=0.0)
    delta_C = c_aug - c_base
    if z_star > 0.1 and delta_C > 0:
        top_chords.append((c, z_star, delta_C))

top_chords.sort(key=lambda x: x[1], reverse=True)
print(f"\nTop {len(top_chords)} Paradox Chords with z* > 0.1 and delta_C > 0:")
for c, z, dC in top_chords[:10]:
    print(f"  Chord {c[0]} -> {c[1]}: V_uv={c[2]:.4f}, k={c[3]}, z*={z:.4f}, delta_C={dC:+.4f}")
