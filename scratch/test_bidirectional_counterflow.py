import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from spectral_braess_core import GridNetwork

class BidirectionalGridNetwork(GridNetwork):
    def __init__(self, m=8, n=16, a=1.0, b=1.0, q=2.5, s_bg=(6, 12), t_bg=(2, 4)):
        self.m = m
        self.n = n
        self.a = a
        self.b = b
        self.q = q
        self.s_bg = s_bg
        self.t_bg = t_bg

        self.num_nodes = (m + 1) * (n + 1)
        self.node_to_idx = {(i, j): i * (n + 1) + j for i in range(m + 1) for j in range(n + 1)}
        self.idx_to_node = {idx: coord for coord, idx in self.node_to_idx.items()}

        self.s_idx = self.node_to_idx[self.s_bg]
        self.t_idx = self.node_to_idx[self.t_bg]

        # Bidirectional edges (North, South, East, West)
        self.edges = []
        for i in range(m + 1):
            for j in range(n + 1):
                u = self.node_to_idx[(i, j)]
                # Down
                if i < m:
                    self.edges.append((u, self.node_to_idx[(i + 1, j)]))
                # Up (Reverse)
                if i > 0:
                    self.edges.append((u, self.node_to_idx[(i - 1, j)]))
                # Right
                if j < n:
                    self.edges.append((u, self.node_to_idx[(i, j + 1)]))
                # Left (Reverse)
                if j > 0:
                    self.edges.append((u, self.node_to_idx[(i, j - 1)]))

        self.num_edges = len(self.edges)
        self._build_spectral_operators()

print("Testing Bidirectional Grid Network with Counter-Flow...")
# Agent will travel (0, 0) -> (8, 16) [Down-Right]
# Background traffic travels (6, 12) -> (2, 4) [Up-Left: HEAD-ON COUNTER-FLOW!]
bg_net = BidirectionalGridNetwork(m=8, n=16, s_bg=(6, 12), t_bg=(2, 4), q=2.5)
print(f"Nodes: {bg_net.num_nodes}, Bidirectional Edges: {bg_net.num_edges}")

f_base, c_base = bg_net.solve_base_equilibrium()
print(f"Counter-flow base equilibrium solved! Cost: {c_base:.3f}")

# Check edge flows: where does counter-flow go?
active_edges = [(bg_net.idx_to_node[u], bg_net.idx_to_node[v], f) 
                for (u, v), f in zip(bg_net.edges, f_base) if f > 0.05]
print(f"Total active counter-flow edges: {len(active_edges)}")
print("Sample active counter-flow edges (moving Up & Left):")
for u, v, f in active_edges[:10]:
    print(f"  {u} -> {v}: flow={f:.3f}")
