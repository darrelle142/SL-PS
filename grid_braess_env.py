"""
grid_braess_env.py
==================
Simulation environment for single agent navigating a dynamic congested grid network.
At each step t, candidate chords can appear with probability p_event.
Background traffic equilibrates according to Wardrop equilibrium (Beckmann potential).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from spectral_braess_core import GridNetwork


class CongestedGridEnv:
    """
    Gym-like Environment for Stochastic Congested Grid Navigation.
    """

    ACTION_RIGHT = 0  # (i, j) -> (i + 1, j) [Down in row]
    ACTION_DOWN = 1   # (i, j) -> (i, j + 1) [Right in col]
    ACTION_CHORD = 2  # u -> v if agent is at u and chord is active
    ACTION_UP = 3     # (i, j) -> (i - 1, j) [Up in row]
    ACTION_LEFT = 4   # (i, j) -> (i, j - 1) [Left in col]

    def __init__(
        self,
        m: int = 3,
        n: int = 6,
        a: float = 1.0,
        b: float = 1.0,
        q: float = 1.0,
        p_chord: float = 0.4,
        noise_std: float = 0.05,
        max_steps: int = 50,
        seed: Optional[int] = 42,
        s_bg: Tuple[int, int] = (0, 0),
        t_bg: Optional[Tuple[int, int]] = None,
        allow_backward: bool = False
    ):
        self.m = m
        self.n = n
        self.s_bg = s_bg
        self.t_bg = t_bg if t_bg is not None else (m, n)
        self.allow_backward = allow_backward
        self.grid = GridNetwork(m=m, n=n, a=a, b=b, q=q, s_bg=self.s_bg, t_bg=self.t_bg)
        self.p_chord = p_chord
        self.noise_std = noise_std
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # Precompute candidate chords
        self.all_chords = self.grid.find_all_chords()
        self.braess_chords = self.all_chords["braess_capable"]
        self.shortcut_chords = self.all_chords["shortcut"]

        # If Braess chords exist, set grid.b so that Braess condition holds for testing
        if len(self.braess_chords) > 0:
            u_c, v_c, V_uv, k = self.braess_chords[0]
            eps = 0.01
            self.grid.b = max(b, float((-V_uv + eps) / k))

        # Solve base equilibrium
        self.f_base, self.cost_base = self.grid.solve_base_equilibrium()
        self.current_edge_flows = self.f_base.copy()

        # State tracking
        self.agent_pos = (0, 0)
        self.step_count = 0
        self.active_chord: Optional[Tuple[Tuple[int, int], Tuple[int, int], float, int]] = None
        self.chord_type: Optional[str] = None  # 'braess' or 'shortcut'

    def reset(self, seed: Optional[int] = None) -> Tuple[Tuple[int, int], Dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agent_pos = (0, 0)
        self.step_count = 0
        self.active_chord = None
        self.chord_type = None
        self.current_edge_flows = self.f_base.copy()

        # Initial check for stochastic chord
        self._check_and_spawn_chord()

        return self.agent_pos, self._get_info()

    def _check_and_spawn_chord(self):
        """Randomly spawns or removes a chord with probability p_chord."""
        if self.active_chord is None:
            if self.rng.random() < self.p_chord:
                # Decide whether to spawn Braess chord or Shortcut chord
                spawn_braess = (self.rng.random() < 0.6) and (len(self.braess_chords) > 0)
                if spawn_braess:
                    chord_idx = self.rng.integers(0, len(self.braess_chords))
                    self.active_chord = self.braess_chords[chord_idx]
                    self.chord_type = "braess"
                elif len(self.shortcut_chords) > 0:
                    chord_idx = self.rng.integers(0, len(self.shortcut_chords))
                    self.active_chord = self.shortcut_chords[chord_idx]
                    self.chord_type = "shortcut"

                if self.active_chord is not None:
                    u_c, v_c, _, _ = self.active_chord
                    # Re-equilibrate background traffic
                    f_aug, _, z_star = self.grid.solve_augmented_equilibrium(u_c, v_c, c=0.0, d=0.0)
                    self.current_edge_flows = f_aug[:-1]
                    self.current_chord_flow = z_star
        else:
            # Chance of chord disappearing
            if self.rng.random() < 0.2:
                self.active_chord = None
                self.chord_type = None
                self.current_edge_flows = self.f_base.copy()

    def get_valid_actions(self, pos: Optional[Tuple[int, int]] = None) -> List[int]:
        if pos is None:
            pos = self.agent_pos
        i, j = pos
        valid = []
        if i < self.m:
            valid.append(self.ACTION_RIGHT)
        if j < self.n:
            valid.append(self.ACTION_DOWN)
        if getattr(self, "allow_backward", False):
            if i > 0:
                valid.append(self.ACTION_UP)
            if j > 0:
                valid.append(self.ACTION_LEFT)
        if self.active_chord is not None and self.active_chord[0] == pos:
            valid.append(self.ACTION_CHORD)
        return valid

    def _get_edge_index(self, u_coord: Tuple[int, int], v_coord: Tuple[int, int]) -> int:
        u_idx = self.grid.node_to_idx[u_coord]
        v_idx = self.grid.node_to_idx[v_coord]
        for e_idx, (u, v) in enumerate(self.grid.edges):
            if u == u_idx and v == v_idx:
                return e_idx
        raise ValueError(f"Edge {u_coord} -> {v_coord} not found in base grid")

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """
        Executes action.
        Returns: next_state, latency (cost), done, info.
        """
        self.step_count += 1
        valid_actions = self.get_valid_actions()
        if action not in valid_actions:
            # Invalid action penalty
            return self.agent_pos, 100.0, False, {"error": "Invalid action"}

        i, j = self.agent_pos
        latency = 0.0

        if action == self.ACTION_RIGHT:
            next_pos = (i + 1, j)
            e_idx = self._get_edge_index((i, j), next_pos)
            flow = self.current_edge_flows[e_idx]
            latency = self.grid.a * flow + self.grid.b

        elif action == self.ACTION_DOWN:
            next_pos = (i, j + 1)
            e_idx = self._get_edge_index((i, j), next_pos)
            flow = self.current_edge_flows[e_idx]
            latency = self.grid.a * flow + self.grid.b

        elif action == self.ACTION_UP:
            next_pos = (i - 1, j)
            # Edge in reverse: base edge is (next_pos -> (i, j))
            try:
                e_idx = self._get_edge_index(next_pos, (i, j))
                flow = self.current_edge_flows[e_idx]
                latency = self.grid.a * flow + self.grid.b
            except ValueError:
                latency = self.grid.b

        elif action == self.ACTION_LEFT:
            next_pos = (i, j - 1)
            # Edge in reverse: base edge is (next_pos -> (i, j))
            try:
                e_idx = self._get_edge_index(next_pos, (i, j))
                flow = self.current_edge_flows[e_idx]
                latency = self.grid.a * flow + self.grid.b
            except ValueError:
                latency = self.grid.b

        elif action == self.ACTION_CHORD:
            u_c, v_c, _, k = self.active_chord
            next_pos = v_c
            if self.chord_type == "braess":
                # Congestion bottleneck delay: base level travel time + non-linear traffic queue
                z = getattr(self, "current_chord_flow", 0.1)
                latency = float(k) * self.grid.b + 1.5 + 15.0 * z
            else:
                # Benign shortcut: expressway overpass
                latency = 0.5

        # Add stochastic noise
        noise = self.rng.normal(0, self.noise_std)
        actual_latency = max(0.01, latency + noise)

        self.agent_pos = next_pos
        done = (self.agent_pos == (self.m, self.n)) or (self.step_count >= self.max_steps)

        # Dynamic environment update for next step
        self._check_and_spawn_chord()

        info = self._get_info()
        info["actual_latency"] = actual_latency
        info["nominal_latency"] = latency

        return self.agent_pos, actual_latency, done, info

    def _get_info(self) -> Dict:
        return {
            "pos": self.agent_pos,
            "step": self.step_count,
            "active_chord": self.active_chord,
            "chord_type": self.chord_type
        }
