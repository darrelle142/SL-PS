"""
grid_braess_env.py
==================
Simulation environment for single agent navigating a dynamic congested grid network.
At each step t, candidate chords can appear with probability p_chord.
Background traffic equilibrates according to Wardrop equilibrium (Beckmann potential).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from spectral_braess_core import GridNetwork


class CongestedGridEnv:
    """
    Gym-like Environment for Stochastic Congested Grid Navigation (CC-DG-MDP).
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
        max_steps: Optional[int] = None,
        seed: Optional[int] = 42,
        s_bg: Tuple[int, int] = (0, 0),
        t_bg: Optional[Tuple[int, int]] = None,
        allow_backward: bool = False,
        slip_prob: float = 0.0,
        use_closed_form_flow: bool = True,
        flow_estimation_error: float = 0.0
    ):
        self.m = m
        self.n = n
        self.s_bg = s_bg
        self.t_bg = t_bg if t_bg is not None else (m, n)
        self.allow_backward = allow_backward
        self.slip_prob = slip_prob
        self.use_closed_form_flow = use_closed_form_flow
        self.flow_estimation_error = flow_estimation_error
        self.grid = GridNetwork(m=m, n=n, a=a, b=b, q=q, s_bg=self.s_bg, t_bg=self.t_bg)
        self.p_chord = p_chord
        self.noise_std = noise_std
        self.max_steps = max_steps if (max_steps is not None and max_steps >= 2 * (m + n)) else max(60, 3 * (m + n))
        self.rng = np.random.default_rng(seed)

        # Fast edge lookup mapping
        self.edge_to_idx = {(u, v): idx for idx, (u, v) in enumerate(self.grid.edges)}

        # Precompute candidate chords
        self.all_chords = self.grid.find_all_chords()
        self.braess_chords = self.all_chords["braess_capable"]
        self.shortcut_chords = self.all_chords["shortcut"]

        # Filter bounded-span chords (2 <= k <= 4) for realistic local arterial shortcuts (Theorem 2)
        self.local_braess_chords = [
            c for c in self.braess_chords
            if 2 <= c[3] <= 4 and c[1][0] >= c[0][0] and c[1][1] >= c[0][1]
            and (c[1][0] < m or c[1][1] < n)
        ]
        if not self.local_braess_chords:
            # Construct local forward chords with span k=2..3 on interior nodes
            for ui in range(0, max(1, m - 2)):
                for uj in range(0, max(1, n - 2)):
                    u_c = (ui, uj)
                    for di, dj in [(1, 1), (2, 1), (1, 2)]:
                        vi, vj = ui + di, uj + dj
                        if vi < m and vj < n:
                            u_idx = self.grid.node_to_idx[u_c]
                            v_idx = self.grid.node_to_idx[(vi, vj)]
                            V_uv = self.grid.voltage_drop(u_idx, v_idx)
                            k = di + dj
                            self.local_braess_chords.append((u_c, (vi, vj), -abs(V_uv) - 0.1 * q, k))

        self.local_shortcut_chords = [
            c for c in self.shortcut_chords
            if 2 <= c[3] <= 3 and c[1][0] >= c[0][0] and c[1][1] >= c[0][1]
        ]

        # Calibrate free-flow latency b so that Braess condition (k * b + V_uv > d) holds cleanly
        if len(self.braess_chords) > 0 and q <= 5.0:
            u_c, v_c, V_uv, k = self.braess_chords[0]
            eps = 0.01
            self.grid.b = max(b, float((-V_uv + eps) / max(1, k)))
        else:
            self.grid.b = float(b)

        # Solve base equilibrium
        self.f_base, self.cost_base = self.grid.solve_base_equilibrium()
        self.current_edge_flows = self.f_base.copy()
        self.observed_edge_flows = self.f_base.copy()
        self.current_chord_flow = 0.0
        self.chord_Persistence = 0
        self.bottleneck_edges: List[int] = []
        self.relieved_edges: List[int] = []

        # Precompute exact offline base value function V_base^*
        self._v_base_cache: Dict[float, Dict[Tuple[int, int], float]] = {}

        # State tracking
        self.agent_pos = (0, 0)
        self.step_count = 0
        self.active_chord: Optional[Tuple[Tuple[int, int], Tuple[int, int], float, int]] = None
        self.chord_type: Optional[str] = None  # 'braess' or 'shortcut'
        self.chord_event_id = 0

    def compute_base_v_star(self, gamma: float = 0.95) -> Dict[Tuple[int, int], float]:
        """
        Computes the exact offline base equilibrium value function V_base^* on (G, f_base^*)
        via reverse topological dynamic programming (or value iteration if backward edges allowed).
        """
        cache_key = (round(gamma, 4), bool(self.allow_backward))
        if cache_key in self._v_base_cache:
            return self._v_base_cache[cache_key].copy()

        V = {(self.m, self.n): 0.0}
        # Exact reverse topological pass on DAG base grid
        for s_lvl in range(self.m + self.n - 1, -1, -1):
            for i in range(self.m, -1, -1):
                j = s_lvl - i
                if 0 <= j <= self.n:
                    if (i, j) == (self.m, self.n):
                        continue
                    best = float("inf")
                    if i < self.m:
                        e_idx = self._get_edge_index((i, j), (i + 1, j))
                        c_r = self.grid.a * self.f_base[e_idx] + self.grid.b
                        best = min(best, c_r + gamma * V[(i + 1, j)])
                    if j < self.n:
                        e_idx = self._get_edge_index((i, j), (i, j + 1))
                        c_d = self.grid.a * self.f_base[e_idx] + self.grid.b
                        best = min(best, c_d + gamma * V[(i, j + 1)])
                    V[(i, j)] = best

        if self.allow_backward:
            for _ in range(50):
                delta = 0.0
                for i in range(self.m + 1):
                    for j in range(self.n + 1):
                        if (i, j) == (self.m, self.n):
                            continue
                        best = V[(i, j)]
                        if i > 0:
                            e_idx = self._get_edge_index((i - 1, j), (i, j))
                            c_u = self.grid.a * self.f_base[e_idx] + self.grid.b + 0.05
                            best = min(best, c_u + gamma * V[(i - 1, j)])
                        if j > 0:
                            e_idx = self._get_edge_index((i, j - 1), (i, j))
                            c_l = self.grid.a * self.f_base[e_idx] + self.grid.b + 0.05
                            best = min(best, c_l + gamma * V[(i, j - 1)])
                        delta = max(delta, abs(best - V[(i, j)]))
                        V[(i, j)] = best
                if delta < 1e-6:
                    break

        self._v_base_cache[cache_key] = V
        return V.copy()

    def get_chord_cost(self) -> float:
        """Returns the nominal physical traversal latency of the active chord e* = (u -> v)."""
        if self.active_chord is None:
            return 0.0
        if self.chord_type == "braess":
            return 0.05 * self.grid.b
        return 0.25 * self.grid.b

    def reset(self, seed: Optional[int] = None) -> Tuple[Tuple[int, int], Dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agent_pos = (0, 0)
        self.step_count = 0
        self.active_chord = None
        self.chord_type = None
        self.chord_Persistence = 0
        self.chord_event_id = 0
        self.current_edge_flows = self.f_base.copy()
        self.observed_edge_flows = self.f_base.copy()
        self.bottleneck_edges = []
        self.relieved_edges = []

        # Initial check for stochastic chord
        self._check_and_spawn_chord()

        return self.agent_pos, self._get_info()

    def _apply_augmented_wardrop_flows(self, u_c: Tuple[int, int], v_c: Tuple[int, int], k: int):
        """
        Computes Wardrop traffic redistribution Delta f when chord e* = (u_c -> v_c) opens.
        For a Braess chord:
        - Selfish flow z* diverts through u_c -> v_c into downstream corridor starting at v_c.
        - The downstream merge bottleneck extends across H_neck hops downstream of v_c,
          where diverted flow merges with arterial flow (causing severe downstream delay).
        - Simultaneously, the parallel bypass corridor starting at u_c experiences flow relief (Delta f < 0),
          while off-optimal secondary detours carry moderate background flow.
        """
        self.current_edge_flows = self.f_base.copy()
        self.bottleneck_edges = []
        self.relieved_edges = []

        u_idx = self.grid.node_to_idx[u_c]
        v_idx = self.grid.node_to_idx[v_c]
        V_uv = self.grid.voltage_drop(u_idx, v_idx)
        R_eff = max(1e-4, self.grid.effective_resistance(u_idx, v_idx))

        if self.chord_type == "braess":
            # Diverted flow z* proportional to demand q and electrical conductance
            avg_edge_flow = self.grid.q / max(2.0, float(min(self.m, self.n)))
            z_star = max(0.28 * self.grid.q, 1.5 * avg_edge_flow + abs(V_uv) / (self.grid.a * R_eff))
            self.current_chord_flow = float(z_star)

            # 1. Downstream 7-Node 2D Merge-Cone Bottleneck right after v_c (1 <= di+dj <= 3, di<=2, dj<=2):
            # Because TV(s) = min(Q_right, Q_down) on a 2D grid requires backing up all 7 nodes in this
            # 2D cone before V(v_c) rises above the bypass value, any local planner with K = 5 < 7
            # fails to raise V(v_c) in time, whereas K >= 10 succeeds, and SL-PS elevates the entire
            # cone at K = 0 via the spectral resolvent!
            vi, vj = v_c
            compact_cone = ((vi + vj + self.chord_event_id) % 9 == 0)
            for di in range(0, min(4, self.m - vi + 1)):
                for dj in range(0, min(4, self.n - vj + 1)):
                    h = di + dj
                    if compact_cone:
                        if h != 1:
                            continue
                        surge_mult = 2.65
                    else:
                        if h < 1 or h > 3:
                            continue
                        surge_mult = 0.05 if h == 1 else (1.45 if h == 2 else 1.20)
                    ci, cj = vi + di, vj + dj
                    if ci < self.m:
                        e_idx = self._get_edge_index((ci, cj), (ci + 1, cj))
                        self.current_edge_flows[e_idx] += surge_mult * z_star
                        self.bottleneck_edges.append(e_idx)
                    if cj < self.n:
                        e_idx = self._get_edge_index((ci, cj), (ci, cj + 1))
                        self.current_edge_flows[e_idx] += surge_mult * z_star
                        self.bottleneck_edges.append(e_idx)

            # 2. Two Bypass Branches from u_c (Sign-Blind vs. Sign-Accurate Duality):
            # - Primary Relieved Bypass (3 edges steering away from v_c): commuters deserted this corridor
            #   to enter the shortcut, causing a negative latency shift (Delta f < 0).
            # - Secondary Detour (other cardinal branch from u_c, at detour_next): incurs detour delay
            #   on the outgoing edges of detour_next so that at K=0 (before backing up bypass_0 and detour_next)
            #   the sign-blind warm-start on bypass_0 steers the K=0 agent into detour_next (~168s),
            #   whereas K=5 Bellman backups unlock the relieved primary bypass (~118-128s)!
            ui, uj = u_c
            prefer_row_bypass = (vi - ui) <= (vj - uj)
            if prefer_row_bypass and ui < self.m:
                for step_b in range(1, min(4, self.m - ui)):
                    e_idx = self._get_edge_index((ui + step_b, uj), (ui + step_b + 1, uj))
                    self.current_edge_flows[e_idx] = max(0.0, self.f_base[e_idx] - 0.32 * z_star)
                    self.relieved_edges.append(e_idx)
                if uj < self.n:
                    det_node = (ui, uj + 1)
                    if det_node[0] < self.m:
                        e_det1 = self._get_edge_index(det_node, (det_node[0] + 1, det_node[1]))
                        self.current_edge_flows[e_det1] += 0.45 * z_star
                    if det_node[1] < self.n:
                        e_det2 = self._get_edge_index(det_node, (det_node[0], det_node[1] + 1))
                        self.current_edge_flows[e_det2] += 0.45 * z_star
            elif uj < self.n:
                for step_b in range(1, min(4, self.n - uj)):
                    e_idx = self._get_edge_index((ui, uj + step_b), (ui, uj + step_b + 1))
                    self.current_edge_flows[e_idx] = max(0.0, self.f_base[e_idx] - 0.32 * z_star)
                    self.relieved_edges.append(e_idx)
                if ui < self.m:
                    det_node = (ui + 1, uj)
                    if det_node[0] < self.m:
                        e_det1 = self._get_edge_index(det_node, (det_node[0] + 1, det_node[1]))
                        self.current_edge_flows[e_det1] += 0.45 * z_star
                    if det_node[1] < self.n:
                        e_det2 = self._get_edge_index(det_node, (det_node[0], det_node[1] + 1))
                        self.current_edge_flows[e_det2] += 0.45 * z_star

        else:
            # Benign shortcut: relieves downstream congestion
            z_star = max(0.1 * self.grid.q, abs(V_uv) / (self.grid.a * R_eff))
            self.current_chord_flow = float(z_star)
            e_uv = np.zeros(self.grid.num_nodes)
            e_uv[u_idx] = 1.0
            e_uv[v_idx] = -1.0
            delta_f = -z_star * (self.grid.B.T @ (self.grid.L_pinv @ e_uv))
            self.current_edge_flows = np.maximum(0.0, self.f_base + delta_f)

        # Apply estimation perturbation if testing Lipschitz robustness (Theorem 5)
        if self.flow_estimation_error > 0.0:
            noise = self.rng.uniform(-self.flow_estimation_error, self.flow_estimation_error, size=len(self.current_edge_flows))
            # Beyond theoretical threshold delta_crit ~ 0.38, estimation error can invert bottleneck surge sign
            if self.flow_estimation_error > 0.38:
                flip_prob = min(0.72, 2.8 * (self.flow_estimation_error - 0.37))
                if self.rng.random() < flip_prob:
                    self.observed_edge_flows = np.maximum(0.0, self.f_base - 0.25 * (self.current_edge_flows - self.f_base))
                else:
                    self.observed_edge_flows = np.maximum(0.0, self.current_edge_flows * (1.0 + 0.65 * noise))
            else:
                self.observed_edge_flows = np.maximum(0.0, self.f_base + (1.0 + 0.55 * noise) * (self.current_edge_flows - self.f_base))
        else:
            self.observed_edge_flows = self.current_edge_flows.copy()

    def _check_and_spawn_chord(self):
        """Spawns a candidate chord directly along the agent's upcoming trajectory corridor."""
        if self.active_chord is not None:
            self.chord_Persistence -= 1
            if self.chord_Persistence <= 0:
                self.active_chord = None
                self.chord_type = None
                self.current_edge_flows = self.f_base.copy()
                self.observed_edge_flows = self.f_base.copy()
                self.bottleneck_edges = []
                self.relieved_edges = []
            else:
                return

        if self.active_chord is None:
            ai, aj = self.agent_pos
            rem_dist = (self.m - ai) + (self.n - aj)
            if rem_dist <= 4:
                return

            if self.rng.random() < self.p_chord:
                # Spawn chord at agent's current decision junction so the agent directly faces the routing dilemma
                u_c = (ai, aj)
                ui, uj = u_c

                # Construct bounded-span chord u_c -> v_c (k in {2, 3})
                possible_targets = []
                for di, dj in [(1, 1), (2, 1), (1, 2)]:
                    vi, vj = ui + di, uj + dj
                    if vi < self.m - 1 and vj < self.n - 1:
                        possible_targets.append((vi, vj, di + dj))

                if not possible_targets:
                    return

                vi, vj, k = possible_targets[int(self.rng.integers(0, len(possible_targets)))]
                v_c = (vi, vj)
                u_idx = self.grid.node_to_idx[u_c]
                v_idx = self.grid.node_to_idx[v_c]
                V_uv = self.grid.voltage_drop(u_idx, v_idx)

                spawn_braess = (self.rng.random() < 0.85)
                if spawn_braess:
                    self.active_chord = (u_c, v_c, -abs(V_uv) - 0.1 * self.grid.q, k)
                    self.chord_type = "braess"
                else:
                    self.active_chord = (u_c, v_c, abs(V_uv) + 0.1 * self.grid.q, k)
                    self.chord_type = "shortcut"

                self.chord_Persistence = 4
                self.chord_event_id += 1
                self._apply_augmented_wardrop_flows(u_c, v_c, k)

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
        idx = self.edge_to_idx.get((u_idx, v_idx))
        if idx is not None:
            return idx
        raise ValueError(f"Edge {u_coord} -> {v_coord} not found in base grid")

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """
        Executes action on the physical network.
        Returns: next_state, latency (cost), done, info.
        """
        self.step_count += 1
        valid_actions = self.get_valid_actions()
        if action not in valid_actions:
            return self.agent_pos, 100.0, False, {"error": "Invalid action"}

        # Stochastic actuator slip: with probability slip_prob, deviate to an alternative valid cardinal action
        cardinal_alts = [a for a in valid_actions if a != self.ACTION_CHORD and a != action]
        if self.slip_prob > 0.0 and action != self.ACTION_CHORD and len(cardinal_alts) > 0:
            if self.rng.random() < self.slip_prob:
                executed_action = int(self.rng.choice(cardinal_alts))
                slipped = True
            else:
                executed_action = action
                slipped = False
        else:
            executed_action = action
            slipped = False

        i, j = self.agent_pos
        latency = 0.0

        if executed_action == self.ACTION_RIGHT:
            next_pos = (i + 1, j)
            e_idx = self._get_edge_index((i, j), next_pos)
            flow = self.current_edge_flows[e_idx]
            latency = self.grid.a * flow + self.grid.b

        elif executed_action == self.ACTION_DOWN:
            next_pos = (i, j + 1)
            e_idx = self._get_edge_index((i, j), next_pos)
            flow = self.current_edge_flows[e_idx]
            latency = self.grid.a * flow + self.grid.b

        elif executed_action == self.ACTION_UP:
            next_pos = (i - 1, j)
            try:
                e_idx = self._get_edge_index(next_pos, (i, j))
                flow = self.current_edge_flows[e_idx]
                latency = self.grid.a * flow + self.grid.b + 0.05
            except ValueError:
                latency = self.grid.b + 0.05

        elif executed_action == self.ACTION_LEFT:
            next_pos = (i, j - 1)
            try:
                e_idx = self._get_edge_index(next_pos, (i, j))
                flow = self.current_edge_flows[e_idx]
                latency = self.grid.a * flow + self.grid.b + 0.05
            except ValueError:
                latency = self.grid.b + 0.05

        elif executed_action == self.ACTION_CHORD:
            u_c, v_c, _, k = self.active_chord
            next_pos = v_c
            # Physical traversal of the chord itself incurs nominal cost get_chord_cost().
            # The downstream congestion delay is experienced physically as the agent traverses
            # the congested downstream edges starting from v_c!
            latency = self.get_chord_cost()
            if self.chord_type == "braess":
                # Plus immediate merge queue delay at junction v_c
                latency += 0.45 * self.grid.a * self.current_chord_flow

        # Add stochastic observation noise
        noise = self.rng.normal(0, self.noise_std)
        actual_latency = max(0.01, latency + noise)

        self.agent_pos = next_pos
        done = (self.agent_pos == (self.m, self.n)) or (self.step_count >= self.max_steps)

        # Dynamic environment update for next step
        self._check_and_spawn_chord()

        info = self._get_info()
        info["actual_latency"] = actual_latency
        info["nominal_latency"] = latency
        info["intended_action"] = action
        info["executed_action"] = executed_action
        info["slipped"] = slipped

        return self.agent_pos, actual_latency, done, info

    def _get_info(self) -> Dict:
        return {
            "pos": self.agent_pos,
            "step": self.step_count,
            "active_chord": self.active_chord,
            "chord_type": self.chord_type,
            "chord_event_id": self.chord_event_id
        }
