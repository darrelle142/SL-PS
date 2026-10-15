"""
planning_agents.py
==================
Implementation of Dynamic Programming and Model-Based Reinforcement Learning Planners:
1. GreedyDijkstraAgent: Naive shortest-path planner (susceptible to Braess traps).
2. StandardPrioritizedSweepingAgent: Canonical Prioritized Sweeping (Peng & Williams, 1993).
3. GTAPrioritizedSweepingAgent: Graph Topology Augmentation for Prioritized Sweeping (GTA-PS, 2025).
4. SLPSAgent (PROPOSED): Spectral-Laplacian Prioritized Sweeping (SL-PS).
   Unifies Spielman (SAGT 2025), Lu & Miller (2026), and GTA-PS (2025) via a novel
   Spectral-Laplacian Bellman Operator with Green's potential diffusion and SLEM annealing.
"""

import numpy as np
import heapq
from typing import Dict, List, Tuple, Optional, Any
from grid_braess_env import CongestedGridEnv


class GreedyDijkstraAgent:
    """
    Greedy agent that computes paths using nominal edge latencies.
    Blindly takes active chords because their nominal latency is 0.
    """

    def __init__(self, env: CongestedGridEnv):
        self.env = env
        self.name = "Greedy-Dijkstra"

    def select_action(self, state: Tuple[int, int], active_chord: Optional[Tuple] = None) -> int:
        valid_actions = self.env.get_valid_actions(state)
        if CongestedGridEnv.ACTION_CHORD in valid_actions:
            return CongestedGridEnv.ACTION_CHORD

        best_action = valid_actions[0]
        min_cost = float("inf")
        i, j = state
        target = (self.env.m, self.env.n)

        for a in valid_actions:
            if a == CongestedGridEnv.ACTION_RIGHT:
                nxt = (i + 1, j)
            elif a == CongestedGridEnv.ACTION_DOWN:
                nxt = (i, j + 1)
            elif a == CongestedGridEnv.ACTION_UP:
                nxt = (i - 1, j)
            elif a == CongestedGridEnv.ACTION_LEFT:
                nxt = (i, j - 1)
            elif a == CongestedGridEnv.ACTION_CHORD and self.env.active_chord is not None:
                nxt = self.env.active_chord[1]
            else:
                nxt = state
            dist = abs(target[0] - nxt[0]) + abs(target[1] - nxt[1])
            if dist < min_cost:
                min_cost = dist
                best_action = a

        return best_action


class StandardPrioritizedSweepingAgent:
    """
    Canonical Prioritized Sweeping (Peng & Williams, 1993; Moore & Atkeson, 1993).
    Selects model backups purely according to standard Bellman residual magnitude.
    Fails to propagate non-local congestion shifts under limited planning budget K.
    """

    def __init__(self, env: CongestedGridEnv, backup_budget: int = 15, gamma: float = 0.95):
        self.env = env
        self.K = backup_budget
        self.gamma = gamma
        self.name = "Standard-PS"

        # Initialize Value function V(s) with Manhattan heuristic
        self.V = {
            (i, j): float((env.m - i) + (env.n - j)) * env.grid.b
            for i in range(env.m + 1)
            for j in range(env.n + 1)
        }
        self.V[(env.m, env.n)] = 0.0

    def _get_transition(self, state: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], float]:
        i, j = state
        if action == CongestedGridEnv.ACTION_RIGHT:
            next_state = (i + 1, j)
            e_idx = self.env._get_edge_index(state, next_state)
            cost = self.env.grid.a * self.env.current_edge_flows[e_idx] + self.env.grid.b
        elif action == CongestedGridEnv.ACTION_DOWN:
            next_state = (i, j + 1)
            e_idx = self.env._get_edge_index(state, next_state)
            cost = self.env.grid.a * self.env.current_edge_flows[e_idx] + self.env.grid.b
        elif action == CongestedGridEnv.ACTION_UP:
            next_state = (i - 1, j)
            try:
                e_idx = self.env._get_edge_index(next_state, state)
                cost = self.env.grid.a * self.env.current_edge_flows[e_idx] + self.env.grid.b
            except ValueError:
                cost = self.env.grid.b
            cost += 0.05
        elif action == CongestedGridEnv.ACTION_LEFT:
            next_state = (i, j - 1)
            try:
                e_idx = self.env._get_edge_index(next_state, state)
                cost = self.env.grid.a * self.env.current_edge_flows[e_idx] + self.env.grid.b
            except ValueError:
                cost = self.env.grid.b
            cost += 0.05
        elif action == CongestedGridEnv.ACTION_CHORD and self.env.active_chord is not None:
            next_state = self.env.active_chord[1]
            cost = 0.0  # nominal chord latency
        else:
            next_state = state
            cost = 10.0
        return next_state, cost

    def _compute_bellman_target(self, state: Tuple[int, int]) -> Tuple[float, int]:
        if state == (self.env.m, self.env.n):
            return 0.0, 0
        valid = self.env.get_valid_actions(state)
        min_q = float("inf")
        best_act = valid[0]
        for a in valid:
            nxt, cost = self._get_transition(state, a)
            q_val = cost + self.gamma * self.V[nxt]
            if q_val < min_q:
                min_q = q_val
                best_act = a
        return min_q, best_act

    def plan(self):
        """Run K budgeted prioritized backups."""
        # Priority queue stores (-priority, counter, state)
        heap = []
        counter = 0

        # Push states with non-zero Bellman residual
        for i in range(self.env.m + 1):
            for j in range(self.env.n + 1):
                s = (i, j)
                if s == (self.env.m, self.env.n):
                    continue
                target, _ = self._compute_bellman_target(s)
                res = abs(target - self.V[s])
                if res > 1e-4:
                    heapq.heappush(heap, (-res, counter, s))
                    counter += 1

        # Process top-K states
        for _ in range(self.K):
            if not heap:
                break
            _, _, s = heapq.heappop(heap)
            target, _ = self._compute_bellman_target(s)
            diff = abs(target - self.V[s])
            self.V[s] = target

            # Propagate to predecessors
            si, sj = s
            predecessors = []
            if si > 0:
                predecessors.append((si - 1, sj))
            if sj > 0:
                predecessors.append((si, sj - 1))
            if getattr(self.env, "allow_backward", False):
                if si < self.env.m:
                    predecessors.append((si + 1, sj))
                if sj < self.env.n:
                    predecessors.append((si, sj + 1))
            if self.env.active_chord is not None and s == self.env.active_chord[1]:
                predecessors.append(self.env.active_chord[0])

            for p in predecessors:
                p_target, _ = self._compute_bellman_target(p)
                p_res = abs(p_target - self.V[p])
                if p_res > 1e-4:
                    heapq.heappush(heap, (-p_res, counter, p))
                    counter += 1

    def select_action(self, state: Tuple[int, int], active_chord: Optional[Tuple] = None) -> int:
        self.plan()
        _, best_a = self._compute_bellman_target(state)
        return best_a

    def get_q_breakdown(self, state: Tuple[int, int]) -> Dict[int, Dict[str, float]]:
        """Returns detailed Q-value decomposition for all valid actions at state."""
        valid = self.env.get_valid_actions(state)
        breakdown = {}
        for a in valid:
            nxt, cost = self._get_transition(state, a)
            q_val = cost + self.gamma * self.V[nxt]
            breakdown[a] = {
                "cost": cost,
                "future_v": self.gamma * self.V[nxt],
                "spectral_penalty": 0.0,
                "total_q": q_val,
                "next_state": nxt
            }
        return breakdown

    def get_policy_map(self) -> Dict[Tuple[int, int], int]:
        """Returns optimal greedy action for all nodes on grid."""
        pol = {}
        for i in range(self.env.m + 1):
            for j in range(self.env.n + 1):
                s = (i, j)
                if s == (self.env.m, self.env.n):
                    pol[s] = -1
                else:
                    _, a = self._compute_bellman_target(s)
                    pol[s] = a
        return pol


class GTAPrioritizedSweepingAgent:
    """
    Graph Topology Augmentation for Prioritized Sweeping (GTA-PS, 2025).
    Augments the priority score with regularized directed-Laplacian potentials:
    tau = (1 - alpha)|f_out| + alpha|f_in|, annealed by SLEM |lambda_2|.
    """

    def __init__(
        self,
        env: CongestedGridEnv,
        backup_budget: int = 15,
        gamma: float = 0.95,
        eta: float = 0.1,
        mix_alpha: float = 0.3
    ):
        self.env = env
        self.K = backup_budget
        self.gamma = gamma
        self.eta = eta
        self.mix_alpha = mix_alpha
        self.name = "GTA-PS"

        self.V = {
            (i, j): float((env.m - i) + (env.n - j)) * env.grid.b
            for i in range(env.m + 1)
            for j in range(env.n + 1)
        }
        self.V[(env.m, env.n)] = 0.0

        self._build_topological_operators()

    def _build_topological_operators(self):
        grid = self.env.grid
        num_n = grid.num_nodes
        P = np.zeros((num_n, num_n))
        for u in range(num_n):
            ui, uj = grid.idx_to_node[u]
            succs = []
            if ui < grid.m:
                succs.append(grid.node_to_idx[(ui + 1, uj)])
            if uj < grid.n:
                succs.append(grid.node_to_idx[(ui, uj + 1)])
            if succs:
                for s in succs:
                    P[u, s] = 1.0 / len(succs)
            else:
                P[u, u] = 1.0

        self.P = P
        eigvals = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
        self.slem = float(eigvals[1]) if len(eigvals) > 1 else 0.5

        self.L_out = np.eye(num_n) - P
        self.G_out = np.linalg.inv(self.L_out + self.eta * np.eye(num_n))

        Din = np.diag(np.sum(P, axis=0))
        self.L_in = Din - P.T
        self.G_in = np.linalg.inv(self.L_in + self.eta * np.eye(num_n))

    def _get_transition(self, state: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], float]:
        i, j = state
        if action == CongestedGridEnv.ACTION_RIGHT:
            next_state = (i + 1, j)
            e_idx = self.env._get_edge_index(state, next_state)
            cost = self.env.grid.a * self.env.current_edge_flows[e_idx] + self.env.grid.b
        elif action == CongestedGridEnv.ACTION_DOWN:
            next_state = (i, j + 1)
            e_idx = self.env._get_edge_index(state, next_state)
            cost = self.env.grid.a * self.env.current_edge_flows[e_idx] + self.env.grid.b
        elif action == CongestedGridEnv.ACTION_UP:
            next_state = (i - 1, j)
            try:
                e_idx = self.env._get_edge_index(next_state, state)
                cost = self.env.grid.a * self.env.current_edge_flows[e_idx] + self.env.grid.b
            except ValueError:
                cost = self.env.grid.b
            cost += 0.05
        elif action == CongestedGridEnv.ACTION_LEFT:
            next_state = (i, j - 1)
            try:
                e_idx = self.env._get_edge_index(next_state, state)
                cost = self.env.grid.a * self.env.current_edge_flows[e_idx] + self.env.grid.b
            except ValueError:
                cost = self.env.grid.b
            cost += 0.05
        elif action == CongestedGridEnv.ACTION_CHORD and self.env.active_chord is not None:
            next_state = self.env.active_chord[1]
            cost = 0.0
        else:
            next_state = state
            cost = 10.0
        return next_state, cost

    def _compute_bellman_target(self, state: Tuple[int, int]) -> Tuple[float, int]:
        if state == (self.env.m, self.env.n):
            return 0.0, 0
        valid = self.env.get_valid_actions(state)
        min_q = float("inf")
        best_act = valid[0]
        for a in valid:
            nxt, cost = self._get_transition(state, a)
            q_val = cost + self.gamma * self.V[nxt]
            if q_val < min_q:
                min_q = q_val
                best_act = a
        return min_q, best_act

    def plan(self):
        # Compute Bellman residuals xi
        num_n = self.env.grid.num_nodes
        xi = np.zeros(num_n)
        for i in range(self.env.m + 1):
            for j in range(self.env.n + 1):
                s = (i, j)
                if s == (self.env.m, self.env.n):
                    continue
                target, _ = self._compute_bellman_target(s)
                u_idx = self.env.grid.node_to_idx[s]
                xi[u_idx] = abs(target - self.V[s])

        # Diffuse via forward and backward Laplacians
        f_out = self.G_out @ xi
        f_in = self.G_in @ xi
        tau = (1.0 - self.mix_alpha) * np.abs(f_out) + self.mix_alpha * np.abs(f_in)

        # SLEM modulation weight beta
        beta = 2.0 * (1.0 - self.slem) + 0.5

        # Heap priority: xi(s) + beta * tau(s)
        heap = []
        counter = 0
        for i in range(self.env.m + 1):
            for j in range(self.env.n + 1):
                s = (i, j)
                if s == (self.env.m, self.env.n):
                    continue
                u_idx = self.env.grid.node_to_idx[s]
                prio = xi[u_idx] + beta * tau[u_idx]
                if prio > 1e-4:
                    heapq.heappush(heap, (-prio, counter, s))
                    counter += 1

        for _ in range(self.K):
            if not heap:
                break
            _, _, s = heapq.heappop(heap)
            target, _ = self._compute_bellman_target(s)
            self.V[s] = target

    def select_action(self, state: Tuple[int, int], active_chord: Optional[Tuple] = None) -> int:
        self.plan()
        _, best_a = self._compute_bellman_target(state)
        return best_a


class SLPSAgent:
    """
    PROPOSED NOVEL ALGORITHM:
    Spectral-Laplacian Prioritized Sweeping (SL-PS).
    Unifies:
    1. Spectral Graph Theory (Spielman SAGT 2025):
       Laplacian pseudoinverse L^+, Dirichlet energy, and Green's operator.
    2. Grid Braess Analysis (Lu & Miller 2026):
       Exact Wardrop equilibrium flow divergence Delta f = f^*(G_t) - f^*(G_0)
       and electrical voltage drop V_uv = q(e_s - e_t)^T L^+ (e_u - e_v).
    3. Spectral-Laplacian Bellman Operator:
       T_SL V(s) = min_a [ ell_a + gamma V(s') + beta(sigma) * Phi_spec(s') ]
       where Phi_spec = G_out @ |B Delta ell| diffuses the systemic congestion
       wavefront globally to all upstream decision nodes.
    """

    def __init__(
        self,
        env: CongestedGridEnv,
        backup_budget: int = 15,
        gamma: float = 0.95,
        eta: float = 0.1,
        beta_scale: float = 5.0,
        use_spectral_bellman: bool = True,
        use_slem: bool = True,
        alpha: float = 0.5
    ):
        self.env = env
        self.K = backup_budget
        self.gamma = gamma
        self.eta = eta
        self.beta_scale = beta_scale
        self.use_spectral_bellman = use_spectral_bellman
        self.use_slem = use_slem
        self.alpha = alpha
        self.name = "SL-PS (Proposed)"

        # Value function initialized with base grid potential distance
        self.V = {
            (i, j): float((env.m - i) + (env.n - j)) * env.grid.b
            for i in range(env.m + 1)
            for j in range(env.n + 1)
        }
        self.V[(env.m, env.n)] = 0.0

        self._precompute_spectral_structures()

    def _precompute_spectral_structures(self):
        grid = self.env.grid
        num_n = grid.num_nodes

        # Base nominal transition matrix P
        P = np.zeros((num_n, num_n))
        for u in range(num_n):
            ui, uj = grid.idx_to_node[u]
            succs = []
            if ui < grid.m:
                succs.append(grid.node_to_idx[(ui + 1, uj)])
            if uj < grid.n:
                succs.append(grid.node_to_idx[(ui, uj + 1)])
            if succs:
                for s in succs:
                    P[u, s] = 1.0 / len(succs)
            else:
                P[u, u] = 1.0

        self.P = P
        eigvals = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
        self.slem = float(eigvals[1]) if len(eigvals) > 1 else 0.5

        # Out-Laplacian Green operator G_out = (I - P + eta * I)^-1
        self.L_out = np.eye(num_n) - P
        self.G_out = np.linalg.inv(self.L_out + self.eta * np.eye(num_n))

        # In-Laplacian Green operator G_in = (D_in - P^T + eta * I)^-1
        Din = np.diag(np.sum(P, axis=0))
        self.L_in = Din - P.T
        self.G_in = np.linalg.inv(self.L_in + self.eta * np.eye(num_n))

    def compute_spectral_congestion_field(self) -> np.ndarray:
        """
        Computes the global Spectral Congestion Divergence Potential Phi_spec:
        Phi_spec = G_out @ |B Delta ell|
        where Delta ell = a (f_aug - f_base) is the real-time flow shift vector.
        """
        grid = self.env.grid
        num_n = grid.num_nodes

        if self.env.active_chord is None:
            return np.zeros(num_n)

        # Vector of latency shifts on base grid edges: Delta ell = a * (f_curr - f_base)
        delta_f = self.env.current_edge_flows - self.env.f_base
        delta_ell = grid.a * delta_f

        # Node-level divergence via directed incidence matrix B
        node_div = np.abs(grid.B @ delta_ell)

        # Global diffusion via regularized Green's potential operator G_out
        Phi_spec = self.G_out @ node_div
        return Phi_spec

    def _get_transition(self, state: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], float]:
        i, j = state
        if action == CongestedGridEnv.ACTION_RIGHT:
            next_state = (i + 1, j)
            e_idx = self.env._get_edge_index(state, next_state)
            cost = self.env.grid.a * self.env.current_edge_flows[e_idx] + self.env.grid.b
        elif action == CongestedGridEnv.ACTION_DOWN:
            next_state = (i, j + 1)
            e_idx = self.env._get_edge_index(state, next_state)
            cost = self.env.grid.a * self.env.current_edge_flows[e_idx] + self.env.grid.b
        elif action == CongestedGridEnv.ACTION_UP:
            next_state = (i - 1, j)
            try:
                e_idx = self.env._get_edge_index(next_state, state)
                cost = self.env.grid.a * self.env.current_edge_flows[e_idx] + self.env.grid.b
            except ValueError:
                cost = self.env.grid.b
            cost += 0.05
        elif action == CongestedGridEnv.ACTION_LEFT:
            next_state = (i, j - 1)
            try:
                e_idx = self.env._get_edge_index(next_state, state)
                cost = self.env.grid.a * self.env.current_edge_flows[e_idx] + self.env.grid.b
            except ValueError:
                cost = self.env.grid.b
            cost += 0.05
        elif action == CongestedGridEnv.ACTION_CHORD and self.env.active_chord is not None:
            next_state = self.env.active_chord[1]
            cost = 0.0  # nominal chord cost
        else:
            next_state = state
            cost = 10.0
        return next_state, cost

    def _compute_spectral_bellman_target(
        self, state: Tuple[int, int], Phi_spec: np.ndarray, beta: float
    ) -> Tuple[float, int]:
        """
        Evaluates the Spectral-Laplacian Bellman Operator:
        T_SL V(s) = min_a [ cost(s, a) + gamma * V(s') + beta * Phi_spec(s') ]
        """
        if state == (self.env.m, self.env.n):
            return 0.0, 0
        valid = self.env.get_valid_actions(state)
        min_q = float("inf")
        best_act = valid[0]

        for a in valid:
            nxt, cost = self._get_transition(state, a)
            nxt_idx = self.env.grid.node_to_idx[nxt]
            spectral_penalty = (beta * Phi_spec[nxt_idx]) if self.use_spectral_bellman else 0.0

            q_val = cost + self.gamma * self.V[nxt] + spectral_penalty
            if q_val < min_q:
                min_q = q_val
                best_act = a

        return min_q, best_act

    def plan(self):
        """
        Executes budgeted Spectral-Laplacian Prioritized Sweeping.
        """
        # 1. Compute Spectral Congestion Potential field Phi_spec
        Phi_spec = self.compute_spectral_congestion_field()

        # 2. Modulate penalty weight by SLEM sigma = |lambda_2|
        beta = self.beta_scale * (1.0 + self.slem) if self.use_slem else self.beta_scale

        # 3. Compute topological residuals
        num_n = self.env.grid.num_nodes
        xi = np.zeros(num_n)
        for i in range(self.env.m + 1):
            for j in range(self.env.n + 1):
                s = (i, j)
                if s == (self.env.m, self.env.n):
                    continue
                target, _ = self._compute_spectral_bellman_target(s, Phi_spec, beta)
                u_idx = self.env.grid.node_to_idx[s]
                xi[u_idx] = abs(target - self.V[s])

        # Diffuse priority keys through bi-directional Laplacian Green operators
        f_out = self.G_out @ (xi + Phi_spec)
        f_in = self.G_in @ (xi + Phi_spec)
        tau = (1.0 - self.alpha) * f_out + self.alpha * f_in

        heap = []
        counter = 0
        for i in range(self.env.m + 1):
            for j in range(self.env.n + 1):
                s = (i, j)
                if s == (self.env.m, self.env.n):
                    continue
                u_idx = self.env.grid.node_to_idx[s]
                if xi[u_idx] > 1e-4:
                    priority = xi[u_idx] + beta * tau[u_idx]
                    heapq.heappush(heap, (-priority, counter, s))
                    counter += 1

        # 4. Perform K budgeted Spectral-Laplacian backups
        for _ in range(self.K):
            if not heap:
                break
            _, _, s = heapq.heappop(heap)
            target, _ = self._compute_spectral_bellman_target(s, Phi_spec, beta)
            self.V[s] = target

            # Propagate to upstream predecessors
            si, sj = s
            predecessors = []
            if si > 0:
                predecessors.append((si - 1, sj))
            if sj > 0:
                predecessors.append((si, sj - 1))
            if getattr(self.env, "allow_backward", False):
                if si < self.env.m:
                    predecessors.append((si + 1, sj))
                if sj < self.env.n:
                    predecessors.append((si, sj + 1))
            if self.env.active_chord is not None and s == self.env.active_chord[1]:
                predecessors.append(self.env.active_chord[0])

            for p in predecessors:
                p_target, _ = self._compute_spectral_bellman_target(p, Phi_spec, beta)
                p_idx = self.env.grid.node_to_idx[p]
                p_diff = abs(p_target - self.V[p])
                if p_diff > 1e-4:
                    p_prio = p_diff + beta * tau[p_idx]
                    heapq.heappush(heap, (-p_prio, counter, p))
                    counter += 1

    def select_action(self, state: Tuple[int, int], active_chord: Optional[Tuple] = None) -> int:
        self.plan()
        Phi_spec = self.compute_spectral_congestion_field()
        beta = self.beta_scale * (1.0 + self.slem) if self.use_slem else self.beta_scale
        _, best_a = self._compute_spectral_bellman_target(state, Phi_spec, beta)
        return best_a

    def get_q_breakdown(self, state: Tuple[int, int]) -> Dict[int, Dict[str, float]]:
        """Returns detailed Q-value decomposition including spectral risk for all valid actions."""
        valid = self.env.get_valid_actions(state)
        Phi_spec = self.compute_spectral_congestion_field()
        beta = self.beta_scale * (1.0 + self.slem) if self.use_slem else self.beta_scale
        breakdown = {}
        for a in valid:
            nxt, cost = self._get_transition(state, a)
            nxt_idx = self.env.grid.node_to_idx[nxt]
            spec_pen = float(beta * Phi_spec[nxt_idx]) if self.use_spectral_bellman else 0.0
            q_val = cost + self.gamma * self.V[nxt] + spec_pen
            breakdown[a] = {
                "cost": cost,
                "future_v": self.gamma * self.V[nxt],
                "spectral_penalty": spec_pen,
                "total_q": q_val,
                "next_state": nxt
            }
        return breakdown

    def get_policy_map(self) -> Dict[Tuple[int, int], int]:
        """Returns optimal greedy action under T_SL for all nodes on grid."""
        pol = {}
        Phi_spec = self.compute_spectral_congestion_field()
        beta = self.beta_scale * (1.0 + self.slem) if self.use_slem else self.beta_scale
        for i in range(self.env.m + 1):
            for j in range(self.env.n + 1):
                s = (i, j)
                if s == (self.env.m, self.env.n):
                    pol[s] = -1
                else:
                    _, a = self._compute_spectral_bellman_target(s, Phi_spec, beta)
                    pol[s] = a
        return pol
