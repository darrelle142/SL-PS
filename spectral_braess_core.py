"""
spectral_braess_core.py
========================
Core mathematical foundation for Spectral Graph Theory and Braess' Paradox on Grid Networks.
Based on:
- Spielman (2025): Spectral and Algebraic Graph Theory (SAGT)
- Lu & Miller (2026): Braess' Paradox in Uniform Affine Grid Networks (arXiv:2608.28938)
"""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize, LinearConstraint, Bounds
from typing import Dict, List, Tuple, Optional


class GridNetwork:
    """
    Directed rectangular grid network G_{m, n} with uniform affine latencies.
    Nodes are (i, j) for 0 <= i <= m, 0 <= j <= n.
    Source s = (0, 0), Sink t = (m, n).
    """

    def __init__(
        self,
        m: int,
        n: int,
        a: float = 1.0,
        b: float = 1.0,
        q: float = 1.0,
        s_bg: Tuple[int, int] = (0, 0),
        t_bg: Optional[Tuple[int, int]] = None
    ):
        self.m = m
        self.n = n
        self.a = float(a)
        self.b = float(b)
        self.q = float(q)
        self.num_nodes = (m + 1) * (n + 1)
        self.s_bg = s_bg
        self.t_bg = t_bg if t_bg is not None else (m, n)

        # Node index mapping
        self.node_to_idx = {
            (i, j): i * (n + 1) + j
            for i in range(m + 1)
            for j in range(n + 1)
        }
        self.idx_to_node = {idx: coord for coord, idx in self.node_to_idx.items()}

        self.s_idx = self.node_to_idx[self.s_bg]
        self.t_idx = self.node_to_idx[self.t_bg]

        # Base grid directed edges
        # Horizontal: (i, j) -> (i + 1, j)
        # Vertical: (i, j) -> (i, j + 1)
        self.edges: List[Tuple[int, int]] = []
        for i in range(m):
            for j in range(n + 1):
                self.edges.append((self.node_to_idx[(i, j)], self.node_to_idx[(i + 1, j)]))
        for i in range(m + 1):
            for j in range(n):
                self.edges.append((self.node_to_idx[(i, j)], self.node_to_idx[(i, j + 1)]))

        self.num_edges = len(self.edges)
        self._build_spectral_operators()

    def _build_spectral_operators(self):
        """Build directed incidence matrix B and Laplacian L = B W B^T."""
        B = np.zeros((self.num_nodes, self.num_edges), dtype=np.float64)
        for e_idx, (u, v) in enumerate(self.edges):
            B[u, e_idx] = 1.0
            B[v, e_idx] = -1.0
        self.B = B

        # Conductance matrix W = (1/a) * I
        # Laplacian L = (1/a) * B @ B.T
        self.L = (1.0 / self.a) * (self.B @ self.B.T)

        # Moore-Penrose pseudoinverse L^+
        self.L_pinv = np.linalg.pinv(self.L)

        # Base injection vector p = q (e_s - e_t)
        self.p_base = np.zeros(self.num_nodes, dtype=np.float64)
        self.p_base[self.s_idx] = self.q
        self.p_base[self.t_idx] = -self.q

        # Base electrical potential phi = L^+ p_base
        self.phi_base = self.L_pinv @ self.p_base

        # Base effective resistance between s and t
        e_st = np.zeros(self.num_nodes, dtype=np.float64)
        e_st[self.s_idx] = 1.0
        e_st[self.t_idx] = -1.0
        self.R_st = float(e_st @ self.L_pinv @ e_st)

    def level(self, u_idx: int) -> int:
        """Level lambda(i, j) = i + j."""
        i, j = self.idx_to_node[u_idx]
        return i + j

    def level_gain(self, u_idx: int, v_idx: int) -> int:
        """Level gain k_{u,v} = lambda(v) - lambda(u)."""
        return self.level(v_idx) - self.level(u_idx)

    def voltage_drop(self, u_idx: int, v_idx: int) -> float:
        """
        Electrical voltage drop across (u, v):
        V_{uv} = phi_u - phi_v = q (e_s - e_t)^T L^+ (e_u - e_v).
        """
        return float(self.phi_base[u_idx] - self.phi_base[v_idx])

    def effective_resistance(self, u_idx: int, v_idx: int) -> float:
        """
        Effective resistance between u and v:
        R_{uv} = (e_u - e_v)^T L^+ (e_u - e_v).
        """
        e_uv = np.zeros(self.num_nodes, dtype=np.float64)
        e_uv[u_idx] = 1.0
        e_uv[v_idx] = -1.0
        return float(e_uv @ self.L_pinv @ e_uv)

    def find_all_chords(self) -> Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int], float, int]]]:
        """
        Classifies all non-adjacent chords u -> v into:
        - 'braess_capable': V_{uv} < 0 and k > 0 (can cause Braess paradox under Theorem 2.15)
        - 'shortcut': V_{uv} > 0 and k > 0 (strictly beneficial forward shortcuts)
        - 'backward': k <= 0
        """
        results = {
            "braess_capable": [],
            "shortcut": [],
            "backward": []
        }

        for u in range(self.num_nodes):
            ui, uj = self.idx_to_node[u]
            for v in range(self.num_nodes):
                if u == v:
                    continue
                vi, vj = self.idx_to_node[v]
                # Skip if already adjacent in undirected grid
                if abs(ui - vi) + abs(uj - vj) <= 1:
                    continue

                k = self.level_gain(u, v)
                V_uv = self.voltage_drop(u, v)
                entry = ((ui, uj), (vi, vj), V_uv, k)

                if k > 0 and V_uv < 0:
                    results["braess_capable"].append(entry)
                elif k > 0 and V_uv > 0:
                    results["shortcut"].append(entry)
                else:
                    results["backward"].append(entry)

        return results

    def solve_base_equilibrium(self) -> Tuple[np.ndarray, float]:
        """
        Solves User Equilibrium (Wardrop) via Beckmann potential minimization on base grid:
        min sum_{e} (0.5 * a * f_e^2 + b * f_e)
        subject to B[:-1] f = p[:-1], f >= 0.
        Returns: (f_base, total_cost).
        """
        if hasattr(self, "_base_cache") and self._base_cache is not None:
            return self._base_cache

        def obj(f):
            return np.sum(0.5 * self.a * f**2 + self.b * f)

        def grad(f):
            return self.a * f + self.b

        # Exact analytical electrical solution when potential decreases monotonically (e.g. s=(0,0), t=(m,n))
        f_elec = (1.0 / self.a) * (self.B.T @ self.phi_base)
        if np.all(f_elec >= -1e-9) and np.allclose(self.B @ f_elec, self.p_base, atol=1e-6):
            f_base = np.maximum(0.0, f_elec)
            total_cost = float(np.sum(f_base * (self.a * f_base + self.b)))
            self._base_cache = (f_base, total_cost)
            return self._base_cache

        constraint = LinearConstraint(self.B[:-1], self.p_base[:-1], self.p_base[:-1])
        bounds = Bounds(0, np.inf)

        f_feas = np.linalg.pinv(self.B[:-1]) @ self.p_base[:-1]
        f_init = np.maximum(1e-4, f_feas)

        res = minimize(obj, f_init, jac=grad, constraints=constraint, bounds=bounds, method="SLSQP", options={"maxiter": 300})
        if not res.success:
            # Fallback to trust-constr if SLSQP fails
            res = minimize(obj, f_init, jac=grad, constraints=constraint, bounds=bounds, method="trust-constr")

        f_base = np.maximum(0.0, res.x)
        total_cost = float(np.sum(f_base * (self.a * f_base + self.b)))
        self._base_cache = (f_base, total_cost)
        return self._base_cache

    def solve_augmented_equilibrium(
        self, u_coord: Tuple[int, int], v_coord: Tuple[int, int], c: float = 0.0, d: float = 0.0
    ) -> Tuple[np.ndarray, float, float]:
        """
        Solves User Equilibrium on G + {u -> v}.
        Chord latency: ell^*(z) = c * z + d.
        Returns: (f_aug, total_cost, chord_flow_z).
        """
        if not hasattr(self, "_aug_cache"):
            self._aug_cache = {}

        cache_key = (u_coord, v_coord, c, d, self.b)
        if cache_key in self._aug_cache:
            return self._aug_cache[cache_key]

        u_idx = self.node_to_idx[u_coord]
        v_idx = self.node_to_idx[v_coord]

        edges_aug = self.edges + [(u_idx, v_idx)]
        num_aug_edges = len(edges_aug)

        B_aug = np.zeros((self.num_nodes, num_aug_edges), dtype=np.float64)
        for e_idx, (u, v) in enumerate(edges_aug):
            B_aug[u, e_idx] = 1.0
            B_aug[v, e_idx] = -1.0

        def obj(f):
            base_part = np.sum(0.5 * self.a * f[:-1]**2 + self.b * f[:-1])
            chord_part = 0.5 * c * f[-1]**2 + d * f[-1]
            return base_part + chord_part

        def grad(f):
            g = np.zeros_like(f)
            g[:-1] = self.a * f[:-1] + self.b
            g[-1] = c * f[-1] + d
            return g

        constraint = LinearConstraint(B_aug[:-1], self.p_base[:-1], self.p_base[:-1])
        bounds = Bounds(0, np.inf)

        f_feas = np.linalg.pinv(B_aug[:-1]) @ self.p_base[:-1]
        f_init = np.maximum(1e-4, f_feas)

        res = minimize(obj, f_init, jac=grad, constraints=constraint, bounds=bounds, method="SLSQP", options={"maxiter": 300})
        if not res.success:
            res = minimize(obj, f_init, jac=grad, constraints=constraint, bounds=bounds, method="trust-constr")

        f_aug = np.maximum(0.0, res.x)
        z_star = float(f_aug[-1])
        base_cost = np.sum(f_aug[:-1] * (self.a * f_aug[:-1] + self.b))
        chord_cost = z_star * (c * z_star + d)
        total_cost = float(base_cost + chord_cost)

        result = (f_aug, total_cost, z_star)
        self._aug_cache[cache_key] = result
        return result
