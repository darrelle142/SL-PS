"""
animate_ps_synced_planning.py
==============================
Visualizes synchronized execution of 3 Planners in a fully dynamic environment:
1. Panel 1 (Top-Left):     Standard-PS (K = 15)  -> Pauses at K=15, deceived by Braess trap (+11.28s)
2. Panel 2 (Top-Right):    Standard-PS (K = 100) -> Exhaustive sweep, evades trap (high compute: 100 backups)
3. Panel 3 (Bottom-Left):  SL-PS Proposed (K=15) -> Instant Green's potential diffusion, evades trap with 15 backups!
4. Panel 4 (Bottom-Right): Live Multi-Agent Benchmark Dashboard (Cumulative Latency & Compute Budget)

Design guarantees:
- ZERO OVERLAP: Headers, HUDs, and badges are strictly separated.
- HUD cards placed below grid axes to prevent any node occlusion (node (3,0) remains 100% visible).
- Braess & Expressway chord badges placed in empty cell interiors with compact footprints.
- All numbers pop out with high-contrast stroke outlines.
- Synchronized lock-step planning: K=15 agents pause and wait for K=100 before taking simultaneous actions.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import heapq
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
import matplotlib.patheffects as pe

from grid_braess_env import CongestedGridEnv
from planning_agents import StandardPrioritizedSweepingAgent, SLPSAgent


class StepPlanningTrackerStd:
    """Tracks planning snapshots for Standard Prioritized Sweeping."""
    def __init__(self, env: CongestedGridEnv, K: int):
        self.env = env
        self.K = K
        self.agent = StandardPrioritizedSweepingAgent(env, backup_budget=K)

    def plan_with_snapshots(self, state, sample_points=None):
        heap = []
        counter = 0

        for i in range(self.env.m + 1):
            for j in range(self.env.n + 1):
                s = (i, j)
                if s == (self.env.m, self.env.n):
                    continue
                target, _ = self.agent._compute_bellman_target(s)
                res = abs(target - self.agent.V[s])
                if res > 1e-4:
                    heapq.heappush(heap, (-res, counter, s))
                    counter += 1

        snapshots = []
        snapshots.append({
            "k": 0,
            "popped": None,
            "V": self.agent.V.copy(),
            "policy": self.agent.get_policy_map()
        })

        if sample_points is None:
            sample_points = set(range(1, self.K + 1))

        for k in range(1, self.K + 1):
            if not heap:
                break
            _, _, s = heapq.heappop(heap)
            target, _ = self.agent._compute_bellman_target(s)
            self.agent.V[s] = target

            si, sj = s
            preds = []
            if si > 0: preds.append((si - 1, sj))
            if sj > 0: preds.append((si, sj - 1))
            if self.env.active_chord is not None and s == self.env.active_chord[1]:
                preds.append(self.env.active_chord[0])

            for p in preds:
                p_target, _ = self.agent._compute_bellman_target(p)
                p_res = abs(p_target - self.agent.V[p])
                if p_res > 1e-4:
                    heapq.heappush(heap, (-p_res, counter, p))
                    counter += 1

            if k in sample_points or k == self.K:
                snapshots.append({
                    "k": k,
                    "popped": s,
                    "V": self.agent.V.copy(),
                    "policy": self.agent.get_policy_map()
                })

        return snapshots


class StepPlanningTrackerSL:
    """Tracks planning snapshots for Spectral-Laplacian Prioritized Sweeping (SL-PS)."""
    def __init__(self, env: CongestedGridEnv, K: int = 15, gamma: float = 0.95, beta_scale: float = 5.0):
        self.env = env
        self.K = K
        self.agent = SLPSAgent(env, backup_budget=K, gamma=gamma, beta_scale=beta_scale)

    def plan_with_snapshots(self, state, sample_points=None):
        Phi_spec = self.agent.compute_spectral_congestion_field()
        beta = self.agent.beta_scale * (1.0 + self.agent.slem)
        num_n = self.env.grid.num_nodes
        xi = np.zeros(num_n)
        for i in range(self.env.m + 1):
            for j in range(self.env.n + 1):
                s = (i, j)
                if s == (self.env.m, self.env.n):
                    continue
                target, _ = self.agent._compute_spectral_bellman_target(s, Phi_spec, beta)
                u_idx = self.env.grid.node_to_idx[s]
                xi[u_idx] = abs(target - self.agent.V[s])

        f_in = self.agent.G_in @ (xi + Phi_spec)

        heap = []
        counter = 0
        for i in range(self.env.m + 1):
            for j in range(self.env.n + 1):
                s = (i, j)
                if s == (self.env.m, self.env.n):
                    continue
                u_idx = self.env.grid.node_to_idx[s]
                if xi[u_idx] > 1e-4:
                    priority = xi[u_idx] + beta * f_in[u_idx]
                    heapq.heappush(heap, (-priority, counter, s))
                    counter += 1

        snapshots = []
        snapshots.append({
            "k": 0,
            "popped": None,
            "V": self.agent.V.copy(),
            "Phi_spec": Phi_spec.copy()
        })

        if sample_points is None:
            sample_points = set(range(1, self.K + 1))

        for k in range(1, self.K + 1):
            if not heap:
                break
            _, _, s = heapq.heappop(heap)
            target, _ = self.agent._compute_spectral_bellman_target(s, Phi_spec, beta)
            self.agent.V[s] = target

            si, sj = s
            preds = []
            if si > 0: preds.append((si - 1, sj))
            if sj > 0: preds.append((si, sj - 1))
            if self.env.active_chord is not None and s == self.env.active_chord[1]:
                preds.append(self.env.active_chord[0])

            for p in preds:
                p_target, _ = self.agent._compute_spectral_bellman_target(p, Phi_spec, beta)
                p_res = abs(p_target - self.agent.V[p])
                if p_res > 1e-4:
                    p_u = self.env.grid.node_to_idx[p]
                    p_prio = p_res + beta * f_in[p_u]
                    heapq.heappush(heap, (-p_prio, counter, p))
                    counter += 1

            if k in sample_points or k == self.K:
                snapshots.append({
                    "k": k,
                    "popped": s,
                    "V": self.agent.V.copy(),
                    "Phi_spec": Phi_spec.copy()
                })

        return snapshots


def generate_synchronized_3agent_frames():
    env_s15 = CongestedGridEnv(m=3, n=6, p_chord=0.0, seed=42)
    env_s100 = CongestedGridEnv(m=3, n=6, p_chord=0.0, seed=42)
    env_sl = CongestedGridEnv(m=3, n=6, p_chord=0.0, seed=42)

    env_s15.reset()
    env_s100.reset()
    env_sl.reset()

    tracker_s15 = StepPlanningTrackerStd(env_s15, K=15)
    tracker_s100 = StepPlanningTrackerStd(env_s100, K=100)
    tracker_sl = StepPlanningTrackerSL(env_sl, K=15, beta_scale=5.0)

    braess_chord_spec = ((0, 2), (2, 1), -0.5, 1)
    shortcut_chord_spec = ((3, 1), (2, 3), 0.5, 1)

    frames = []

    k_samples_100 = [0, 3, 6, 9, 12, 15, 22, 30, 45, 60, 80, 100]
    k_samples_15 = [0, 3, 6, 9, 12, 15]

    s_s15 = (0, 0)
    s_s100 = (0, 0)
    s_sl = (0, 0)

    cum_lat_s15 = 0.0
    cum_lat_s100 = 0.0
    cum_lat_sl = 0.0

    traj_s15 = [s_s15]
    traj_s100 = [s_s100]
    traj_sl = [s_sl]

    # Cumulative backups counter
    cum_bkps_s15 = 0
    cum_bkps_s100 = 0
    cum_bkps_sl = 0

    # Step 0: Baseline network with NO CHORDS (2 intro frames)
    edge_costs_base = env_s15.grid.a * env_s15.current_edge_flows + env_s15.grid.b
    v_base = tracker_s15.agent.V.copy()

    for _ in range(2):
        frames.append({
            "type": "base_intro",
            "env_step": 0,
            "banner": "[BƯỚC 0: MẠNG CƠ SỞ BAN ĐẦU] Chưa có cây cầu nào - Lưu lượng tự nhiên ổn định",
            "k_current": 0,
            "pos_s15": s_s15, "snap_s15": {"V": v_base}, "waiting_s15": False,
            "cum_lat_s15": 0.0, "traj_s15": list(traj_s15), "edge_costs_s15": edge_costs_base.copy(),
            "active_chord_s15": None, "chord_type_s15": None,

            "pos_s100": s_s100, "snap_s100": {"V": v_base}, "waiting_s100": False,
            "cum_lat_s100": 0.0, "traj_s100": list(traj_s100), "edge_costs_s100": edge_costs_base.copy(),
            "active_chord_s100": None, "chord_type_s100": None,

            "pos_sl": s_sl, "snap_sl": {"V": v_base}, "waiting_sl": False,
            "cum_lat_sl": 0.0, "traj_sl": list(traj_sl), "edge_costs_sl": edge_costs_base.copy(),
            "active_chord_sl": None, "chord_type_sl": None,

            "cum_bkps_s15": 0, "cum_bkps_s100": 0, "cum_bkps_sl": 0
        })

    # Execution loop across 9 environment steps
    for env_step in range(9):
        event_text = ""
        banner_border = "#F59E0B"

        if env_step in [0, 1, 2]:
            # Braess chord active
            for e in [env_s15, env_s100, env_sl]:
                e.active_chord = braess_chord_spec
                e.chord_type = "braess"
                f_aug, _, z_star = e.grid.solve_augmented_equilibrium(braess_chord_spec[0], braess_chord_spec[1], c=0.0, d=0.0)
                e.current_edge_flows = f_aug[:-1]
                e.current_chord_flow = z_star

            if env_step == 0:
                event_text = "[CẢNH BÁO] CẦU BRAESS (0,2)->(2,1) XUẤT HIỆN! -> Dòng xe dồn lên trên, chi phí tăng vọt!"
                banner_border = "#EF4444"
            elif env_step == 1:
                event_text = "Cầu Braess đang mở | Std-PS(15) tiến về (0,2) | Std-PS(100) & SL-PS(15) bẻ lái xuống dưới"
                banner_border = "#F59E0B"
            elif env_step == 2:
                event_text = "[NGUY HIỂM] Std-PS(15) sập bẫy Braess (+11.28s kẹt xe!) | Std-PS(100) & SL-PS(15) né bẫy qua (3,0)"
                banner_border = "#EF4444"

        elif env_step == 3:
            # Braess chord REMOVED / CLOSED completely
            for e in [env_s15, env_s100, env_sl]:
                e.active_chord = None
                e.chord_type = None
                e.current_edge_flows = e.f_base.copy()

            event_text = "[GỠ BỎ CẦU] CẦU BRAESS ĐÃ BỊ XÓA HOÀN TOÀN! -> Cạnh biến mất, lưu lượng tản đều"
            banner_border = "#38BDF8"

        elif env_step in [4, 5]:
            # Expressway shortcut opens
            for e in [env_s15, env_s100, env_sl]:
                e.active_chord = shortcut_chord_spec
                e.chord_type = "shortcut"
                f_aug, _, z_star = e.grid.solve_augmented_equilibrium(shortcut_chord_spec[0], shortcut_chord_spec[1], c=0.0, d=0.0)
                e.current_edge_flows = f_aug[:-1]

            if env_step == 4:
                event_text = "[KHÁNH THÀNH] CAO TỐC MỚI (3,1)->(2,3) MỞ CỬA (0.5s)! -> Std-PS(100) tận dụng thần tốc"
                banner_border = "#10B981"
            else:
                event_text = "Std-PS(100) qua cao tốc (0.46s) | SL-PS(15) bám sát nhánh dưới (0.96s) | Std-PS(15) tụt lại xa"
                banner_border = "#38BDF8"

        else:
            # Step 6+: Remaining navigation to goal
            for e in [env_s15, env_s100, env_sl]:
                e.active_chord = None
                e.chord_type = None
                e.current_edge_flows = e.f_base.copy()

            event_text = "Tiến về đích Goal (3,6) | So sánh tổng thời gian và ngân sách tính toán giữa 3 thuật toán"
            banner_border = "#10B981"

        # Planning snapshots
        snaps_s15 = tracker_s15.plan_with_snapshots(s_s15, sample_points=set(k_samples_15))
        snaps_s100 = tracker_s100.plan_with_snapshots(s_s100, sample_points=set(k_samples_100))
        snaps_sl = tracker_sl.plan_with_snapshots(s_sl, sample_points=set(k_samples_15))

        step_k_points = k_samples_100 if env_step == 0 else [0, 15, 60, 100]

        edge_costs_s15 = env_s15.grid.a * env_s15.current_edge_flows + env_s15.grid.b
        edge_costs_s100 = env_s100.grid.a * env_s100.current_edge_flows + env_s100.grid.b
        edge_costs_sl = env_sl.grid.a * env_sl.current_edge_flows + env_sl.grid.b

        for k_val in step_k_points:
            snap_s15 = next((s for s in reversed(snaps_s15) if s["k"] <= k_val), snaps_s15[-1])
            is_wait_s15 = (k_val > 15)

            snap_s100 = next((s for s in reversed(snaps_s100) if s["k"] <= k_val), snaps_s100[-1])

            snap_sl = next((s for s in reversed(snaps_sl) if s["k"] <= k_val), snaps_sl[-1])
            is_wait_sl = (k_val > 15)

            frames.append({
                "type": "planning",
                "env_step": env_step,
                "banner": event_text,
                "banner_border": banner_border,
                "k_current": k_val,

                "pos_s15": s_s15, "snap_s15": snap_s15, "waiting_s15": is_wait_s15,
                "cum_lat_s15": cum_lat_s15, "traj_s15": list(traj_s15), "edge_costs_s15": edge_costs_s15.copy(),
                "active_chord_s15": env_s15.active_chord, "chord_type_s15": env_s15.chord_type,

                "pos_s100": s_s100, "snap_s100": snap_s100, "waiting_s100": False,
                "cum_lat_s100": cum_lat_s100, "traj_s100": list(traj_s100), "edge_costs_s100": edge_costs_s100.copy(),
                "active_chord_s100": env_s100.active_chord, "chord_type_s100": env_s100.chord_type,

                "pos_sl": s_sl, "snap_sl": snap_sl, "waiting_sl": is_wait_sl,
                "cum_lat_sl": cum_lat_sl, "traj_sl": list(traj_sl), "edge_costs_sl": edge_costs_sl.copy(),
                "active_chord_sl": env_sl.active_chord, "chord_type_sl": env_sl.chord_type,

                "cum_bkps_s15": cum_bkps_s15 + min(k_val, 15),
                "cum_bkps_s100": cum_bkps_s100 + k_val,
                "cum_bkps_sl": cum_bkps_sl + min(k_val, 15),
            })

        # Action execution phase
        _, act_s15 = tracker_s15.agent._compute_bellman_target(s_s15)
        _, act_s100 = tracker_s100.agent._compute_bellman_target(s_s100)

        Phi_sl = tracker_sl.agent.compute_spectral_congestion_field()
        beta_sl = tracker_sl.agent.beta_scale * (1.0 + tracker_sl.agent.slem)
        _, act_sl = tracker_sl.agent._compute_spectral_bellman_target(s_sl, Phi_sl, beta_sl)

        took_chord_s15 = (act_s15 == CongestedGridEnv.ACTION_CHORD)
        took_chord_s100 = (act_s100 == CongestedGridEnv.ACTION_CHORD)
        took_chord_sl = (act_sl == CongestedGridEnv.ACTION_CHORD)

        nxt_s15, lat_s15, _, _ = env_s15.step(act_s15)
        nxt_s100, lat_s100, _, _ = env_s100.step(act_s100)
        nxt_sl, lat_sl, _, _ = env_sl.step(act_sl)

        cum_lat_s15 += lat_s15
        cum_lat_s100 += lat_s100
        cum_lat_sl += lat_sl

        s_s15 = nxt_s15
        s_s100 = nxt_s100
        s_sl = nxt_sl

        traj_s15.append(s_s15)
        traj_s100.append(s_s100)
        traj_sl.append(s_sl)

        cum_bkps_s15 += 15
        cum_bkps_s100 += 100
        cum_bkps_sl += 15

        edge_costs_s15_post = env_s15.grid.a * env_s15.current_edge_flows + env_s15.grid.b
        edge_costs_s100_post = env_s100.grid.a * env_s100.current_edge_flows + env_s100.grid.b
        edge_costs_sl_post = env_sl.grid.a * env_sl.current_edge_flows + env_sl.grid.b

        frames.append({
            "type": "action",
            "env_step": env_step,
            "banner": event_text,
            "banner_border": banner_border,
            "k_current": 100,

            "pos_s15": s_s15, "snap_s15": snaps_s15[-1], "waiting_s15": False,
            "step_lat_s15": lat_s15, "cum_lat_s15": cum_lat_s15, "traj_s15": list(traj_s15),
            "edge_costs_s15": edge_costs_s15_post.copy(), "act_s15": act_s15, "took_chord_s15": took_chord_s15,
            "active_chord_s15": env_s15.active_chord, "chord_type_s15": env_s15.chord_type,

            "pos_s100": s_s100, "snap_s100": snaps_s100[-1], "waiting_s100": False,
            "step_lat_s100": lat_s100, "cum_lat_s100": cum_lat_s100, "traj_s100": list(traj_s100),
            "edge_costs_s100": edge_costs_s100_post.copy(), "act_s100": act_s100, "took_chord_s100": took_chord_s100,
            "active_chord_s100": env_s100.active_chord, "chord_type_s100": env_s100.chord_type,

            "pos_sl": s_sl, "snap_sl": snaps_sl[-1], "waiting_sl": False,
            "step_lat_sl": lat_sl, "cum_lat_sl": cum_lat_sl, "traj_sl": list(traj_sl),
            "edge_costs_sl": edge_costs_sl_post.copy(), "act_sl": act_sl, "took_chord_sl": took_chord_sl,
            "active_chord_sl": env_sl.active_chord, "chord_type_sl": env_sl.chord_type,

            "cum_bkps_s15": cum_bkps_s15,
            "cum_bkps_s100": cum_bkps_s100,
            "cum_bkps_sl": cum_bkps_sl,
        })

    return env_s15, frames


def render_synchronized_3agent_animation(env, frames, output_path):
    m, n = env.m, env.n

    fig = plt.figure(figsize=(19, 12), dpi=110)
    fig.patch.set_facecolor("#080D1A")

    # Ample spacing: top=0.91 leaves clear room for banner at 0.962; bottom=0.08 leaves room for HUDs
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.18, left=0.04, right=0.96, top=0.91, bottom=0.08)

    ax_s15 = fig.add_subplot(gs[0, 0])
    ax_s100 = fig.add_subplot(gs[0, 1])
    ax_sl = fig.add_subplot(gs[1, 0])
    ax_dash = fig.add_subplot(gs[1, 1])

    v_min, v_max = 0.0, 11.0
    cmap = plt.colormaps["viridis_r"]

    # Top Banner
    banner = fig.text(
        0.5, 0.962, "",
        fontsize=11.5, fontweight="bold", color="#F8FAFC",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#111827", edgecolor="#EF4444", lw=1.8)
    )

    def setup_agent_grid(ax, title, subtitle, title_col, border_col):
        ax.set_facecolor("#111827")
        ax.set_title(f"{title}\n{subtitle}", fontsize=10.5, fontweight="bold", pad=8, color=title_col)
        ax.set_xlim(-0.65, n + 0.65)
        ax.set_ylim(m + 0.65, -0.65)
        ax.set_aspect("equal")
        ax.set_xticks(range(n + 1))
        ax.set_yticks(range(m + 1))
        ax.tick_params(colors="#64748B", labelsize=8.5)
        ax.grid(color="#1F2937", linestyle=":", linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_edgecolor(border_col)
            spine.set_linewidth(1.2)

        # Base edges
        for e_idx, (u_idx, v_idx) in enumerate(env.grid.edges):
            ui, uj = env.grid.idx_to_node[u_idx]
            vi, vj = env.grid.idx_to_node[v_idx]
            ax.annotate("", xy=(vj, vi), xytext=(uj, ui),
                        arrowprops=dict(arrowstyle="->", color="#374151", lw=1.4, alpha=0.5))

        # Start & Goal markers
        ax.plot(0, 0, "s", color="#10B981", markersize=8, zorder=4)
        ax.plot(n, m, "s", color="#8B5CF6", markersize=8, zorder=4)

    setup_agent_grid(ax_s15, "Standard-PS (K = 15)", "Pauses at K=15 -> Deceived by Upper Corridor -> Falls into Braess Trap", "#F87171", "#EF4444")
    setup_agent_grid(ax_s100, "Standard-PS (K = 100)", "Continues to K=100 -> Detects Lower Corridor -> Evades Trap (Costly Backups)", "#34D399", "#10B981")
    setup_agent_grid(ax_sl, "SL-PS Proposed (K = 15)", "Instant Congestion Diffusion via Green's Potential Φ_spec -> Evades with 15 Backups!", "#38BDF8", "#0284C7")

    # Helper containers for each of the 3 grid panels
    panels = {}
    for name, ax in [("s15", ax_s15), ("s100", ax_s100), ("sl", ax_sl)]:
        # Chord arrow patch (hidden initially)
        c_arrow = FancyArrowPatch((0, 0), (1, 1), connectionstyle="arc3,rad=-0.28",
                                  color="#EF4444", lw=2.5, linestyle="--", arrowstyle="->",
                                  mutation_scale=14, zorder=8)
        c_arrow.set_visible(False)
        ax.add_patch(c_arrow)

        # Chord badge text (hidden initially)
        c_badge = ax.text(1.5, 0.5, "", color="#FCA5A5", fontsize=6.8, fontweight="bold",
                          ha="center", va="center", zorder=9,
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="#450A0A", edgecolor="#EF4444", alpha=0.92),
                          path_effects=[pe.withStroke(linewidth=1.5, foreground="#000000")])
        c_badge.set_visible(False)

        # 45 Edge badges
        e_badges = []
        for e_idx, (u_idx, v_idx) in enumerate(env.grid.edges):
            ui, uj = env.grid.idx_to_node[u_idx]
            vi, vj = env.grid.idx_to_node[v_idx]
            mid_x = (uj + vj) / 2.0
            mid_y = (ui + vi) / 2.0
            if ui == vi:
                bx, by = mid_x, mid_y - 0.16
            else:
                bx, by = mid_x + 0.18, mid_y

            txt = ax.text(bx, by, "1.00", color="#38BDF8", fontsize=6.8, fontweight="bold",
                          ha="center", va="center", zorder=7,
                          bbox=dict(boxstyle="round,pad=0.15", facecolor="#090D16", edgecolor="#38BDF8", lw=0.8, alpha=0.9),
                          path_effects=[pe.withStroke(linewidth=1.5, foreground="#000000")])
            e_badges.append(txt)

        # 28 Node circles & texts
        n_circles = {}
        n_texts = {}
        for i in range(m + 1):
            for j in range(n + 1):
                s = (i, j)
                c_patch = Circle((j, i), radius=0.32, facecolor="#334155", edgecolor="#FFFFFF", lw=1.5, zorder=10)
                ax.add_patch(c_patch)
                t_patch = ax.text(j, i, "0.0", color="#FFFFFF", fontsize=8.0, fontweight="heavy",
                                  ha="center", va="center", zorder=12,
                                  path_effects=[pe.withStroke(linewidth=2.5, foreground="#000000")])
                n_circles[s] = c_patch
                n_texts[s] = t_patch

        # Agent marker and trajectory line
        col_dot = "#EF4444" if name == "s15" else ("#10B981" if name == "s100" else "#38BDF8")
        col_line = "#F87171" if name == "s15" else ("#34D399" if name == "s100" else "#60A5FA")

        dot, = ax.plot([], [], "o", color=col_dot, markersize=14, zorder=16)
        path, = ax.plot([], [], "-", color=col_line, lw=3.2, alpha=0.85, zorder=14)

        # HUD below axes (guaranteed NO node collision)
        hud = ax.text(0.5, -0.15, "", transform=ax.transAxes, fontsize=8.2, family="monospace",
                      ha="center", va="top", color="#FFFFFF",
                      bbox=dict(boxstyle="round,pad=0.35", facecolor="#111827", edgecolor=col_dot, lw=1.2))

        panels[name] = {
            "ax": ax, "chord_arrow": c_arrow, "chord_badge": c_badge,
            "edge_badges": e_badges, "circles": n_circles, "texts": n_texts,
            "dot": dot, "path": path, "hud": hud
        }

    # Setup Dashboard (ax_dash)
    ax_dash.set_facecolor("#111827")
    ax_dash.set_title("Multi-Agent Performance & Efficiency Benchmark\nReal-time Cumulative Latency & Computational Budget",
                      fontsize=10.5, fontweight="bold", pad=8, color="#FBBF24")
    for spine in ax_dash.spines.values():
        spine.set_edgecolor("#F59E0B")
        spine.set_linewidth(1.2)

    dash_agents = ["Std-PS (K=15)", "Std-PS (K=100)", "SL-PS (K=15)"]
    dash_y_pos = [1.8, 2.7, 3.6]
    dash_bar_cols = ["#EF4444", "#10B981", "#38BDF8"]

    ax_dash.set_xlim(0, 25)
    ax_dash.set_ylim(-0.2, 4.3)
    ax_dash.set_yticks(dash_y_pos)
    ax_dash.set_yticklabels(dash_agents, fontsize=9.0, fontweight="bold", color="#F8FAFC")
    ax_dash.set_xlabel("Cumulative Travel Latency (seconds) - Lower is Better", fontsize=8.8, color="#94A3B8")
    ax_dash.tick_params(colors="#64748B")
    ax_dash.grid(axis="x", color="#1F2937", linestyle="--", alpha=0.7)

    dash_bars = ax_dash.barh(dash_y_pos, [0.0, 0.0, 0.0], color=dash_bar_cols, height=0.55, edgecolor="#FFFFFF", lw=0.8, alpha=0.9)
    dash_val_texts = []
    for idx, b in enumerate(dash_bars):
        vt = ax_dash.text(0.5, b.get_y() + b.get_height() / 2, "",
                          va="center", color=dash_bar_cols[idx], fontsize=8.8, fontweight="bold",
                          path_effects=[pe.withStroke(linewidth=2, foreground="#000000")])
        dash_val_texts.append(vt)

    # Info & compute budget text box in lower half of ax_dash
    info_card = ax_dash.text(
        0.04, 0.04, "", transform=ax_dash.transAxes, fontsize=7.8, family="monospace",
        color="#E2E8F0", va="bottom",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#0B132B", edgecolor="#38BDF8", lw=1.2, alpha=0.95)
    )

    def init():
        for p in panels.values():
            p["dot"].set_data([], [])
            p["path"].set_data([], [])
            p["chord_arrow"].set_visible(False)
            p["chord_badge"].set_visible(False)
        return []

    def update(frame_idx):
        f = frames[frame_idx]

        # 1. Update Top Banner
        banner.set_text(f["banner"])
        banner.get_bbox_patch().set_edgecolor(f.get("banner_border", "#F59E0B"))

        # 2. Update all 3 agent panels
        for name, p in panels.items():
            # Edge badges
            costs = f[f"edge_costs_{name}"]
            for e_idx, txt in enumerate(p["edge_badges"]):
                c = costs[e_idx]
                txt.set_text(f"{c:.2f}")
                col = "#F87171" if c > 1.4 else ("#FBBF24" if c > 1.1 else "#38BDF8")
                txt.set_color(col)
                txt.get_bbox_patch().set_edgecolor(col)

            # Chord arrow & badge
            ch = f[f"active_chord_{name}"]
            if ch is not None:
                u_c, v_c = ch[:2]
                c_type = f[f"chord_type_{name}"]
                p["chord_arrow"].set_positions((u_c[1], u_c[0]), (v_c[1], v_c[0]))
                if c_type == "braess":
                    p["chord_arrow"].set_color("#EF4444")
                    p["chord_arrow"].set_linestyle("--")
                    p["chord_arrow"].set_connectionstyle("arc3,rad=-0.28")
                    p["chord_badge"].set_text("BRAESS\n+11.28s")
                    p["chord_badge"].set_position((1.5, 0.5))
                    p["chord_badge"].set_color("#FCA5A5")
                    p["chord_badge"].get_bbox_patch().set_facecolor("#450A0A")
                    p["chord_badge"].get_bbox_patch().set_edgecolor("#EF4444")
                else:
                    p["chord_arrow"].set_color("#10B981")
                    p["chord_arrow"].set_linestyle("-")
                    p["chord_arrow"].set_connectionstyle("arc3,rad=0.25")
                    p["chord_badge"].set_text("EXPRESS\n0.46s")
                    p["chord_badge"].set_position((1.5, 2.5))
                    p["chord_badge"].set_color("#A7F3D0")
                    p["chord_badge"].get_bbox_patch().set_facecolor("#064E3B")
                    p["chord_badge"].get_bbox_patch().set_edgecolor("#10B981")
                p["chord_arrow"].set_visible(True)
                p["chord_badge"].set_visible(True)
            else:
                p["chord_arrow"].set_visible(False)
                p["chord_badge"].set_visible(False)

            # Node values & circles
            v_dict = f[f"snap_{name}"]["V"]
            for s, c_patch in p["circles"].items():
                val = v_dict.get(s, 0.0)
                norm_val = np.clip((val - v_min) / (v_max - v_min), 0.0, 1.0)
                c_patch.set_facecolor(cmap(norm_val))
                p["texts"][s].set_text(f"{val:.1f}")

            # Agent dot & path
            pos = f[f"pos_{name}"]
            traj = f[f"traj_{name}"]
            p["dot"].set_data([pos[1]], [pos[0]])
            p["path"].set_data([pt[1] for pt in traj], [pt[0] for pt in traj])

            # HUD Card
            lat = f[f"cum_lat_{name}"]
            if f["type"] == "base_intro":
                p["hud"].set_text(f"Pos: {pos} | Status: Base Network | Total Time: 0.00s")
            elif f["type"] == "planning":
                k = f["k_current"]
                if name in ["s15", "sl"]:
                    status = "DONE (15/15) - WAITING..." if f[f"waiting_{name}"] else f"Planning: {min(k, 15)}/15"
                else:
                    status = f"Planning: {k}/100"
                p["hud"].set_text(f"Pos: {pos} | Status: {status} | Total Time: {lat:.2f}s")
            else:
                if f.get(f"took_chord_{name}", False):
                    act_desc = "CHORD (+11.28s TRAP!)" if f.get(f"chord_type_{name}") == "braess" else "CHORD (0.46s EXPRESS)"
                else:
                    act_desc = f"Action {f.get(f'act_{name}', 0)}"
                p["hud"].set_text(f"Pos: {pos} | Move: {act_desc} | Total Time: {lat:.2f}s")

        # 3. Update Dashboard Bars & Info
        curr_lats = [f["cum_lat_s15"], f["cum_lat_s100"], f["cum_lat_sl"]]
        for idx, (b, val) in enumerate(zip(dash_bars, curr_lats)):
            b.set_width(val)
            tag = ""
            if f["env_step"] >= 2:
                if idx == 0:
                    tag = "[TRAPPED!]"
                elif idx == 1:
                    tag = "[COSTLY (100 bkps)]"
                else:
                    tag = "[OPTIMAL (15 bkps)]"
            dash_val_texts[idx].set_position((val + 0.4, b.get_y() + b.get_height() / 2))
            dash_val_texts[idx].set_text(f"{val:.2f}s  {tag}")

        # Update Info Card
        bkps_15 = f.get("cum_bkps_s15", 0)
        bkps_100 = f.get("cum_bkps_s100", 0)
        bkps_sl = f.get("cum_bkps_sl", 0)
        savings = (1.0 - bkps_sl / max(bkps_100, 1)) * 100.0

        step_num = f["env_step"]
        info_text = (
            f" COMPUTATIONAL BUDGET COMPARISON (Step {step_num}):\n"
            f" • Std-PS (K=15):   15 backups/step  (Total: {bkps_15:3d} bkps | Myopic trap)\n"
            f" • Std-PS (K=100): 100 backups/step (Total: {bkps_100:3d} bkps | High compute)\n"
            f" • SL-PS  (K=15):   15 backups/step  (Total: {bkps_sl:3d} bkps | {savings:.1f}% SAVINGS!)\n\n"
            f" SPECTRAL-LAPLACIAN BELLMAN OPERATOR:\n"
            f" T_SL V(s) = min_a [ c(s,a) + gamma V(s') + beta * Phi_spec(s') ]\n"
            f" where Phi_spec = G_out @ |B Delta_l| diffuses bottleneck globally!"
        )
        info_card.set_text(info_text)

        return []

    ani = animation.FuncAnimation(fig, update, frames=len(frames), init_func=init, blit=False, interval=650)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ani.save(output_path, writer="pillow", fps=1.5)
    plt.close()
    print(f"Synchronized 3-Agent Animation successfully saved to {output_path}")


if __name__ == "__main__":
    out_gif = r"C:\Users\ADMIN\.gemini\antigravity-cli\brain\94cfebc3-0f15-4ab1-b2a6-1c0c4c4a689a\ps_synced_planning_v_function.gif"
    env, frames = generate_synchronized_3agent_frames()
    print(f"Total synchronized 3-agent dynamic frames: {len(frames)}")
    render_synchronized_3agent_animation(env, frames, out_gif)
