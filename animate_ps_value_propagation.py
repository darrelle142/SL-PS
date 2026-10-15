"""
animate_ps_value_propagation.py
================================
Visualizes step-by-step how the Value Function V(s) evolves during Prioritized Sweeping (PS) backups
when a chord u -> w (with V_uw < 0) is dynamically added to the network.

Side-by-side comparison:
- Left: Standard-PS (falsely drops V(s) upstream, falling into the Braess trap)
- Right: SL-PS (injects Phi_spec, elevates Q(chord), routes policy away from trap)
"""

import os
import argparse
import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Dict, List, Tuple, Optional

from grid_braess_env import CongestedGridEnv
from planning_agents import StandardPrioritizedSweepingAgent, SLPSAgent


def capture_ps_backup_trace(env: CongestedGridEnv, agent_type: str = "std", K: int = 14):
    """
    Executes exactly K prioritized backups step-by-step, recording a snapshot at every step k.
    agent_type: 'std' (Standard-PS) or 'sl' (SL-PS)
    """
    m, n = env.m, env.n
    grid = env.grid
    num_n = grid.num_nodes

    # Initialize V
    V = {
        (i, j): float((m - i) + (n - j)) * grid.b
        for i in range(m + 1)
        for j in range(n + 1)
    }
    V[(m, n)] = 0.0

    # If SL-PS, compute spectral structures and Phi_spec
    if agent_type == "sl":
        sl_agent = SLPSAgent(env, backup_budget=K, beta_scale=6.0)
        Phi_spec = sl_agent.compute_spectral_congestion_field()
        beta = sl_agent.beta_scale * (1.0 + sl_agent.slem)
        G_in = sl_agent.G_in
    else:
        Phi_spec = np.zeros(num_n)
        beta = 0.0
        G_in = None

    def get_transition(s, a):
        i, j = s
        if a == CongestedGridEnv.ACTION_RIGHT:
            nxt = (i + 1, j)
            e_idx = env._get_edge_index(s, nxt)
            cost = env.grid.a * env.current_edge_flows[e_idx] + env.grid.b
        elif a == CongestedGridEnv.ACTION_DOWN:
            nxt = (i, j + 1)
            e_idx = env._get_edge_index(s, nxt)
            cost = env.grid.a * env.current_edge_flows[e_idx] + env.grid.b
        elif a == CongestedGridEnv.ACTION_UP:
            nxt = (i - 1, j)
            try:
                e_idx = env._get_edge_index(nxt, s)
                cost = env.grid.a * env.current_edge_flows[e_idx] + env.grid.b
            except ValueError:
                cost = env.grid.b
            cost += 0.05
        elif a == CongestedGridEnv.ACTION_LEFT:
            nxt = (i, j - 1)
            try:
                e_idx = env._get_edge_index(nxt, s)
                cost = env.grid.a * env.current_edge_flows[e_idx] + env.grid.b
            except ValueError:
                cost = env.grid.b
            cost += 0.05
        elif a == CongestedGridEnv.ACTION_CHORD and env.active_chord is not None:
            nxt = env.active_chord[1]
            cost = 0.0  # nominal chord latency = 0
        else:
            nxt = s
            cost = 10.0
        return nxt, cost

    def compute_target(s, curr_V):
        if s == (m, n):
            return 0.0, 0, {}
        valid = env.get_valid_actions(s)
        min_q = float("inf")
        best_a = valid[0]
        q_breakdown = {}
        for a in valid:
            nxt, cost = get_transition(s, a)
            nxt_idx = grid.node_to_idx[nxt]
            spec_pen = float(beta * Phi_spec[nxt_idx]) if agent_type == "sl" else 0.0
            q_val = cost + 0.95 * curr_V[nxt] + spec_pen
            q_breakdown[a] = (cost, 0.95 * curr_V[nxt], spec_pen, q_val)
            if q_val < min_q:
                min_q = q_val
                best_a = a
        return min_q, best_a, q_breakdown

    def get_policy_map(curr_V):
        pol = {}
        for i in range(m + 1):
            for j in range(n + 1):
                s = (i, j)
                if s == (m, n):
                    pol[s] = -1
                else:
                    _, a, _ = compute_target(s, curr_V)
                    pol[s] = a
        return pol

    # Snapshot 0: Baseline state before chord opens
    snapshots = []
    pol_0 = get_policy_map(V)
    snapshots.append({
        "backup_idx": 0,
        "popped_node": None,
        "priority_val": 0.0,
        "delta_v": 0.0,
        "V_table": V.copy(),
        "policy": pol_0.copy(),
        "predecessors": [],
        "note": "Baseline V_0(s) before Chord opens"
    })

    # When chord opens, compute initial priority queue
    heap = []
    counter = 0

    if agent_type == "sl":
        xi = np.zeros(num_n)
        for i in range(m + 1):
            for j in range(n + 1):
                s = (i, j)
                if s == (m, n):
                    continue
                target, _, _ = compute_target(s, V)
                u_idx = grid.node_to_idx[s]
                xi[u_idx] = abs(target - V[s])
        f_in = G_in @ (xi + Phi_spec)
        for i in range(m + 1):
            for j in range(n + 1):
                s = (i, j)
                if s == (m, n):
                    continue
                u_idx = grid.node_to_idx[s]
                prio = xi[u_idx] + beta * f_in[u_idx]
                if prio > 1e-4:
                    heapq.heappush(heap, (-prio, counter, s))
                    counter += 1
    else:
        for i in range(m + 1):
            for j in range(n + 1):
                s = (i, j)
                if s == (m, n):
                    continue
                target, _, _ = compute_target(s, V)
                res = abs(target - V[s])
                if res > 1e-4:
                    heapq.heappush(heap, (-res, counter, s))
                    counter += 1

    # Snapshot 1: Queue formed immediately when chord appears
    snapshots.append({
        "backup_idx": 0,
        "popped_node": None,
        "priority_val": -heap[0][0] if heap else 0.0,
        "delta_v": 0.0,
        "V_table": V.copy(),
        "policy": pol_0.copy(),
        "predecessors": [],
        "note": "Chord opens! Priority Queue populated"
    })

    # Run K backups step-by-step
    for k in range(1, K + 1):
        if not heap:
            break
        prio_neg, _, s_pop = heapq.heappop(heap)
        prio_val = -prio_neg
        target, best_a, _ = compute_target(s_pop, V)
        delta_v = target - V[s_pop]
        V[s_pop] = target

        # Find predecessors
        si, sj = s_pop
        predecessors = []
        if si > 0:
            predecessors.append((si - 1, sj))
        if sj > 0:
            predecessors.append((si, sj - 1))
        if getattr(env, "allow_backward", False):
            if si < m:
                predecessors.append((si + 1, sj))
            if sj < n:
                predecessors.append((si, sj + 1))
        if env.active_chord is not None and s_pop == env.active_chord[1]:
            predecessors.append(env.active_chord[0])

        # Push predecessors
        for p in predecessors:
            p_target, _, _ = compute_target(p, V)
            p_res = abs(p_target - V[p])
            if agent_type == "sl":
                p_idx = grid.node_to_idx[p]
                score = p_res + beta * f_in[p_idx]
            else:
                score = p_res
            if score > 1e-4:
                heapq.heappush(heap, (-score, counter, p))
                counter += 1

        curr_pol = get_policy_map(V)
        snapshots.append({
            "backup_idx": k,
            "popped_node": s_pop,
            "priority_val": prio_val,
            "delta_v": delta_v,
            "V_table": V.copy(),
            "policy": curr_pol.copy(),
            "predecessors": predecessors.copy(),
            "note": f"Backup #{k}: Updated {s_pop} -> V={target:.2f} (dV={delta_v:+.2f})"
        })

    return snapshots, Phi_spec


def render_ps_value_animation(save_path: str = "ps_value_propagation.gif", fps: int = 2):
    """
    Renders side-by-side animation of Standard-PS vs SL-PS value backups.
    """
    m, n = 3, 6
    env_std = CongestedGridEnv(m=m, n=n, a=1.0, b=1.0, q=1.0, p_chord=0.0, allow_backward=True)
    env_sl = CongestedGridEnv(m=m, n=n, a=1.0, b=1.0, q=1.0, p_chord=0.0, allow_backward=True)

    # Pick a prominent Braess chord: (0, 2) -> (2, 1) or (1, 4) -> (3, 3)
    braess_chord = ((0, 2), (2, 1), -0.05, 1)
    for c in env_std.braess_chords:
        if c[0] == (0, 2) or c[0] == (1, 4):
            braess_chord = c
            break

    u_c, v_c, V_uv, k_gain = braess_chord
    env_std.active_chord = braess_chord
    env_std.chord_type = "braess"
    env_sl.active_chord = braess_chord
    env_sl.chord_type = "braess"

    # Re-equilibrate traffic
    f_aug, _, z_star = env_std.grid.solve_augmented_equilibrium(u_c, v_c, c=0.0, d=0.0)
    env_std.current_edge_flows = f_aug[:-1]
    env_std.current_chord_flow = z_star
    env_sl.current_edge_flows = f_aug[:-1]
    env_sl.current_chord_flow = z_star

    K_steps = 14
    trace_std, _ = capture_ps_backup_trace(env_std, agent_type="std", K=K_steps)
    trace_sl, Phi_spec = capture_ps_backup_trace(env_sl, agent_type="sl", K=K_steps)

    num_frames = min(len(trace_std), len(trace_sl))

    # Setup Figure
    fig = plt.figure(figsize=(18, 9.5), dpi=100)
    fig.patch.set_facecolor("#0e1117")

    gs = fig.add_gridspec(2, 2, height_ratios=[4.5, 1.2], hspace=0.25, wspace=0.15)
    ax_std = fig.add_subplot(gs[0, 0])
    ax_sl = fig.add_subplot(gs[0, 1])
    ax_hud = fig.add_subplot(gs[1, :])

    # Range for V color scaling
    v_vals = [s["V_table"][k] for s in trace_std for k in s["V_table"]]
    v_min, v_max = min(v_vals), max(v_vals)

    def draw_grid_base(ax, title, is_sl=False):
        ax.set_facecolor("#161b22")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.6, n + 0.6)
        ax.set_ylim(m + 0.6, -0.6)
        ax.set_xticks(range(n + 1))
        ax.set_yticks(range(m + 1))
        ax.tick_params(colors="#8b949e", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.grid(True, color="#21262d", linestyle="--", linewidth=0.6, alpha=0.7)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10,
                     color="#58a6ff" if not is_sl else "#7ee787")

    def update_frame(frame_idx):
        ax_std.clear()
        ax_sl.clear()
        ax_hud.clear()

        snap_std = trace_std[frame_idx]
        snap_sl = trace_sl[frame_idx]

        draw_grid_base(ax_std, "STANDARD-PS (Local Residuals Only)", is_sl=False)
        draw_grid_base(ax_sl, "SL-PS (Spectral-Laplacian Bellman Operator)", is_sl=True)

        for ax, snap, is_sl in [(ax_std, snap_std, False), (ax_sl, snap_sl, True)]:
            V_curr = snap["V_table"]
            pol_curr = snap["policy"]
            popped = snap["popped_node"]
            preds = snap["predecessors"]

            # 1. Draw base edges (dim background lines)
            for i in range(m + 1):
                for j in range(n + 1):
                    if i < m:
                        ax.annotate("", xy=(j, i + 0.65), xytext=(j, i + 0.35),
                                    arrowprops=dict(arrowstyle="->", color="#30363d", lw=1.0, mutation_scale=8))
                    if j < n:
                        ax.annotate("", xy=(j + 0.65, i), xytext=(j + 0.35, i),
                                    arrowprops=dict(arrowstyle="->", color="#30363d", lw=1.0, mutation_scale=8))

            # 2. Draw active chord u -> w
            u_y, u_x = u_c
            w_y, w_x = v_c
            chord_color = "#f85149" if not is_sl else "#d29922"
            ax.annotate("", xy=(w_x, w_y), xytext=(u_x, u_y),
                        arrowprops=dict(arrowstyle="-|>", color=chord_color, lw=2.4, ls="--",
                                        mutation_scale=16, shrinkA=12, shrinkB=12))
            ax.text((u_x + w_x) / 2.0, (u_y + w_y) / 2.0 - 0.18,
                    f"Braess Chord (V={V_uv:.2f}<0)", color=chord_color,
                    fontsize=8.0, fontweight="bold", ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#0e1117", edgecolor=chord_color, alpha=0.85))

            # If SL-PS, draw glowing spectral potential halo around sink w_c
            if is_sl and frame_idx >= 1:
                for (ni, nj), pot in zip([(i, j) for i in range(m+1) for j in range(n+1)], Phi_spec):
                    if pot > 0.15:
                        circ = plt.Circle((nj, ni), 0.38 + pot * 0.10, color="#f85149", alpha=min(0.35, pot * 0.15), zorder=2)
                        ax.add_patch(circ)

            # 3. Draw nodes with heatmap of V(s)
            radius = 0.28
            for i in range(m + 1):
                for j in range(n + 1):
                    s = (i, j)
                    val = V_curr[s]
                    norm_val = np.clip((val - v_min) / (v_max - v_min + 1e-6), 0.0, 1.0)
                    cmap = plt.cm.viridis if not is_sl else plt.cm.plasma
                    node_col = cmap(norm_val)

                    is_popped = (s == popped)
                    is_pred = (s in preds)

                    border_col = "#e3b341" if is_popped else ("#58a6ff" if is_pred else "#8b949e")
                    border_lw = 2.8 if is_popped else (1.8 if is_pred else 1.0)
                    curr_rad = radius if not is_popped else (radius + 0.04)

                    circ = plt.Circle((j, i), curr_rad, facecolor=node_col, edgecolor=border_col,
                                      linewidth=border_lw, zorder=4)
                    ax.add_patch(circ)

                    # Text value of V - perfectly centered and never obscured
                    ax.text(j, i, f"{val:.1f}", color="#ffffff" if norm_val < 0.65 else "#000000",
                            fontsize=8.5, fontweight="bold", ha="center", va="center", zorder=5)

                    # Draw greedy action arrow OUTSIDE the circle (between nodes)
                    act = pol_curr.get(s, -1)
                    if s != (m, n):
                        if act == CongestedGridEnv.ACTION_RIGHT and i < m:
                            # Moves to (i+1, j) -> Downwards in y
                            start_x, start_y = j, i + curr_rad + 0.04
                            end_x, end_y = j, i + 1.0 - radius - 0.04
                            ax.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y),
                                        arrowprops=dict(arrowstyle="-|>", color="#ffffff", lw=2.2,
                                        mutation_scale=12, zorder=6))
                        elif act == CongestedGridEnv.ACTION_DOWN and j < n:
                            # Moves to (i, j+1) -> Rightwards in x
                            start_x, start_y = j + curr_rad + 0.04, i
                            end_x, end_y = j + 1.0 - radius - 0.04, i
                            ax.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y),
                                        arrowprops=dict(arrowstyle="-|>", color="#ffffff", lw=2.2,
                                        mutation_scale=12, zorder=6))
                        elif act == CongestedGridEnv.ACTION_CHORD and s == u_c:
                            # Points into the chord
                            vec_x, vec_y = w_x - u_x, w_y - u_y
                            dist = np.hypot(vec_x, vec_y)
                            start_x = j + (vec_x / dist) * (curr_rad + 0.04)
                            start_y = i + (vec_y / dist) * (curr_rad + 0.04)
                            end_x = j + (vec_x / dist) * 0.55
                            end_y = i + (vec_y / dist) * 0.55
                            ax.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y),
                                        arrowprops=dict(arrowstyle="-|>", color="#ffffff", lw=2.4,
                                        mutation_scale=14, zorder=6))

            # Highlight Source & Destination
            ax.text(0, -0.38, "START (0,0)", color="#79c0ff", fontsize=9, fontweight="bold", ha="center")
            ax.text(n, m + 0.45, "DEST (m,n)", color="#7ee787", fontsize=9, fontweight="bold", ha="center")

        # 4. Draw HUD / Control Panel
        ax_hud.set_facecolor("#161b22")
        for spine in ax_hud.spines.values():
            spine.set_color("#30363d")
        ax_hud.set_xticks([])
        ax_hud.set_yticks([])

        step_num = snap_std["backup_idx"]
        ax_hud.text(0.02, 0.85, f"PRIORITIZED SWEEPING BACKUP ITERATION: Step {step_num} / {K_steps}",
                    color="#f0f6fc", fontsize=12, fontweight="bold")

        # Standard-PS status card
        pop_s = snap_std["popped_node"]
        prio_s = snap_std["priority_val"]
        dv_s = snap_std["delta_v"]
        txt_s = (f"Standard-PS: Popped {pop_s} | Priority={prio_s:.2f} | dV={dv_s:+.2f}\n"
                 f"- Apparent 0-cost on chord creates large negative residual\n"
                 f"- Drops V(u) and propagates fake cheap cost UPSTREAM!")
        ax_hud.text(0.02, 0.55, txt_s, color="#ff7b72", fontsize=9.2, family="monospace", va="top")

        # SL-PS status card
        pop_l = snap_sl["popped_node"]
        prio_l = snap_sl["priority_val"]
        dv_l = snap_sl["delta_v"]
        txt_l = (f"SL-PS: Popped {pop_l} | Priority={prio_l:.2f} | dV={dv_l:+.2f}\n"
                 f"- Phi_spec diffuses continuous downstream congestion dome\n"
                 f"- Overwhelms 0-cost lure, forcing policy to BYPASS!")
        ax_hud.text(0.52, 0.55, txt_l, color="#7ee787", fontsize=9.2, family="monospace", va="top")

        # Legend / Explanation
        ax_hud.text(0.02, 0.08,
                    "Yellow Ring = State popped from Heap | Blue Ring = Predecessors queued | Numbers = Current V(s) | Arrows = Policy pi(s)",
                    color="#8b949e", fontsize=8.8, style="italic")

    anim = FuncAnimation(fig, update_frame, frames=num_frames, interval=800, repeat=True)

    print(f"Rendering {num_frames} frames to {save_path}...")
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer)
    plt.close(fig)
    print(f"Successfully generated: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default="ps_value_propagation.gif")
    parser.add_argument("--fps", type=int, default=2)
    args = parser.parse_args()

    render_ps_value_animation(save_path=args.save, fps=args.fps)
