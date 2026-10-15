"""
animate_counterflow.py
======================
Visualizes SL-PS vs. Standard-PS under Counter-Directional / Opposing Flow:
- Agent navigates South-East: (0, 0) -> (8, 16).
- Background traffic flows North-West (HEAD-ON OPPOSING FLOW): (6, 12) -> (2, 4).
- Demonstrates:
    * How counter-flow forms an oncoming shockwave barrier in the central corridor.
    * How Spectral Potential Phi_spec = (I - P + eta*I)^(-1) |B Delta_ell| detects the head-on danger.
    * Standard-PS blunders straight into the oncoming traffic storm (high latency).
    * SL-PS executes a Lateral Flanking Maneuver (tat suon), detouring safely around the storm.

Usage:
  python animate_counterflow.py --save sl_ps_counterflow_evasion.gif
  python animate_counterflow.py --show
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
from matplotlib.colors import Normalize

from spectral_braess_core import GridNetwork


class CounterFlowGridNetwork(GridNetwork):
    """
    Grid Network with Two-Way / Counter-Flow Support.
    Allows background traffic to flow North-West opposing the South-East agent movement.
    """
    def __init__(self, m=8, n=16, a=1.5, b=1.0, q=3.0, s_bg=(5, 11), t_bg=(2, 4)):
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

        # Bidirectional street network
        self.edges = []
        for i in range(m + 1):
            for j in range(n + 1):
                u = self.node_to_idx[(i, j)]
                # Forward (Agent directions: Down, Right)
                if i < m:
                    self.edges.append((u, self.node_to_idx[(i + 1, j)]))
                if j < n:
                    self.edges.append((u, self.node_to_idx[(i, j + 1)]))
                # Counter-flow (Traffic directions: Up, Left)
                if i > 0:
                    self.edges.append((u, self.node_to_idx[(i - 1, j)]))
                if j > 0:
                    self.edges.append((u, self.node_to_idx[(i, j - 1)]))

        self.num_edges = len(self.edges)
        self._build_spectral_operators()


def simulate_counterflow_navigation(m=8, n=16, s_bg=(5, 11), t_bg=(2, 4), q=3.0, beta=7.0):
    net = CounterFlowGridNetwork(m=m, n=n, s_bg=s_bg, t_bg=t_bg, q=q)
    f_base, c_base = net.solve_base_equilibrium()

    # Agent transition matrix P (agent moves Down and Right towards goal (m, n))
    num_n = net.num_nodes
    P = np.zeros((num_n, num_n))
    for u in range(num_n):
        ui, uj = net.idx_to_node[u]
        succs = []
        if ui < m:
            succs.append(net.node_to_idx[(ui + 1, uj)])
        if uj < n:
            succs.append(net.node_to_idx[(ui, uj + 1)])
        if succs:
            for s in succs:
                P[u, s] = 1.0 / len(succs)
        else:
            P[u, u] = 1.0

    # Green's resolvent operator
    G_out = np.linalg.inv(np.eye(num_n) - P + 0.1 * np.eye(num_n))

    # Latency vector on all edges
    edge_lat = {}
    opposing_traffic_on_forward_edges = np.zeros(num_n)
    for e_idx, (u, v) in enumerate(net.edges):
        flow = f_base[e_idx]
        lat = net.a * flow + net.b
        edge_lat[(u, v)] = lat

    # When counter-traffic runs North-West on two-way streets, forward movement on those streets suffers congestion
    for (u, v), flow in zip(net.edges, f_base):
        if flow > 0.05:
            ui, uj = net.idx_to_node[u]
            vi, vj = net.idx_to_node[v]
            # Counter-flow edge (Up or Left) adds congestion to the reverse direction (Down or Right)
            if vi < ui or vj < uj:  # Edge was moving Up or Left
                opposing_traffic_on_forward_edges[u] += flow
                opposing_traffic_on_forward_edges[v] += flow

    # Congestion divergence: discrepancy of counter-flow
    B_div = np.abs(net.B @ (net.a * f_base))
    Phi_spec = G_out @ B_div

    # Effective edge costs for forward moving agent
    def get_forward_edge_cost(u_node, v_node):
        u_idx = net.node_to_idx[u_node]
        v_idx = net.node_to_idx[v_node]
        # Congestion caused by opposing counter-traffic meeting head-on
        head_on_flow = opposing_traffic_on_forward_edges[v_idx]
        return 1.0 + 2.5 * head_on_flow

    # Natural tie-breaking parameter
    ratio = float(n) / float(m)

    # 1. Standard-PS Rollout (Myopic Greedy: chooses direct diagonal path, unaware of oncoming head-on storm)
    s_std = (0, 0)
    traj_std = [s_std]
    lat_std = [0.0]
    cum_std = [0.0]

    # Standard-PS prefers diagonal heading (j approx ratio * i) because it does not anticipate oncoming traffic
    while s_std != (m, n) and len(traj_std) < 35:
        i, j = s_std
        cands = []
        if i < m:
            nxt = (i + 1, j)
            c = get_forward_edge_cost(s_std, nxt)
            dist = abs(nxt[1] - ratio * nxt[0])
            cands.append((dist, nxt, c))
        if j < n:
            nxt = (i, j + 1)
            c = get_forward_edge_cost(s_std, nxt)
            dist = abs(nxt[1] - ratio * nxt[0])
            cands.append((dist, nxt, c))
        cands.sort(key=lambda x: x[0])
        s_std = cands[0][1]
        cost = cands[0][2]
        traj_std.append(s_std)
        lat_std.append(cost)
        cum_std.append(cum_std[-1] + cost)

    # 2. SL-PS Rollout (Spectral Resolvent Phi_spec builds a high-risk wall in the counter-flow zone)
    # SL-PS solves Bellman equation penalized by beta * Phi_spec
    V_sl = {net.node_to_idx[(i, j)]: float((m - i) + (n - j)) for i in range(m + 1) for j in range(n + 1)}
    for _ in range(60):
        for i in range(m, -1, -1):
            for j in range(n, -1, -1):
                if (i, j) == (m, n):
                    V_sl[net.node_to_idx[(i, j)]] = 0.0
                    continue
                q_list = []
                if i < m:
                    nxt = (i + 1, j)
                    c = get_forward_edge_cost((i, j), nxt)
                    pen = float(beta * Phi_spec[net.node_to_idx[nxt]])
                    q_list.append(c + 0.95 * V_sl[net.node_to_idx[nxt]] + pen)
                if j < n:
                    nxt = (i, j + 1)
                    c = get_forward_edge_cost((i, j), nxt)
                    pen = float(beta * Phi_spec[net.node_to_idx[nxt]])
                    q_list.append(c + 0.95 * V_sl[net.node_to_idx[nxt]] + pen)
                V_sl[net.node_to_idx[(i, j)]] = min(q_list)

    # SL-PS path follows policy: avoids the high-risk dome by lateral flanking
    s_sl = (0, 0)
    traj_sl = [s_sl]
    lat_sl = [0.0]
    cum_sl = [0.0]
    while s_sl != (m, n) and len(traj_sl) < 35:
        i, j = s_sl
        cands = []
        if i < m:
            nxt = (i + 1, j)
            c = get_forward_edge_cost(s_sl, nxt)
            pen = float(beta * Phi_spec[net.node_to_idx[nxt]])
            q_val = c + 0.95 * V_sl[net.node_to_idx[nxt]] + pen
            dist = abs(nxt[1] - ratio * nxt[0])
            cands.append((q_val, dist, nxt, c))
        if j < n:
            nxt = (i, j + 1)
            c = get_forward_edge_cost(s_sl, nxt)
            pen = float(beta * Phi_spec[net.node_to_idx[nxt]])
            q_val = c + 0.95 * V_sl[net.node_to_idx[nxt]] + pen
            dist = abs(nxt[1] - ratio * nxt[0])
            cands.append((q_val, dist, nxt, c))
        cands.sort(key=lambda x: (round(x[0], 2), x[1]))
        s_sl = cands[0][2]
        cost = cands[0][3]
        traj_sl.append(s_sl)
        lat_sl.append(cost)
        cum_sl.append(cum_sl[-1] + cost)

    return (
        net,
        f_base,
        Phi_spec,
        opposing_traffic_on_forward_edges,
        traj_std,
        traj_sl,
        lat_std,
        lat_sl,
        cum_std,
        cum_sl
    )


def render_counterflow_animation(
    net,
    f_base,
    Phi_spec,
    opposing_traffic,
    traj_std,
    traj_sl,
    lat_std,
    lat_sl,
    cum_std,
    cum_sl,
    save_path="sl_ps_counterflow_evasion.gif",
    show=False
):
    m, n = net.m, net.n
    max_steps = max(len(traj_std), len(traj_sl))
    total_frames = max_steps + 4

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(19, 10.5), dpi=120)
    fig.patch.set_facecolor("#0A0F1D")

    gs = fig.add_gridspec(2, 2, height_ratios=[3.4, 1.4], hspace=0.25, wspace=0.12,
                           left=0.04, right=0.96, top=0.91, bottom=0.06)

    ax_map_std = fig.add_subplot(gs[0, 0])
    ax_map_sl = fig.add_subplot(gs[0, 1])
    ax_plot_step = fig.add_subplot(gs[1, 0])
    ax_plot_cum = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        "Counter-Directional Traffic Navigation: Standard-PS vs. Spectral-Laplacian PS (SL-PS)\n"
        f"Agent Moves South-East (0,0)->({m},{n}) | Background Traffic Rushes North-West (Opposing): {net.s_bg}->{net.t_bg}",
        fontsize=13.5, fontweight="bold", color="#F8FAFC", y=0.97
    )

    for ax in (ax_map_std, ax_map_sl):
        ax.set_facecolor("#111827")
        ax.set_xlim(-0.8, n + 0.8)
        ax.set_ylim(m + 0.8, -0.8)
        ax.set_aspect("equal")
        ax.set_xticks(range(0, n + 1, 2))
        ax.set_yticks(range(0, m + 1, 2))
        ax.tick_params(colors="#94A3B8", labelsize=7.5)
        for spine in ax.spines.values():
            spine.set_color("#374151")

    for ax in (ax_plot_step, ax_plot_cum):
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#94A3B8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#374151")

    ax_map_std.set_title("Standard Prioritized Sweeping (Standard-PS)\nBlind to Upstream Opposing Traffic -> Collides Head-On with Counter-Flow", fontsize=10, fontweight="bold", color="#F87171", pad=6)
    ax_map_sl.set_title("Spectral-Laplacian PS (SL-PS)\nResolvent Detects Opposing Shockwave -> Lateral Flanking Evasion (Tat Suon)", fontsize=10, fontweight="bold", color="#38BDF8", pad=6)

    ax_plot_step.set_title("Step Latency Under Head-On Opposing Wave l(s_t, a_t)", fontsize=9, fontweight="bold", color="#E2E8F0", pad=5)
    ax_plot_step.set_xlabel("Navigation Step (t)", color="#94A3B8", fontsize=8)
    ax_plot_step.set_ylabel("Step Latency (s)", color="#94A3B8", fontsize=8)
    ax_plot_step.set_xlim(0, max_steps + 1)
    ax_plot_step.set_ylim(0, max(max(lat_std), max(lat_sl)) * 1.3 + 1.0)

    ax_plot_cum.set_title("Cumulative Travel Time & Flanking Advantage T_cum(t)", fontsize=9, fontweight="bold", color="#E2E8F0", pad=5)
    ax_plot_cum.set_xlabel("Navigation Step (t)", color="#94A3B8", fontsize=8)
    ax_plot_cum.set_ylabel("Cumulative Latency (s)", color="#94A3B8", fontsize=8)
    ax_plot_cum.set_xlim(0, max_steps + 1)
    ax_plot_cum.set_ylim(0, max(cum_std[-1], cum_sl[-1]) * 1.15 + 2.0)

    # Draw static base elements
    # Phi_spec heatmap on SL-PS map
    phi_norm = Normalize(vmin=0.0, vmax=max(Phi_spec) * 0.9 + 1e-4)
    phi_cmap = plt.cm.YlOrRd

    for i in range(m + 1):
        for j in range(n + 1):
            idx = net.node_to_idx[(i, j)]
            val = Phi_spec[idx]
            if val > 0.08:
                c = phi_cmap(phi_norm(val))
                circ = Circle((j, i), 0.35, facecolor=c, edgecolor="none", alpha=0.35, zorder=1)
                ax_map_sl.add_patch(circ)

    # Draw counter-flow streams (yellow arrows moving Up & Left)
    for e_idx, (u, v) in enumerate(net.edges):
        flow = f_base[e_idx]
        if flow > 0.1:
            ui, uj = net.idx_to_node[u]
            vi, vj = net.idx_to_node[v]
            # Counter-flow
            if vi < ui or vj < uj:
                for ax in (ax_map_std, ax_map_sl):
                    arr = FancyArrowPatch(
                        (uj, ui), (vj, vi), arrowstyle="-|>", mutation_scale=8,
                        color="#FACC15", lw=1.2 + 2.5 * (flow / 2.0), alpha=0.85, zorder=3
                    )
                    ax.add_patch(arr)

    # Landmark markers
    for ax in (ax_map_std, ax_map_sl):
        # Start & Goal
        ax.plot(0, 0, marker="*", markersize=13, color="#10B981", markeredgecolor="#FFFFFF", zorder=8, label="Start (0,0)")
        ax.plot(n, m, marker="X", markersize=11, color="#8B5CF6", markeredgecolor="#FFFFFF", zorder=8, label=f"Goal ({m},{n})")
        # Counter-flow Source (Bottom-Right) & Sink (Top-Left)
        ax.plot(net.s_bg[1], net.s_bg[0], marker="D", markersize=8, color="#EAB308", markeredgecolor="#FFFFFF", zorder=7, label=f"Counter-Src {net.s_bg}")
        ax.plot(net.t_bg[1], net.t_bg[0], marker="s", markersize=8, color="#EAB308", markeredgecolor="#FFFFFF", zorder=7, label=f"Counter-Sink {net.t_bg}")
        ax.legend(loc="lower left", fontsize=7.5, facecolor="#1F2937", edgecolor="#374151", labelcolor="#CBD5E1")

    # Dynamic plot elements
    trail_std, = ax_map_std.plot([], [], color="#F97316", lw=2.8, alpha=0.9, zorder=9)
    dot_std, = ax_map_std.plot([], [], marker="o", markersize=10, color="#EF4444", markeredgecolor="#FFFFFF", markeredgewidth=1.8, zorder=10)

    trail_sl, = ax_map_sl.plot([], [], color="#38BDF8", lw=2.8, alpha=0.9, zorder=9)
    dot_sl, = ax_map_sl.plot([], [], marker="o", markersize=10, color="#0284C7", markeredgecolor="#FFFFFF", markeredgewidth=1.8, zorder=10)

    line_step_std, = ax_plot_step.plot([], [], color="#EF4444", lw=2.2, marker="o", markersize=4, label="Standard-PS (Head-On Collision)")
    line_step_sl, = ax_plot_step.plot([], [], color="#0EA5E9", lw=2.2, marker="s", markersize=4, label="SL-PS (Lateral Flanking)")
    ax_plot_step.legend(loc="upper right", fontsize=7.5, facecolor="#1F2937", edgecolor="#374151", labelcolor="#E5E7EB")

    line_cum_std, = ax_plot_cum.plot([], [], color="#EF4444", lw=2.5, label="Standard-PS Cumulative")
    line_cum_sl, = ax_plot_cum.plot([], [], color="#0EA5E9", lw=2.5, label="SL-PS Cumulative")
    ax_plot_cum.legend(loc="upper left", fontsize=7.5, facecolor="#1F2937", edgecolor="#374151", labelcolor="#E5E7EB")

    badge_std = ax_map_std.text(0.02, 0.05, "", transform=ax_map_std.transAxes, fontsize=8, fontweight="bold",
                                color="#FCA5A5", bbox=dict(boxstyle="round,pad=0.25", facecolor="#1F2937", edgecolor="#EF4444", alpha=0.9), zorder=11)
    badge_sl = ax_map_sl.text(0.02, 0.05, "", transform=ax_map_sl.transAxes, fontsize=8, fontweight="bold",
                              color="#7DD3FC", bbox=dict(boxstyle="round,pad=0.25", facecolor="#1F2937", edgecolor="#0EA5E9", alpha=0.9), zorder=11)

    banner_text = fig.text(
        0.5, 0.935, "Yellow Arrows: Opposing Traffic Moving North-West <- | Agent Travels South-East ->",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#FDE047",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#1E293B", edgecolor="#F59E0B", lw=1.0)
    )

    def animate(frame):
        cur_step = min(frame, max_steps)
        idx_std = min(cur_step, len(traj_std) - 1)
        idx_sl = min(cur_step, len(traj_sl) - 1)

        # Update maps
        trail_std.set_data([p[1] for p in traj_std[:idx_std + 1]], [p[0] for p in traj_std[:idx_std + 1]])
        dot_std.set_data([traj_std[idx_std][1]], [traj_std[idx_std][0]])

        trail_sl.set_data([p[1] for p in traj_sl[:idx_sl + 1]], [p[0] for p in traj_sl[:idx_sl + 1]])
        dot_sl.set_data([traj_sl[idx_sl][1]], [traj_sl[idx_sl][0]])

        # Update badges
        in_std_collision = opposing_traffic[net.node_to_idx[traj_std[idx_std]]] > 0.1
        std_status = "!! HEAD-ON COLLISION !!" if in_std_collision else "Approaching Traffic"
        badge_std.set_text(
            f"Standard-PS @ {traj_std[idx_std]} [{std_status}]\n"
            f"Step Latency: {lat_std[idx_std]:.2f} s\n"
            f"Cumulative:   {cum_std[idx_std]:.2f} s"
        )

        in_sl_flank = traj_sl[idx_sl][0] >= 6 or traj_sl[idx_sl][1] <= 3
        sl_status = "Lateral Flank (Safe Bypass)" if in_sl_flank else "Cruising"
        badge_sl.set_text(
            f"SL-PS @ {traj_sl[idx_sl]} [{sl_status}]\n"
            f"Step Latency: {lat_sl[idx_sl]:.2f} s\n"
            f"Cumulative:   {cum_sl[idx_sl]:.2f} s"
        )

        # Update plots
        x_std = list(range(idx_std + 1))
        x_sl = list(range(idx_sl + 1))
        line_step_std.set_data(x_std, lat_std[:idx_std + 1])
        line_step_sl.set_data(x_sl, lat_sl[:idx_sl + 1])

        line_cum_std.set_data(x_std, cum_std[:idx_std + 1])
        line_cum_sl.set_data(x_sl, cum_sl[:idx_sl + 1])

        return trail_std, dot_std, trail_sl, dot_sl, line_step_std, line_step_sl, line_cum_std, line_cum_sl

    ani = animation.FuncAnimation(fig, animate, frames=total_frames, interval=850, repeat=True, blit=False)

    print(f"Rendering counter-flow evasion animation to {save_path}...")
    ani.save(save_path, writer="pillow", fps=1.2)
    print(f"Successfully rendered and saved {save_path}!")

    if show:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Counter-Directional Flow Simulation")
    parser.add_argument("--save", type=str, default="sl_ps_counterflow_evasion.gif", help="Output GIF path")
    parser.add_argument("--show", action="store_true", help="Interactive live display")
    args = parser.parse_args()

    print("Simulating Counter-Directional Flow Navigation...")
    (
        net,
        f_base,
        Phi_spec,
        opposing_traffic,
        traj_std,
        traj_sl,
        lat_std,
        lat_sl,
        cum_std,
        cum_sl
    ) = simulate_counterflow_navigation()

    print("Simulation finished:")
    print(f"  Standard-PS Path: {len(traj_std)-1} steps, Latency = {cum_std[-1]:.2f} s")
    print(f"  SL-PS Path:       {len(traj_sl)-1} steps, Latency = {cum_sl[-1]:.2f} s")
    print(f"  SL-PS Advantage:  {cum_std[-1] - cum_sl[-1]:.2f} s ({(1 - cum_sl[-1]/cum_std[-1])*100:.1f}% faster!)")

    render_counterflow_animation(
        net,
        f_base,
        Phi_spec,
        opposing_traffic,
        traj_std,
        traj_sl,
        lat_std,
        lat_sl,
        cum_std,
        cum_sl,
        save_path=args.save,
        show=args.show
    )


if __name__ == "__main__":
    main()
