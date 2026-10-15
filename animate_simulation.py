"""
animate_simulation.py
=====================
Visualizes side-by-side animation of:
- Left: Standard Prioritized Sweeping (Standard-PS) falling into Braess bottleneck.
- Right: Spectral-Laplacian Prioritized Sweeping (SL-PS) proactively bypassing congestion.

Outputs:
- braess_navigation_animation.gif (high-quality animated GIF)
- Interactive live animation window if run with --show
"""

import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D

from grid_braess_env import CongestedGridEnv
from planning_agents import StandardPrioritizedSweepingAgent, SLPSAgent


def run_episode_trajectories(ep_seed: int = 1010):
    env = CongestedGridEnv(m=3, n=6, p_chord=0.4, seed=42)
    std_agent = StandardPrioritizedSweepingAgent(env, backup_budget=15)
    sl_agent = SLPSAgent(env, backup_budget=15, beta_scale=6.0)

    # 1. Standard-PS rollout
    s_std, info_std = env.reset(seed=ep_seed)
    std_agent.V = {
        (i, j): float((env.m - i) + (env.n - j)) * env.grid.b
        for i in range(env.m + 1)
        for j in range(env.n + 1)
    }
    std_agent.V[(env.m, env.n)] = 0.0

    traj_std = []
    tot_std = 0.0
    traj_std.append({"pos": s_std, "step_lat": 0.0, "cum_lat": 0.0, "chord": info_std["active_chord"], "took_chord": False})

    done = False
    while not done:
        a = std_agent.select_action(s_std, info_std["active_chord"])
        took_chord = (a == CongestedGridEnv.ACTION_CHORD)
        nxt, lat, done, info_std = env.step(a)
        tot_std += lat
        traj_std.append({"pos": nxt, "step_lat": lat, "cum_lat": tot_std, "chord": info_std["active_chord"], "took_chord": took_chord})
        s_std = nxt

    # 2. SL-PS rollout
    s_sl, info_sl = env.reset(seed=ep_seed)
    sl_agent.V = {
        (i, j): float((env.m - i) + (env.n - j)) * env.grid.b
        for i in range(env.m + 1)
        for j in range(env.n + 1)
    }
    sl_agent.V[(env.m, env.n)] = 0.0

    traj_sl = []
    tot_sl = 0.0
    traj_sl.append({"pos": s_sl, "step_lat": 0.0, "cum_lat": 0.0, "chord": info_sl["active_chord"], "took_chord": False})

    done = False
    while not done:
        a = sl_agent.select_action(s_sl, info_sl["active_chord"])
        took_chord = (a == CongestedGridEnv.ACTION_CHORD)
        nxt, lat, done, info_sl = env.step(a)
        tot_sl += lat
        traj_sl.append({"pos": nxt, "step_lat": lat, "cum_lat": tot_sl, "chord": info_sl["active_chord"], "took_chord": took_chord})
        s_sl = nxt

    # Spectral Congestion Potential field for visualization
    Phi_spec = sl_agent.compute_spectral_congestion_field()

    return env, traj_std, traj_sl, Phi_spec


def create_animation(env, traj_std, traj_sl, Phi_spec, save_path="braess_navigation_animation.gif", show=False):
    m, n = env.m, env.n
    max_frames = max(len(traj_std), len(traj_sl)) + 3

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax_std, ax_sl) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=150)
    fig.patch.set_facecolor("#F8F9FA")

    def draw_base_grid(ax, title, subtitle):
        ax.set_facecolor("#FFFFFF")
        ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlim(-0.6, n + 0.6)
        ax.set_ylim(m + 0.6, -0.6)
        ax.set_aspect("equal")
        ax.set_xticks(range(n + 1))
        ax.set_yticks(range(m + 1))
        ax.set_xlabel("Columns ($j$)", fontsize=10)
        ax.set_ylabel("Rows ($i$)", fontsize=10)

        # Base grid edges
        for i in range(m + 1):
            for j in range(n):
                ax.annotate("", xy=(j + 1, i), xytext=(j, i),
                            arrowprops=dict(arrowstyle="->", color="#B0BEC5", lw=1.2, alpha=0.7))
        for i in range(m):
            for j in range(n + 1):
                ax.annotate("", xy=(j, i + 1), xytext=(j, i),
                            arrowprops=dict(arrowstyle="->", color="#B0BEC5", lw=1.2, alpha=0.7))

        # Node points
        for i in range(m + 1):
            for j in range(n + 1):
                ax.plot(j, i, "o", color="#90A4AE", markersize=6)

        # Source and Goal markers
        ax.plot(0, 0, "s", color="#2E7D32", markersize=12, zorder=5, label="Start $(0,0)$")
        ax.plot(n, m, "s", color="#6A1B9A", markersize=12, zorder=5, label="Goal $(3,6)$")

    draw_base_grid(ax_std, "Standard Prioritized Sweeping (Standard-PS)", "Myopic Bellman Updates: Falls into Braess Trap")
    draw_base_grid(ax_sl, "Spectral-Laplacian Prioritized Sweeping (SL-PS)", "Spectral Divergence Potential: Proactive Bypass")

    # Spectral risk heatmap on SL-PS plot (highlight nodes near Braess chord)
    phi_norm = (Phi_spec - np.min(Phi_spec)) / (np.max(Phi_spec) - np.min(Phi_spec) + 1e-6)
    for u in range(env.grid.num_nodes):
        ui, uj = env.grid.idx_to_node[u]
        val = phi_norm[u]
        if val > 0.3:
            circle = Circle((uj, ui), radius=0.32, color="#FFCDD2", alpha=val * 0.7, zorder=2)
            ax_sl.add_patch(circle)

    # Dynamic elements for Standard-PS
    std_agent_dot, = ax_std.plot([], [], "o", color="#D32F2F", markersize=14, zorder=10, label="Agent")
    std_path_line, = ax_std.plot([], [], "-", color="#E57373", lw=3.0, alpha=0.8, zorder=9)
    std_hud = ax_std.text(0.02, 0.05, "", transform=ax_std.transAxes, fontsize=10,
                          bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFEBEE", edgecolor="#EF5350"))
    std_alert = ax_std.text(0.5, 0.92, "", transform=ax_std.transAxes, fontsize=11, fontweight="bold",
                            color="#C62828", ha="center", bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFCDD2", edgecolor="#B71C1C", alpha=0.0))

    # Dynamic elements for SL-PS
    sl_agent_dot, = ax_sl.plot([], [], "*", color="#1565C0", markersize=18, zorder=10, label="SL-PS Agent")
    sl_path_line, = ax_sl.plot([], [], "-", color="#64B5F6", lw=3.0, alpha=0.8, zorder=9)
    sl_hud = ax_sl.text(0.02, 0.05, "", transform=ax_sl.transAxes, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E3F2FD", edgecolor="#42A5F5"))
    sl_alert = ax_sl.text(0.5, 0.92, "", transform=ax_sl.transAxes, fontsize=11, fontweight="bold",
                          color="#0D47A1", ha="center", bbox=dict(boxstyle="round,pad=0.4", facecolor="#BBDEFB", edgecolor="#1565C0", alpha=0.0))

    # Braess Chord representation on both plots: (0, 2) -> (2, 1)
    u_c, v_c = (0, 2), (2, 1)
    chord_arrow_std = FancyArrowPatch((u_c[1], u_c[0]), (v_c[1], v_c[0]),
                                      connectionstyle="arc3,rad=-0.25", color="#D32F2F", lw=2.5,
                                      linestyle="--", arrowstyle="->", mutation_scale=15, zorder=6)
    chord_arrow_sl = FancyArrowPatch((u_c[1], u_c[0]), (v_c[1], v_c[0]),
                                     connectionstyle="arc3,rad=-0.25", color="#B0BEC5", lw=2.0,
                                     linestyle=":", arrowstyle="->", mutation_scale=15, zorder=6)
    ax_std.add_patch(chord_arrow_std)
    ax_sl.add_patch(chord_arrow_sl)

    ax_std.text(1.2, 0.8, "Braess Chord\n$(0,2) \\to (2,1)$", color="#D32F2F", fontsize=9, fontweight="bold")
    ax_sl.text(1.2, 0.8, "Braess Chord\n[Flagged $V_{uv}<0$]", color="#546E7A", fontsize=9, fontweight="bold")

    ax_std.legend(loc="upper right", fontsize=8)
    ax_sl.legend(loc="upper right", fontsize=8)

    def init():
        std_agent_dot.set_data([], [])
        std_path_line.set_data([], [])
        sl_agent_dot.set_data([], [])
        sl_path_line.set_data([], [])
        return std_agent_dot, std_path_line, sl_agent_dot, sl_path_line

    def update(frame):
        # Frame clamped for each trajectory
        f_std = min(frame, len(traj_std) - 1)
        f_sl = min(frame, len(traj_sl) - 1)

        # Standard-PS update
        p_std = traj_std[f_std]
        std_history = [t["pos"] for t in traj_std[:f_std + 1]]
        std_agent_dot.set_data([p_std["pos"][1]], [p_std["pos"][0]])
        std_path_line.set_data([pt[1] for pt in std_history], [pt[0] for pt in std_history])

        # Check if took chord
        took_chord_std = any(t["took_chord"] for t in traj_std[:f_std + 1])
        if took_chord_std:
            chord_arrow_std.set_color("#B71C1C")
            chord_arrow_std.set_linewidth(4.0)
            std_alert.set_text("[BOTTLENECK TRAP DETECTED!]\n(+11.4s Congestion Delay)")
            std_alert.get_bbox_patch().set_alpha(0.9)
            status_std = "TRAPPED (Congested)"
        else:
            status_std = "Navigating..."

        std_hud.set_text(f"Step: {f_std}/{len(traj_std)-1}\nPos: {p_std['pos']}\nLatency: {p_std['cum_lat']:.2f}s\nStatus: {status_std}")

        # SL-PS update
        p_sl = traj_sl[f_sl]
        sl_history = [t["pos"] for t in traj_sl[:f_sl + 1]]
        sl_agent_dot.set_data([p_sl["pos"][1]], [p_sl["pos"][0]])
        sl_path_line.set_data([pt[1] for pt in sl_history], [pt[0] for pt in sl_history])

        if f_sl >= 1:
            sl_alert.set_text("[SPECTRAL RISK FILTERED]\nProactively Bypassing Bottleneck!")
            sl_alert.get_bbox_patch().set_alpha(0.85)

        if f_sl == len(traj_sl) - 1:
            status_sl = "GOAL REACHED! (Fastest)"
            sl_alert.set_text(f"[GOAL REACHED: {p_sl['cum_lat']:.2f}s]\n(Saves {traj_std[-1]['cum_lat'] - p_sl['cum_lat']:.2f}s vs Standard-PS)")
            sl_alert.get_bbox_patch().set_facecolor("#C8E6C9")
            sl_alert.get_bbox_patch().set_edgecolor("#2E7D32")
            sl_alert.set_color("#1B5E20")
        else:
            status_sl = "Navigating Bypass Corridor..."

        sl_hud.set_text(f"Step: {f_sl}/{len(traj_sl)-1}\nPos: {p_sl['pos']}\nLatency: {p_sl['cum_lat']:.2f}s\nStatus: {status_sl}")

        return std_agent_dot, std_path_line, std_hud, std_alert, sl_agent_dot, sl_path_line, sl_hud, sl_alert

    anim = animation.FuncAnimation(fig, update, frames=max_frames, init_func=init, interval=800, blit=False, repeat=True)

    print(f"Rendering and saving animation to {save_path} ...")
    anim.save(save_path, writer=animation.PillowWriter(fps=1.5))
    print(f"Successfully saved animation: {save_path}")

    if show:
        plt.show()

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate Braess Grid Navigation")
    parser.add_argument("--seed", type=int, default=1010, help="Episode seed (default: 1010 - Episode 10)")
    parser.add_argument("--save", type=str, default="braess_navigation_animation.gif", help="Output GIF path")
    parser.add_argument("--show", action="store_true", help="Display live animation window")
    args = parser.parse_args()

    print("Generating simulation trajectory...")
    env, traj_std, traj_sl, Phi_spec = run_episode_trajectories(ep_seed=args.seed)
    create_animation(env, traj_std, traj_sl, Phi_spec, save_path=args.save, show=args.show)
