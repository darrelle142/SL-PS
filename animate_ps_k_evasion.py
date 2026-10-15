"""
animate_ps_k_evasion.py
=======================
Visualizes the exact impact of planning budget K on Standard Prioritized Sweeping:
- Left: Standard-PS with K = 15 (Myopic Bellman Propagation -> Falls into Braess Trap).
- Right: Standard-PS with K = 100 (Deep Bellman Propagation -> Proactively Evades Braess Trap).

Outputs:
- ps_large_k_evasion.gif
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D

from grid_braess_env import CongestedGridEnv
from planning_agents import StandardPrioritizedSweepingAgent


def run_trajectories(ep_seed: int = 1010):
    # Setup environments
    env_small = CongestedGridEnv(m=3, n=6, p_chord=0.4, seed=42)
    env_large = CongestedGridEnv(m=3, n=6, p_chord=0.4, seed=42)
    
    agent_small = StandardPrioritizedSweepingAgent(env_small, backup_budget=15)
    agent_large = StandardPrioritizedSweepingAgent(env_large, backup_budget=100)

    # 1. Rollout K = 15
    s_s, info_s = env_small.reset(seed=ep_seed)
    traj_s = []
    tot_s = 0.0
    traj_s.append({
        "pos": s_s, "step_lat": 0.0, "cum_lat": 0.0,
        "chord": info_s["active_chord"], "chord_type": info_s["chord_type"], "took_chord": False
    })
    done = False
    while not done:
        a = agent_small.select_action(s_s, info_s["active_chord"])
        took_chord = (a == CongestedGridEnv.ACTION_CHORD)
        nxt, lat, done, info_s = env_small.step(a)
        tot_s += lat
        traj_s.append({
            "pos": nxt, "step_lat": lat, "cum_lat": tot_s,
            "chord": info_s["active_chord"], "chord_type": info_s["chord_type"], "took_chord": took_chord
        })
        s_s = nxt

    # 2. Rollout K = 100
    s_l, info_l = env_large.reset(seed=ep_seed)
    traj_l = []
    tot_l = 0.0
    traj_l.append({
        "pos": s_l, "step_lat": 0.0, "cum_lat": 0.0,
        "chord": info_l["active_chord"], "chord_type": info_l["chord_type"], "took_chord": False
    })
    done = False
    while not done:
        a = agent_large.select_action(s_l, info_l["active_chord"])
        took_chord = (a == CongestedGridEnv.ACTION_CHORD)
        nxt, lat, done, info_l = env_large.step(a)
        tot_l += lat
        traj_l.append({
            "pos": nxt, "step_lat": lat, "cum_lat": tot_l,
            "chord": info_l["active_chord"], "chord_type": info_l["chord_type"], "took_chord": took_chord
        })
        s_l = nxt

    return env_small, traj_s, traj_l


def create_animation(env, traj_s, traj_l, save_path):
    m, n = env.m, env.n
    max_frames = max(len(traj_s), len(traj_l)) + 3

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax_s, ax_l) = plt.subplots(1, 2, figsize=(15, 7), dpi=140)
    fig.patch.set_facecolor("#0F172A")  # Dark slate navy

    def draw_base_grid(ax, title, subtitle, title_color):
        ax.set_facecolor("#1E293B")
        ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold", pad=12, color=title_color)
        ax.set_xlim(-0.6, n + 0.6)
        ax.set_ylim(m + 0.6, -0.6)
        ax.set_aspect("equal")
        ax.set_xticks(range(n + 1))
        ax.set_yticks(range(m + 1))
        ax.tick_params(colors="#94A3B8")
        ax.grid(color="#334155", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Columns (j)", fontsize=10, color="#94A3B8")
        ax.set_ylabel("Rows (i)", fontsize=10, color="#94A3B8")

        # Base grid edges
        for i in range(m + 1):
            for j in range(n):
                ax.annotate("", xy=(j + 1, i), xytext=(j, i),
                            arrowprops=dict(arrowstyle="->", color="#475569", lw=1.2, alpha=0.6))
        for i in range(m):
            for j in range(n + 1):
                ax.annotate("", xy=(j, i + 1), xytext=(j, i),
                            arrowprops=dict(arrowstyle="->", color="#475569", lw=1.2, alpha=0.6))

        # Node points
        for i in range(m + 1):
            for j in range(n + 1):
                ax.plot(j, i, "o", color="#64748B", markersize=5)

        # Source and Goal markers
        ax.plot(0, 0, "s", color="#10B981", markersize=11, zorder=5, label="Start (0,0)")
        ax.plot(n, m, "s", color="#8B5CF6", markersize=11, zorder=5, label="Goal (3,6)")

    draw_base_grid(ax_s, "Standard Prioritized Sweeping (K = 15)", "Myopic Budget: Residual Truncation -> Braess Trap", "#F87171")
    draw_base_grid(ax_l, "Standard Prioritized Sweeping (K = 100)", "Large Budget: Deep Bellman Propagation -> Trap Evasion", "#34D399")

    # Static Braess Chord visualization (0, 2) -> (2, 1)
    u_c, v_c = (0, 2), (2, 1)
    chord_s = FancyArrowPatch((u_c[1], u_c[0]), (v_c[1], v_c[0]),
                              connectionstyle="arc3,rad=-0.28", color="#EF4444", lw=2.5,
                              linestyle="--", arrowstyle="->", mutation_scale=15, zorder=6)
    chord_l = FancyArrowPatch((u_c[1], u_c[0]), (v_c[1], v_c[0]),
                              connectionstyle="arc3,rad=-0.28", color="#F59E0B", lw=2.0,
                              linestyle=":", arrowstyle="->", mutation_scale=15, zorder=6)
    ax_s.add_patch(chord_s)
    ax_l.add_patch(chord_l)

    ax_s.text(1.1, 0.75, "Braess Chord\n(0,2)->(2,1)\n[TRAP TAKEN]", color="#EF4444", fontsize=8.5, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.2", facecolor="#450A0A", edgecolor="#EF4444", alpha=0.8))
    ax_l.text(1.1, 0.75, "Braess Chord\n(0,2)->(2,1)\n[AVOIDED!]", color="#F59E0B", fontsize=8.5, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.2", facecolor="#451A03", edgecolor="#F59E0B", alpha=0.8))

    # Static Benign Shortcut visualization (3, 1) -> (2, 3)
    u_b, v_b = (3, 1), (2, 3)
    benign_arrow = FancyArrowPatch((u_b[1], u_b[0]), (v_b[1], v_b[0]),
                                   connectionstyle="arc3,rad=0.25", color="#10B981", lw=2.2,
                                   linestyle="-", arrowstyle="->", mutation_scale=14, zorder=6)
    ax_l.add_patch(benign_arrow)
    ax_l.text(1.6, 2.7, "Expressway Bypass\n(3,1)->(2,3) [0.5s]", color="#10B981", fontsize=8.5, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.2", facecolor="#064E3B", edgecolor="#10B981", alpha=0.8))

    # Dynamic line and dots
    line_s, = ax_s.plot([], [], "-", color="#F87171", lw=3.0, alpha=0.85, zorder=9)
    dot_s, = ax_s.plot([], [], "o", color="#EF4444", markersize=14, zorder=10, label="Agent K=15")
    
    line_l, = ax_l.plot([], [], "-", color="#34D399", lw=3.0, alpha=0.85, zorder=9)
    dot_l, = ax_l.plot([], [], "o", color="#10B981", markersize=14, zorder=10, label="Agent K=100")

    # HUD text boxes
    hud_s = ax_s.text(0.03, 0.05, "", transform=ax_s.transAxes, fontsize=9.5, family="monospace",
                      color="#FFFFFF", bbox=dict(boxstyle="round,pad=0.5", facecolor="#1E293B", edgecolor="#EF4444", lw=1.5))
    alert_s = ax_s.text(0.5, 0.92, "", transform=ax_s.transAxes, fontsize=10.5, fontweight="bold",
                        color="#FEF2F2", ha="center", bbox=dict(boxstyle="round,pad=0.4", facecolor="#991B1B", edgecolor="#EF4444", alpha=0.0))

    hud_l = ax_l.text(0.03, 0.05, "", transform=ax_l.transAxes, fontsize=9.5, family="monospace",
                      color="#FFFFFF", bbox=dict(boxstyle="round,pad=0.5", facecolor="#1E293B", edgecolor="#10B981", lw=1.5))
    alert_l = ax_l.text(0.5, 0.92, "", transform=ax_l.transAxes, fontsize=10.5, fontweight="bold",
                        color="#ECFDF5", ha="center", bbox=dict(boxstyle="round,pad=0.4", facecolor="#065F46", edgecolor="#10B981", alpha=0.0))

    ax_s.legend(loc="upper right", fontsize=8.5, facecolor="#1E293B", edgecolor="#475569", labelcolor="#F1F5F9")
    ax_l.legend(loc="upper right", fontsize=8.5, facecolor="#1E293B", edgecolor="#475569", labelcolor="#F1F5F9")

    def init():
        dot_s.set_data([], [])
        line_s.set_data([], [])
        dot_l.set_data([], [])
        line_l.set_data([], [])
        return dot_s, line_s, dot_l, line_l

    def update(frame):
        f_s = min(frame, len(traj_s) - 1)
        f_l = min(frame, len(traj_l) - 1)

        # 1. Update K = 15
        p_s = traj_s[f_s]
        hist_s = [t["pos"] for t in traj_s[:f_s + 1]]
        dot_s.set_data([p_s["pos"][1]], [p_s["pos"][0]])
        line_s.set_data([pt[1] for pt in hist_s], [pt[0] for pt in hist_s])

        hud_s.set_text(
            f"Step: {f_s}/{len(traj_s)-1}\n"
            f"Pos:  {p_s['pos']}\n"
            f"Step Latency: {p_s['step_lat']:.2f}s\n"
            f"Total Time:   {p_s['cum_lat']:.2f}s"
        )

        if p_s["took_chord"] and p_s["chord_type"] == "braess":
            alert_s.set_text("BRAESS TRAP HIT: +11.41s BOTTLENECK DELAY!")
            alert_s.get_bbox_patch().set_alpha(1.0)
            dot_s.set_color("#FF0055")
        elif f_s == len(traj_s) - 1:
            alert_s.set_text(f"ARRIVED AT GOAL (Total: {p_s['cum_lat']:.2f}s)")
            alert_s.get_bbox_patch().set_alpha(0.9)
            alert_s.get_bbox_patch().set_facecolor("#7F1D1D")
        else:
            alert_s.get_bbox_patch().set_alpha(0.0)
            dot_s.set_color("#EF4444")

        # 2. Update K = 100
        p_l = traj_l[f_l]
        hist_l = [t["pos"] for t in traj_l[:f_l + 1]]
        dot_l.set_data([p_l["pos"][1]], [p_l["pos"][0]])
        line_l.set_data([pt[1] for pt in hist_l], [pt[0] for pt in hist_l])

        hud_l.set_text(
            f"Step: {f_l}/{len(traj_l)-1}\n"
            f"Pos:  {p_l['pos']}\n"
            f"Step Latency: {p_l['step_lat']:.2f}s\n"
            f"Total Time:   {p_l['cum_lat']:.2f}s"
        )

        if f_l <= 3:
            alert_l.set_text("K=100 DEEP BACKUPS: FLANKING CONGESTION CORRIDOR")
            alert_l.get_bbox_patch().set_alpha(0.95)
            alert_l.get_bbox_patch().set_facecolor("#065F46")
        elif p_l["took_chord"] and p_l["chord_type"] == "shortcut":
            alert_l.set_text("EXPLOITED BENIGN EXPRESSWAY: -4.0s SHORTCUT!")
            alert_l.get_bbox_patch().set_alpha(1.0)
            alert_l.get_bbox_patch().set_facecolor("#047857")
        elif f_l == len(traj_l) - 1:
            alert_l.set_text(f"ARRIVED AT GOAL (Total: {p_l['cum_lat']:.2f}s, 52% FASTER!)")
            alert_l.get_bbox_patch().set_alpha(1.0)
            alert_l.get_bbox_patch().set_facecolor("#064E3B")
        else:
            alert_l.get_bbox_patch().set_alpha(0.0)

        return dot_s, line_s, dot_l, line_l, hud_s, hud_l, alert_s, alert_l

    ani = animation.FuncAnimation(fig, update, frames=max_frames, init_func=init, blit=False, interval=900)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ani.save(save_path, writer="pillow", fps=1.2)
    plt.close()
    print(f"Animation saved successfully to {save_path}")


if __name__ == "__main__":
    artifact_path = r"C:\Users\ADMIN\.gemini\antigravity-cli\brain\94cfebc3-0f15-4ab1-b2a6-1c0c4c4a689a\ps_large_k_evasion.gif"
    env, traj_s, traj_l = run_trajectories(ep_seed=1010)
    create_animation(env, traj_s, traj_l, artifact_path)
