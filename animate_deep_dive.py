"""
animate_deep_dive.py
====================
Deep-Dive Side-by-Side Visualization of Standard-PS vs. SL-PS:
- Arbitrary background flow s_bg -> t_bg inside grid (e.g., (1,1) -> (3,6))
- Expanded 4x8 grid (45 nodes, 76 edges, 2:1 ratio)
- Value function V(s) 2D heatmap
- Policy vector field pi(s) arrows across all nodes
- Real-time HUD showing Action-value Q(s_t, a) decomposition at current agent location
- Explicit demonstration answering: "When does an agent STILL take a V_uv < 0 chord?"

Usage:
  python animate_deep_dive.py --traffic high --save sl_ps_deep_dive_high_traffic.gif
  python animate_deep_dive.py --traffic low --save sl_ps_deep_dive_low_traffic.gif
  python animate_deep_dive.py --traffic high --show
"""

import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from grid_braess_env import CongestedGridEnv
from planning_agents import StandardPrioritizedSweepingAgent, SLPSAgent


def setup_simulation(
    m: int = 4,
    n: int = 8,
    s_bg: tuple = (1, 1),
    t_bg: tuple = (3, 6),
    start_pos: tuple = (0, 0),
    q_val: float = 1.0,
    beta_scale: float = 6.0
):
    """
    Initializes environment, background equilibrium, and both agents.
    Returns env, std_agent, sl_agent, target_chord.
    """
    env = CongestedGridEnv(
        m=m,
        n=n,
        s_bg=s_bg,
        t_bg=t_bg,
        q=q_val,
        p_chord=0.0,
        noise_std=0.0,  # Zero noise for reproducible visual analysis
        seed=42
    )

    # Candidate Braess chord: (1, 4) -> (3, 3)
    target_chord = None
    for c in env.braess_chords:
        if c[0] == (1, 4) and c[1] == (3, 3):
            target_chord = c
            break
    if target_chord is None:
        if len(env.braess_chords) > 0:
            target_chord = env.braess_chords[0]
        else:
            raise RuntimeError("No Braess chords found on grid!")

    # Activate chord and solve Beckmann equilibrium
    env.active_chord = target_chord
    env.chord_type = "braess"
    u_c, v_c, _, _ = target_chord
    f_aug, _, z_star = env.grid.solve_augmented_equilibrium(u_c, v_c, c=0.0, d=0.0)
    env.current_edge_flows = f_aug[:-1]
    env.current_chord_flow = z_star
    env._check_and_spawn_chord = lambda: None  # Keep chord stable

    # Agents
    std_agent = StandardPrioritizedSweepingAgent(env, backup_budget=30)
    sl_agent = SLPSAgent(env, backup_budget=30, beta_scale=beta_scale)

    std_agent.plan()
    sl_agent.plan()

    # Pre-rollout trajectories
    def rollout(agent):
        env.agent_pos = start_pos
        env.step_count = 0
        traj = []
        cum_lat = 0.0
        traj.append({
            "pos": env.agent_pos,
            "cost": 0.0,
            "cum_lat": 0.0,
            "action": -1,
            "q_breakdown": agent.get_q_breakdown(env.agent_pos)
        })
        done = False
        while not done and len(traj) < 30:
            a = agent.select_action(env.agent_pos, target_chord)
            nxt, cost, done, _ = env.step(a)
            cum_lat += cost
            traj.append({
                "pos": nxt,
                "cost": cost,
                "cum_lat": cum_lat,
                "action": a,
                "q_breakdown": agent.get_q_breakdown(nxt) if not done else {}
            })
        return traj

    traj_std = rollout(std_agent)
    traj_sl = rollout(sl_agent)

    return env, std_agent, sl_agent, target_chord, traj_std, traj_sl, z_star


def render_animation(
    env,
    std_agent,
    sl_agent,
    target_chord,
    traj_std,
    traj_sl,
    z_star: float,
    traffic_label: str = "High",
    save_path: str = "sl_ps_deep_dive_animation.gif",
    show: bool = False
):
    m, n = env.m, env.n
    u_c, v_c, V_uv, k = target_chord
    max_steps = max(len(traj_std), len(traj_sl))
    total_frames = max_steps + 4

    # Matplotlib styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(17, 9.5), dpi=130)
    fig.patch.set_facecolor("#0F172A")  # Dark elegant aesthetic

    # Subplot grid: 2 main map panels + 2 lower HUD panels
    gs = fig.add_gridspec(2, 2, height_ratios=[3.6, 1.2], hspace=0.28, wspace=0.15,
                           left=0.06, right=0.94, top=0.90, bottom=0.06)

    ax_map_std = fig.add_subplot(gs[0, 0])
    ax_map_sl = fig.add_subplot(gs[0, 1])
    ax_hud_std = fig.add_subplot(gs[1, 0])
    ax_hud_sl = fig.add_subplot(gs[1, 1])

    # Shared value range for heatmaps
    all_v = [std_agent.V[(i, j)] for i in range(m + 1) for j in range(n + 1)] + \
            [sl_agent.V[(i, j)] for i in range(m + 1) for j in range(n + 1)]
    vmin, vmax = min(all_v), max(all_v)
    v_norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.plasma

    # Header text
    regime_desc = f"Regime: {traffic_label} Traffic (q={env.grid.q:.2f}, Chord Flow z*={z_star:.3f})"
    fig.suptitle(
        f"Deep Dive: Standard-PS vs. Spectral-Laplacian PS (SL-PS) on 4x8 Congested Grid\n{regime_desc}",
        fontsize=15, fontweight="bold", color="#F8FAFC", y=0.97
    )

    def draw_base_map(ax, agent, title, subtitle, is_sl=False):
        ax.set_facecolor("#1E293B")
        ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold", color="#E2E8F0", pad=10)
        ax.set_xlim(-0.7, n + 0.7)
        ax.set_ylim(m + 0.7, -0.7)
        ax.set_aspect("equal")
        ax.set_xticks(range(n + 1))
        ax.set_yticks(range(m + 1))
        ax.tick_params(colors="#94A3B8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334155")

        # 1. Background flow corridor highlight box [1..3] x [1..6]
        bg_rect = Rectangle(
            (env.s_bg[1] - 0.35, env.s_bg[0] - 0.35),
            env.t_bg[1] - env.s_bg[1] + 0.7,
            env.t_bg[0] - env.s_bg[0] + 0.7,
            linewidth=1.5, edgecolor="#EAB308", facecolor="#EAB308", alpha=0.08, linestyle="--", zorder=1
        )
        ax.add_patch(bg_rect)

        # 2. Draw base edges
        for (u, v) in env.grid.edges:
            ui, uj = env.grid.idx_to_node[u]
            vi, vj = env.grid.idx_to_node[v]
            e_idx = env._get_edge_index((ui, uj), (vi, vj))
            flow = env.current_edge_flows[e_idx]
            lw = 1.0 + 3.0 * (flow / (env.grid.q + 1e-6))
            edge_col = "#F59E0B" if flow > 0.05 else "#475569"
            ax.annotate("", xy=(vj, vi), xytext=(uj, ui),
                        arrowprops=dict(arrowstyle="-|>", color=edge_col, lw=lw, alpha=0.7, mutation_scale=10),
                        zorder=2)

        # 3. Braess Chord: (u_c) -> (v_c)
        arc_col = "#EF4444" if not is_sl else "#06B6D4"
        ax.annotate(
            "", xy=(v_c[1], v_c[0]), xytext=(u_c[1], u_c[0]),
            arrowprops=dict(
                arrowstyle="-|>",
                connectionstyle="arc3,rad=-0.35",
                color=arc_col,
                lw=2.8,
                linestyle="--",
                mutation_scale=16
            ),
            zorder=6
        )
        # Chord label
        mid_x = (u_c[1] + v_c[1]) / 2.0 - 0.35
        mid_y = (u_c[0] + v_c[0]) / 2.0 - 0.55
        ax.text(
            mid_x, mid_y,
            f"Braess Chord (1,4)->(3,3)\nV_uv = {V_uv:.3f} < 0",
            fontsize=7.5, fontweight="bold", color="#FCA5A5" if not is_sl else "#67E8F9",
            ha="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="#0F172A", edgecolor=arc_col, alpha=0.9),
            zorder=7
        )

        # 4. Heatmap of V(s) at each node
        for i in range(m + 1):
            for j in range(n + 1):
                val = agent.V[(i, j)]
                c = cmap(v_norm(val))
                circ = Circle((j, i), 0.28, facecolor=c, edgecolor="#64748B", lw=1.0, zorder=3)
                ax.add_patch(circ)
                # Text showing V(s)
                ax.text(j, i, f"{val:.1f}", fontsize=6.5, ha="center", va="center", color="#F8FAFC", fontweight="bold", zorder=4)

        # 5. Policy vector field pi(s) arrows
        pol = agent.get_policy_map()
        for i in range(m + 1):
            for j in range(n + 1):
                s = (i, j)
                if s == (m, n):
                    continue
                act = pol.get(s, -1)
                if act == 0:  # Down (Row + 1)
                    ax.annotate("", xy=(j, i + 0.32), xytext=(j, i),
                                arrowprops=dict(arrowstyle="->", color="#A7F3D0", lw=1.5, mutation_scale=8), zorder=5)
                elif act == 1:  # Right (Col + 1)
                    ax.annotate("", xy=(j + 0.32, i), xytext=(j, i),
                                arrowprops=dict(arrowstyle="->", color="#A7F3D0", lw=1.5, mutation_scale=8), zorder=5)
                elif act == 2:  # Chord
                    ax.annotate("", xy=(j - 0.25, i + 0.25), xytext=(j, i),
                                arrowprops=dict(arrowstyle="->", color="#F43F5E", lw=2.2, mutation_scale=10), zorder=5)

        # 6. Landmark markers
        # Agent Start & Goal
        ax.plot(0, 0, marker="*", markersize=14, color="#10B981", markeredgecolor="#ECFDF5", zorder=8, label="Start (0,0)")
        ax.plot(n, m, marker="X", markersize=12, color="#8B5CF6", markeredgecolor="#EDE9FE", zorder=8, label=f"Goal ({m},{n})")
        # Background Flow Source & Sink
        ax.plot(env.s_bg[1], env.s_bg[0], marker="D", markersize=9, color="#EAB308", markeredgecolor="#FEF08A", zorder=8, label=f"BG Src {env.s_bg}")
        ax.plot(env.t_bg[1], env.t_bg[0], marker="s", markersize=9, color="#EAB308", markeredgecolor="#FEF08A", zorder=8, label=f"BG Sink {env.t_bg}")

        ax.legend(loc="lower left", fontsize=7.5, facecolor="#0F172A", edgecolor="#334155", labelcolor="#CBD5E1")

    # Draw both static maps
    draw_base_map(
        ax_map_std, std_agent,
        "Standard Prioritized Sweeping (Standard-PS)",
        "Local Greedy Bellman Backup -> Blind to Network Divergence",
        is_sl=False
    )
    draw_base_map(
        ax_map_sl, sl_agent,
        "Spectral-Laplacian Prioritized Sweeping (SL-PS)",
        "Spectral Congestion Potential Phi_spec -> Proactive Bypass",
        is_sl=True
    )

    # Dynamic elements
    # Trajectory lines
    trail_line_std, = ax_map_std.plot([], [], color="#F97316", lw=3.0, alpha=0.85, zorder=9)
    agent_dot_std, = ax_map_std.plot([], [], marker="o", markersize=12, color="#EF4444", markeredgecolor="#FFFFFF", markeredgewidth=2, zorder=10)

    trail_line_sl, = ax_map_sl.plot([], [], color="#38BDF8", lw=3.0, alpha=0.85, zorder=9)
    agent_dot_sl, = ax_map_sl.plot([], [], marker="o", markersize=12, color="#0EA5E9", markeredgecolor="#FFFFFF", markeredgewidth=2, zorder=10)

    # HUD Configuration
    for ax_hud in (ax_hud_std, ax_hud_sl):
        ax_hud.set_facecolor("#1E293B")
        ax_hud.tick_params(colors="#94A3B8", labelsize=8)
        for spine in ax_hud.spines.values():
            spine.set_color("#334155")

    def update_hud(ax, step_info, title_agent, is_sl=False):
        ax.clear()
        ax.set_facecolor("#1E293B")
        pos = step_info["pos"]
        q_dict = step_info["q_breakdown"]
        cum_lat = step_info["cum_lat"]

        if not q_dict:
            # Reached goal
            ax.text(0.5, 0.5, f"{title_agent} ARRIVED AT GOAL ({m},{n})!\nTotal Latency: {cum_lat:.2f} s",
                    ha="center", va="center", color="#34D399", fontsize=11, fontweight="bold", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        actions = list(q_dict.keys())
        labels = []
        q_totals = []
        nom_costs = []
        fut_vs = []
        spec_pens = []

        action_names = {0: "Down (Row+1)", 1: "Right (Col+1)", 2: "CHORD (1,4)->(3,3)"}
        for a in actions:
            labels.append(action_names.get(a, f"Act {a}"))
            d = q_dict[a]
            nom_costs.append(d["cost"])
            fut_vs.append(d["future_v"])
            spec_pens.append(d.get("spectral_penalty", 0.0))
            q_totals.append(d["total_q"])

        best_a = actions[int(np.argmin(q_totals))]
        x = np.arange(len(actions))
        width = 0.55

        # Stacked bar breakdown
        p1 = ax.bar(x, nom_costs, width, label="Edge Cost c(s,a)", color="#F59E0B", alpha=0.9)
        p2 = ax.bar(x, fut_vs, width, bottom=nom_costs, label="Discounted V(s')", color="#6366F1", alpha=0.9)
        bottoms = [c + v for c, v in zip(nom_costs, fut_vs)]
        if is_sl:
            p3 = ax.bar(x, spec_pens, width, bottom=bottoms, label="Beta * Phi_spec (Penalty)", color="#EC4899", alpha=0.9)

        # Highlight best action with text on top
        for idx, (tot, a) in enumerate(zip(q_totals, actions)):
            star = " * [BEST]" if a == best_a else ""
            col = "#34D399" if a == best_a else "#94A3B8"
            ax.text(idx, tot + 0.3, f"Q={tot:.2f}{star}", ha="center", color=col, fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5, color="#E2E8F0", fontweight="bold")
        ax.set_ylabel("Q(s, a)", color="#94A3B8", fontsize=8)
        ax.set_title(f"{title_agent} Live Q-Breakdown at Current State {pos} | Cum Latency: {cum_lat:.2f}",
                     fontsize=9.5, fontweight="bold", color="#F8FAFC", pad=6)
        ax.set_ylim(0, max(q_totals) * 1.35 + 1.0)
        ax.legend(loc="upper right", fontsize=7, facecolor="#0F172A", edgecolor="#334155", labelcolor="#CBD5E1")

    def animate(frame):
        # Frame index clamped
        idx_std = min(frame, len(traj_std) - 1)
        idx_sl = min(frame, len(traj_sl) - 1)

        info_std = traj_std[idx_std]
        info_sl = traj_sl[idx_sl]

        # Update Std trail and dot
        xs_std = [step["pos"][1] for step in traj_std[:idx_std + 1]]
        ys_std = [step["pos"][0] for step in traj_std[:idx_std + 1]]
        trail_line_std.set_data(xs_std, ys_std)
        agent_dot_std.set_data([info_std["pos"][1]], [info_std["pos"][0]])

        # Update SL trail and dot
        xs_sl = [step["pos"][1] for step in traj_sl[:idx_sl + 1]]
        ys_sl = [step["pos"][0] for step in traj_sl[:idx_sl + 1]]
        trail_line_sl.set_data(xs_sl, ys_sl)
        agent_dot_sl.set_data([info_sl["pos"][1]], [info_sl["pos"][0]])

        # Update HUDs
        update_hud(ax_hud_std, info_std, "Standard-PS", is_sl=False)
        update_hud(ax_hud_sl, info_sl, "SL-PS", is_sl=True)

        return trail_line_std, agent_dot_std, trail_line_sl, agent_dot_sl

    ani = animation.FuncAnimation(fig, animate, frames=total_frames, interval=800, repeat=True, blit=False)

    if save_path:
        print(f"Rendering deep-dive animation to {save_path}...")
        ani.save(save_path, writer="pillow", fps=1.2)
        print(f"Successfully saved {save_path}!")

    if show:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Deep Dive Animation: Standard-PS vs SL-PS")
    parser.add_argument("--traffic", choices=["high", "low"], default="high",
                        help="Traffic regime: high (q=1.0) or low (q=0.05)")
    parser.add_argument("--start", type=str, default="0,0",
                        help="Agent start coordinate 'i,j' (default: 0,0)")
    parser.add_argument("--save", type=str, default="sl_ps_deep_dive_animation.gif",
                        help="Path to save output animated GIF")
    parser.add_argument("--show", action="store_true",
                        help="Show interactive window instead of just saving")
    args = parser.parse_args()

    # Parse start position
    start_pos = tuple(map(int, args.start.split(",")))

    # Traffic parameters
    if args.traffic == "high":
        q_val = 1.0
        label = "High Congestion (q=1.00)"
    else:
        q_val = 0.05
        label = "Light Traffic (q=0.05)"

    print(f"Setting up simulation on 4x8 grid with {label}, start={start_pos}...")
    env, std_agent, sl_agent, target_chord, traj_std, traj_sl, z_star = setup_simulation(
        m=4, n=8,
        s_bg=(1, 1), t_bg=(3, 6),
        start_pos=start_pos,
        q_val=q_val,
        beta_scale=6.0
    )

    print(f"Simulation completed:")
    print(f"  Standard-PS steps: {len(traj_std)-1}, final latency: {traj_std[-1]['cum_lat']:.2f}")
    print(f"  SL-PS steps:       {len(traj_sl)-1}, final latency: {traj_sl[-1]['cum_lat']:.2f}")

    render_animation(
        env, std_agent, sl_agent, target_chord,
        traj_std, traj_sl, z_star,
        traffic_label=label,
        save_path=args.save,
        show=args.show
    )


if __name__ == "__main__":
    main()
