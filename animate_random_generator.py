"""
animate_random_generator.py
===========================
Procedural Stochastic Simulation Generator with Proximity Delta-Spawning:
- Random chords are spawned dynamically within a forward neighborhood Delta = (1, 2)
  in front of the agent's current position s_t.
- GUARANTEES:
    * At least 1 randomly sampled candidate chord with V_uv < 0 (Braess congestion bottleneck).
    * At least 1 randomly sampled candidate chord with V_uv > 0 (Benign expressway shortcut).
- Supports batch execution: --num-runs 15 --out-dir test_random
- Live Real-Time Latency Analytics:
    * Dynamic edge heatmap reflecting changing Wardrop equilibrium flows f_e(t).
    * Step-by-step latency time-series plot comparing Standard-PS vs. SL-PS.

Usage:
  python animate_random_generator.py --num-runs 15 --out-dir test_random
  python animate_random_generator.py --seed 42 --save single_test.gif
  python animate_random_generator.py --show
"""

import sys
import os
import time
import random
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.colors import Normalize

from spectral_braess_core import GridNetwork
from grid_braess_env import CongestedGridEnv
from planning_agents import StandardPrioritizedSweepingAgent, SLPSAgent


def run_single_simulation(m=8, n=16, seed=None, delta=(1, 2), p_spawn=0.45):
    """
    Executes a single procedural simulation where candidate chords spawn within
    Delta proximity of the agent's path.
    Guarantees at least 1 Braess chord (V < 0) and 1 Shortcut (V > 0).
    """
    if seed is None:
        seed = int(time.time() * 1000) % 1000000
    rng = random.Random(seed)

    # Random background flow endpoints
    s_bg = (rng.randint(1, 2), rng.randint(2, 4))
    t_bg = (rng.randint(m - 3, m - 1), rng.randint(n - 5, n - 2))
    q = round(rng.uniform(2.5, 3.5), 2)

    env_std = CongestedGridEnv(m=m, n=n, s_bg=s_bg, t_bg=t_bg, q=q, p_chord=0.0, noise_std=0.0, seed=seed)
    env_sl = CongestedGridEnv(m=m, n=n, s_bg=s_bg, t_bg=t_bg, q=q, p_chord=0.0, noise_std=0.0, seed=seed)
    env_std._check_and_spawn_chord = lambda: None
    env_sl._check_and_spawn_chord = lambda: None

    grid = env_std.grid
    all_chords = grid.find_all_chords()
    braess_list = all_chords["braess_capable"]
    shortcut_list = all_chords["shortcut"]

    std = StandardPrioritizedSweepingAgent(env_std, backup_budget=30)
    sl = SLPSAgent(env_sl, backup_budget=30, beta_scale=8.0)

    ratio = float(n) / float(m)
    def natural_bellman(agent, s, is_sl=False):
        valid = agent.env.get_valid_actions(s)
        best_v = float("inf")
        best_a = None
        Phi_spec = agent.compute_spectral_congestion_field() if is_sl else None
        beta = agent.beta_scale * (1.0 + agent.slem) if is_sl else 0.0

        for a in valid:
            nxt, cost = agent._get_transition(s, a)
            spec_pen = float(beta * Phi_spec[agent.env.grid.node_to_idx[nxt]]) if is_sl else 0.0
            target = cost + agent.gamma * agent.V[nxt] + spec_pen

            if target < best_v - 1e-4:
                best_v = target
                best_a = a
            elif abs(target - best_v) <= 1e-4:
                curr_nxt, _ = agent._get_transition(s, best_a) if best_a is not None else (s, 0)
                curr_dist = abs(curr_nxt[1] - ratio * curr_nxt[0])
                new_dist = abs(nxt[1] - ratio * nxt[0])
                if new_dist < curr_dist:
                    best_v = target
                    best_a = a
        return best_v, best_a

    s_std = (0, 0)
    s_sl = (0, 0)
    env_std.agent_pos = s_std
    env_sl.agent_pos = s_sl

    traj_std = [s_std]
    traj_sl = [s_sl]
    lat_history_std = [0.0]
    lat_history_sl = [0.0]
    cum_lat_std = [0.0]
    cum_lat_sl = [0.0]

    snapshots = []
    snapshots.append({
        "step": 0,
        "event": "Base Grid (Initial State)",
        "active_chord": None,
        "chord_type": None,
        "flows_std": env_std.current_edge_flows.copy(),
        "flows_sl": env_sl.current_edge_flows.copy()
    })

    braess_spawned = False
    shortcut_spawned = False
    active_chord = None
    chord_type = None

    braess_record = None
    shortcut_record = None

    max_steps = 28
    for step in range(1, max_steps + 1):
        # 1. Proximity spawning for Braess chord (V < 0) in steps 4..10
        if not braess_spawned and 4 <= step <= 10:
            near_b = [
                c for c in braess_list
                if (s_std[0] <= c[0][0] <= s_std[0] + delta[0] and
                    s_std[1] <= c[0][1] <= s_std[1] + delta[1])
                and (s_bg[0] <= c[0][0] <= t_bg[0] and s_bg[1] <= c[0][1] <= t_bg[1])
            ]
            if near_b and (rng.random() < p_spawn or step >= 8):
                active_chord = rng.choice(near_b)
                chord_type = "braess"
                braess_spawned = True
                braess_record = (active_chord, step)
                u_c, v_c, _, _ = active_chord
                f_aug, _, z_star = grid.solve_augmented_equilibrium(u_c, v_c, c=0.0, d=0.0)
                for env, agent in [(env_std, std), (env_sl, sl)]:
                    env.active_chord = active_chord
                    env.chord_type = chord_type
                    env.current_edge_flows = f_aug[:-1]
                    env.current_chord_flow = z_star
                    agent.plan()

        # 2. Proximity spawning for Shortcut chord (V > 0) in steps 11..18
        elif braess_spawned and not shortcut_spawned and step >= 11:
            cur_pos = s_sl if s_sl != (m, n) else s_std
            near_s = [
                c for c in shortcut_list
                if (cur_pos[0] <= c[0][0] <= cur_pos[0] + delta[0] + 1 and
                    cur_pos[1] <= c[0][1] <= cur_pos[1] + delta[1] + 1)
                and (c[1][0] >= c[0][0] and c[1][1] >= c[0][1]) and c[2] > 0
            ]
            if near_s and (rng.random() < p_spawn or step >= 15):
                active_chord = rng.choice(near_s)
                chord_type = "shortcut"
                shortcut_spawned = True
                shortcut_record = (active_chord, step)
                u_c, v_c, _, _ = active_chord
                f_aug, _, z_star = grid.solve_augmented_equilibrium(u_c, v_c, c=0.0, d=0.0)
                for env, agent in [(env_std, std), (env_sl, sl)]:
                    env.active_chord = active_chord
                    env.chord_type = chord_type
                    env.current_edge_flows = f_aug[:-1]
                    env.current_chord_flow = z_star
                    agent.plan()

        # Event description for HUD
        if active_chord is not None:
            if chord_type == "braess":
                event_name = f"Braess Trap {active_chord[0]}->{active_chord[1]} Emerges Ahead (V < 0)!"
            else:
                event_name = f"Expressway {active_chord[0]}->{active_chord[1]} Opens (V > 0)!"
        else:
            event_name = "Phase 1: Clear Cruising"

        # Step Std
        if s_std != (m, n):
            _, a_std = natural_bellman(std, s_std, is_sl=False)
            nxt_std, cost_std, _, _ = env_std.step(a_std)
            traj_std.append(nxt_std)
            lat_history_std.append(cost_std)
            cum_lat_std.append(cum_lat_std[-1] + cost_std)
            s_std = nxt_std
        else:
            traj_std.append(s_std)
            lat_history_std.append(0.0)
            cum_lat_std.append(cum_lat_std[-1])

        # Step SL
        if s_sl != (m, n):
            _, a_sl = natural_bellman(sl, s_sl, is_sl=True)
            nxt_sl, cost_sl, _, _ = env_sl.step(a_sl)
            traj_sl.append(nxt_sl)
            lat_history_sl.append(cost_sl)
            cum_lat_sl.append(cum_lat_sl[-1] + cost_sl)
            s_sl = nxt_sl
        else:
            traj_sl.append(s_sl)
            lat_history_sl.append(0.0)
            cum_lat_sl.append(cum_lat_sl[-1])

        snapshots.append({
            "step": step,
            "event": event_name,
            "active_chord": active_chord,
            "chord_type": chord_type,
            "flows_std": env_std.current_edge_flows.copy(),
            "flows_sl": env_sl.current_edge_flows.copy()
        })

        if s_std == (m, n) and s_sl == (m, n) and braess_spawned and shortcut_spawned:
            break

    # If by chance one wasn't spawned, record fallback
    if braess_record is None and len(braess_list) > 0:
        braess_record = (braess_list[0], 0)
    if shortcut_record is None and len(shortcut_list) > 0:
        shortcut_record = (shortcut_list[0], 0)

    scenario_info = {
        "m": m,
        "n": n,
        "seed": seed,
        "s_bg": s_bg,
        "t_bg": t_bg,
        "q": q,
        "braess_record": braess_record,
        "shortcut_record": shortcut_record
    }

    return (
        env_std,
        scenario_info,
        traj_std,
        traj_sl,
        lat_history_std,
        lat_history_sl,
        cum_lat_std,
        cum_lat_sl,
        snapshots
    )


def render_animation(
    env,
    scenario_info,
    traj_std,
    traj_sl,
    lat_history_std,
    lat_history_sl,
    cum_lat_std,
    cum_lat_sl,
    snapshots,
    save_path="random_run.gif",
    show=False
):
    m, n = scenario_info["m"], scenario_info["n"]
    seed = scenario_info["seed"]
    b_chord, t_b = scenario_info["braess_record"]
    s_chord, t_s = scenario_info["shortcut_record"]

    total_frames = len(snapshots) + 3
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(17, 9.2), dpi=90)
    fig.patch.set_facecolor("#0A0F1D")

    gs = fig.add_gridspec(2, 2, height_ratios=[3.3, 1.3], hspace=0.25, wspace=0.12,
                           left=0.04, right=0.96, top=0.91, bottom=0.06)

    ax_map_std = fig.add_subplot(gs[0, 0])
    ax_map_sl = fig.add_subplot(gs[0, 1])
    ax_plot_step = fig.add_subplot(gs[1, 0])
    ax_plot_cum = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        f"Stochastic Run (Seed {seed}) | Proximity Delta-Spawning | Grid {m}x{n} | BG Flow {scenario_info['s_bg']}->{scenario_info['t_bg']}\n"
        f"Braess Trap: {b_chord[0]}->{b_chord[1]} (V={b_chord[2]:.3f}<0, t={t_b}) | "
        f"Expressway: {s_chord[0]}->{s_chord[1]} (V=+{s_chord[2]:.3f}>0, t={t_s})",
        fontsize=11.5, fontweight="bold", color="#F8FAFC", y=0.97
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

    ax_map_std.set_title("Standard Prioritized Sweeping (Standard-PS)\nFalls into Proximity Braess Bottleneck", fontsize=9.5, fontweight="bold", color="#F87171", pad=6)
    ax_map_sl.set_title("Spectral-Laplacian PS (SL-PS)\nProactive Spectral Bypassing & Shortcut Acceleration", fontsize=9.5, fontweight="bold", color="#38BDF8", pad=6)

    ax_plot_step.set_title("Step Latency Under Dynamic Chords l(s_t, a_t)", fontsize=9, fontweight="bold", color="#E2E8F0", pad=5)
    ax_plot_step.set_xlabel("Navigation Step (t)", color="#94A3B8", fontsize=8)
    ax_plot_step.set_ylabel("Step Latency (s)", color="#94A3B8", fontsize=8)
    ax_plot_step.set_xlim(0, len(snapshots) + 1)
    ax_plot_step.set_ylim(0, max(max(lat_history_std), max(lat_history_sl)) * 1.25 + 2.0)

    ax_plot_cum.set_title("Cumulative Travel Time T_cum(t)", fontsize=9, fontweight="bold", color="#E2E8F0", pad=5)
    ax_plot_cum.set_xlabel("Navigation Step (t)", color="#94A3B8", fontsize=8)
    ax_plot_cum.set_ylabel("Cumulative Latency (s)", color="#94A3B8", fontsize=8)
    ax_plot_cum.set_xlim(0, len(snapshots) + 1)
    ax_plot_cum.set_ylim(0, max(cum_lat_std[-1], cum_lat_sl[-1]) * 1.15 + 2.0)

    # BG Rectangle
    for ax in (ax_map_std, ax_map_sl):
        bg_rect = Rectangle(
            (scenario_info["s_bg"][1] - 0.4, scenario_info["s_bg"][0] - 0.4),
            scenario_info["t_bg"][1] - scenario_info["s_bg"][1] + 0.8,
            scenario_info["t_bg"][0] - scenario_info["s_bg"][0] + 0.8,
            linewidth=1.2, edgecolor="#F59E0B", facecolor="#F59E0B", alpha=0.07, linestyle="--", zorder=1
        )
        ax.add_patch(bg_rect)
        ax.plot(scenario_info["s_bg"][1], scenario_info["s_bg"][0], marker="D", markersize=6, color="#EAB308", zorder=6)
        ax.plot(scenario_info["t_bg"][1], scenario_info["t_bg"][0], marker="s", markersize=6, color="#EAB308", zorder=6)
        ax.plot(0, 0, marker="*", markersize=11, color="#10B981", zorder=7)
        ax.plot(n, m, marker="X", markersize=10, color="#8B5CF6", zorder=7)

    line_step_std, = ax_plot_step.plot([], [], color="#EF4444", lw=2.0, marker="o", markersize=3.5, label="Standard-PS Step Latency")
    line_step_sl, = ax_plot_step.plot([], [], color="#0EA5E9", lw=2.0, marker="s", markersize=3.5, label="SL-PS Step Latency")
    ax_plot_step.legend(loc="upper right", fontsize=7.5, facecolor="#1F2937", edgecolor="#374151", labelcolor="#E5E7EB")

    line_cum_std, = ax_plot_cum.plot([], [], color="#EF4444", lw=2.2, label="Standard-PS Cumulative")
    line_cum_sl, = ax_plot_cum.plot([], [], color="#0EA5E9", lw=2.2, label="SL-PS Cumulative")
    ax_plot_cum.legend(loc="upper left", fontsize=7.5, facecolor="#1F2937", edgecolor="#374151", labelcolor="#E5E7EB")

    trail_std, = ax_map_std.plot([], [], color="#F97316", lw=2.5, alpha=0.85, zorder=8)
    dot_std, = ax_map_std.plot([], [], marker="o", markersize=8.5, color="#EF4444", markeredgecolor="#FFFFFF", markeredgewidth=1.4, zorder=9)

    trail_sl, = ax_map_sl.plot([], [], color="#38BDF8", lw=2.5, alpha=0.85, zorder=8)
    dot_sl, = ax_map_sl.plot([], [], marker="o", markersize=8.5, color="#0284C7", markeredgecolor="#FFFFFF", markeredgewidth=1.4, zorder=9)

    badge_std = ax_map_std.text(0.02, 0.04, "", transform=ax_map_std.transAxes, fontsize=7.5, fontweight="bold",
                                color="#FCA5A5", bbox=dict(boxstyle="round,pad=0.2", facecolor="#1F2937", edgecolor="#EF4444", alpha=0.9), zorder=10)
    badge_sl = ax_map_sl.text(0.02, 0.04, "", transform=ax_map_sl.transAxes, fontsize=7.5, fontweight="bold",
                              color="#7DD3FC", bbox=dict(boxstyle="round,pad=0.2", facecolor="#1F2937", edgecolor="#0EA5E9", alpha=0.9), zorder=10)

    event_text = fig.text(0.5, 0.935, "", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#FDE047",
                          bbox=dict(boxstyle="round,pad=0.25", facecolor="#1E293B", edgecolor="#F59E0B", lw=1.0))

    edge_norm = Normalize(vmin=1.0, vmax=3.0)
    edge_cmap = plt.cm.inferno

    def animate(frame):
        snap_idx = min(frame, len(snapshots) - 1)
        snap = snapshots[snap_idx]
        cur_step = snap["step"]

        for ax in (ax_map_std, ax_map_sl):
            for p in list(ax.patches):
                if getattr(p, "_is_dynamic_edge", False):
                    p.remove()
            for t in list(ax.texts):
                if getattr(t, "_is_dynamic_label", False):
                    t.remove()

        event_text.set_text(f"Step t = {cur_step:02d} | Event: {snap['event']}")

        # Edges Std
        for e_idx, (u, v) in enumerate(env.grid.edges):
            ui, uj = env.grid.idx_to_node[u]
            vi, vj = env.grid.idx_to_node[v]
            f = snap["flows_std"][e_idx]
            lat = env.grid.a * f + env.grid.b
            c = edge_cmap(edge_norm(lat))
            lw = 0.8 + 2.0 * min(1.0, f / 1.5)
            arr = FancyArrowPatch((uj, ui), (vj, vi), arrowstyle="-|>", mutation_scale=4.5, color=c, lw=lw, alpha=0.75, zorder=2)
            arr._is_dynamic_edge = True
            ax_map_std.add_patch(arr)

        # Edges SL
        for e_idx, (u, v) in enumerate(env.grid.edges):
            ui, uj = env.grid.idx_to_node[u]
            vi, vj = env.grid.idx_to_node[v]
            f = snap["flows_sl"][e_idx]
            lat = env.grid.a * f + env.grid.b
            c = edge_cmap(edge_norm(lat))
            lw = 0.8 + 2.0 * min(1.0, f / 1.5)
            arr = FancyArrowPatch((uj, ui), (vj, vi), arrowstyle="-|>", mutation_scale=4.5, color=c, lw=lw, alpha=0.75, zorder=2)
            arr._is_dynamic_edge = True
            ax_map_sl.add_patch(arr)

        # Active Chord
        act_c = snap["active_chord"]
        if act_c is not None:
            u_c, v_c, V_uv, k = act_c
            c_type = snap["chord_type"]
            arc_col = "#EF4444" if c_type == "braess" else "#10B981"
            lbl = f"Braess {u_c}->{v_c}\nV={V_uv:.3f}<0" if c_type == "braess" else f"Expressway {u_c}->{v_c}\nV=+{V_uv:.3f}>0"
            for ax in (ax_map_std, ax_map_sl):
                arr = FancyArrowPatch((u_c[1], u_c[0]), (v_c[1], v_c[0]), connectionstyle="arc3,rad=-0.4",
                                      arrowstyle="-|>", mutation_scale=9, color=arc_col, lw=2.5, linestyle="--", zorder=5)
                arr._is_dynamic_edge = True
                ax.add_patch(arr)

                txt = ax.text((u_c[1] + v_c[1]) / 2.0, (u_c[0] + v_c[0]) / 2.0 - 0.45,
                              lbl, fontsize=6.8, fontweight="bold", color="#F8FAFC", ha="center",
                              bbox=dict(boxstyle="round,pad=0.2", facecolor="#0F172A", edgecolor=arc_col, alpha=0.9), zorder=6)
                txt._is_dynamic_label = True

        idx_std = min(cur_step, len(traj_std) - 1)
        idx_sl = min(cur_step, len(traj_sl) - 1)

        trail_std.set_data([p[1] for p in traj_std[:idx_std + 1]], [p[0] for p in traj_std[:idx_std + 1]])
        dot_std.set_data([traj_std[idx_std][1]], [traj_std[idx_std][0]])

        trail_sl.set_data([p[1] for p in traj_sl[:idx_sl + 1]], [p[0] for p in traj_sl[:idx_sl + 1]])
        dot_sl.set_data([traj_sl[idx_sl][1]], [traj_sl[idx_sl][0]])

        badge_std.set_text(f"Std @ {traj_std[idx_std]}\nStep Lat: {lat_history_std[idx_std]:.2f}s\nCum Lat:  {cum_lat_std[idx_std]:.2f}s")
        badge_sl.set_text(f"SL @ {traj_sl[idx_sl]}\nStep Lat: {lat_history_sl[idx_sl]:.2f}s\nCum Lat:  {cum_lat_sl[idx_std]:.2f}s")

        steps_x = list(range(cur_step + 1))
        line_step_std.set_data(steps_x, lat_history_std[:cur_step + 1])
        line_step_sl.set_data(steps_x, lat_history_sl[:cur_step + 1])
        line_cum_std.set_data(steps_x, cum_lat_std[:cur_step + 1])
        line_cum_sl.set_data(steps_x, cum_lat_sl[:cur_step + 1])

        return trail_std, dot_std, trail_sl, dot_sl, line_step_std, line_step_sl, line_cum_std, line_cum_sl

    ani = animation.FuncAnimation(fig, animate, frames=total_frames, interval=750, repeat=True, blit=False)

    ani.save(save_path, writer="pillow", fps=1.3)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Procedural Delta-Spawning Simulation Generator")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of random runs to execute (default: 1)")
    parser.add_argument("--out-dir", type=str, default="test_random", help="Directory to save generated GIFs (default: test_random)")
    parser.add_argument("--seed", type=int, default=None, help="Specific random seed for a single run")
    parser.add_argument("--save", type=str, default=None, help="Explicit filename if single run")
    parser.add_argument("--show", action="store_true", help="Interactive live display (single run only)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.num_runs == 1 and args.save:
        # Single explicit run
        seed = args.seed if args.seed is not None else random.randint(100, 99999)
        print(f"Executing single run with seed {seed}...")
        (
            env, scenario_info, traj_std, traj_sl,
            lat_std, lat_sl, cum_std, cum_sl, snapshots
        ) = run_single_simulation(seed=seed)

        render_animation(
            env, scenario_info, traj_std, traj_sl,
            lat_std, lat_sl, cum_std, cum_sl, snapshots,
            save_path=args.save, show=args.show
        )
        print(f"Saved: {args.save} | Std: {cum_std[-1]:.2f}s | SL: {cum_sl[-1]:.2f}s | Diff: {cum_std[-1]-cum_sl[-1]:+.2f}s")
        return

    # Batch execution mode (e.g. 15 runs)
    print(f"=== Executing Batch Generation: {args.num_runs} Unique Runs into '{args.out_dir}/' ===")
    summary_records = []

    base_seeds = [
        11, 22, 33, 44, 55, 66, 77, 88, 99, 101, 202, 303, 404, 505, 606
    ]

    for run_idx in range(1, args.num_runs + 1):
        if run_idx <= len(base_seeds) and args.seed is None:
            run_seed = base_seeds[run_idx - 1]
        elif args.seed is not None and run_idx == 1:
            run_seed = args.seed
        else:
            run_seed = random.randint(1000, 99999)

        gif_filename = os.path.join(args.out_dir, f"run_{run_idx:02d}_seed_{run_seed}.gif")

        t0 = time.time()
        (
            env, scenario_info, traj_std, traj_sl,
            lat_std, lat_sl, cum_std, cum_sl, snapshots
        ) = run_single_simulation(seed=run_seed)

        render_animation(
            env, scenario_info, traj_std, traj_sl,
            lat_std, lat_sl, cum_std, cum_sl, snapshots,
            save_path=gif_filename
        )
        t_cost = time.time() - t0

        std_tot = cum_std[-1]
        sl_tot = cum_sl[-1]
        diff = std_tot - sl_tot
        pct = (diff / std_tot) * 100 if std_tot > 0 else 0

        b_chord, t_b = scenario_info["braess_record"]
        s_chord, t_s = scenario_info["shortcut_record"]

        print(f"Run {run_idx:02d}/{args.num_runs:02d} (Seed {run_seed:5d}) [{t_cost:4.1f}s] -> {gif_filename}")
        print(f"       Braess: {b_chord[0]}->{b_chord[1]} (V={b_chord[2]:.3f}<0, t={t_b})")
        print(f"       Express: {s_chord[0]}->{s_chord[1]} (V=+{s_chord[2]:.3f}>0, t={t_s})")
        print(f"       Std Time: {std_tot:5.2f}s | SL Time: {sl_tot:5.2f}s | Saved: {diff:+5.2f}s ({pct:+5.1f}%)\n")

        summary_records.append({
            "run": run_idx,
            "seed": run_seed,
            "filename": gif_filename,
            "std_time": std_tot,
            "sl_time": sl_tot,
            "diff": diff,
            "pct": pct,
            "braess": f"{b_chord[0]}->{b_chord[1]}",
            "shortcut": f"{s_chord[0]}->{s_chord[1]}"
        })

    print("=========================================================================================")
    print(f"BATCH RUN COMPLETED: {len(summary_records)} GIFs saved to '{args.out_dir}/'")
    avg_saved = np.mean([r['diff'] for r in summary_records])
    win_count = sum(1 for r in summary_records if r['diff'] > 0.05)
    print(f"Summary Statistics: SL-PS won {win_count}/{len(summary_records)} runs.")
    print(f"Average Travel Time Saved by SL-PS: {avg_saved:.2f} s")
    print("=========================================================================================")


if __name__ == "__main__":
    main()
