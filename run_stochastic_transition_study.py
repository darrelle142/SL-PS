"""
run_stochastic_transition_study.py
==================================
Empirical Benchmark for Stochastic Environmental Transitions:
Addresses Professor's critique: "Nếu P(s'|s,a) mà deterministic thì dễ quá, đồ thị của mình chỉ có toàn 0 với 1"
Sweeps slip probability epsilon_slip in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25] on G_{3,6} across 30 episodes.
Compares:
1. Standard-PS (K = 15)
2. Standard-PS (K = 100)
3. SL-PS (Proposed, K = 15)

Outputs:
- stochastic_transition_results.json: Benchmark metrics
- fig4_stochastic_transitions.png: 300 DPI APA 7.0 multi-panel publication figure
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import shutil
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from grid_braess_env import CongestedGridEnv
from planning_agents import (
    StandardPrioritizedSweepingAgent,
    SLPSAgent
)


def evaluate_agent_on_env(agent_cls, agent_kwargs: dict, slip_prob: float, num_episodes: int = 30) -> dict:
    total_travel_times = []
    total_steps = []
    braess_traps_taken = 0
    total_chords_encountered = 0
    decision_times = []
    total_slips = 0

    # Instantiate environment with specific slip_prob
    env = CongestedGridEnv(
        m=3,
        n=6,
        a=1.0,
        b=1.0,
        q=1.0,
        p_chord=0.4,
        noise_std=0.02,
        seed=42,
        slip_prob=slip_prob
    )
    agent = agent_cls(env=env, **agent_kwargs)

    for ep in range(num_episodes):
        state, info = env.reset(seed=1000 + ep)
        done = False
        ep_latency = 0.0
        steps = 0
        ep_slips = 0

        # Reset agent value function per episode for fair comparison
        if hasattr(agent, "V"):
            agent.V = {
                (i, j): float((env.m - i) + (env.n - j)) * env.grid.b
                for i in range(env.m + 1)
                for j in range(env.n + 1)
            }
            agent.V[(env.m, env.n)] = 0.0

        while not done:
            t0 = time.perf_counter()
            action = agent.select_action(state, info["active_chord"])
            t1 = time.perf_counter()
            decision_times.append((t1 - t0) * 1000.0)

            if action == CongestedGridEnv.ACTION_CHORD and info["active_chord"] is not None:
                total_chords_encountered += 1
                if info["chord_type"] == "braess":
                    braess_traps_taken += 1

            next_state, latency, done, info = env.step(action)
            if info.get("slipped", False):
                ep_slips += 1
            ep_latency += latency
            steps += 1
            state = next_state

        total_travel_times.append(ep_latency)
        total_steps.append(steps)
        total_slips += ep_slips

    trap_rate = (braess_traps_taken / max(1, total_chords_encountered)) * 100.0

    return {
        "mean_latency": float(np.mean(total_travel_times)),
        "std_latency": float(np.std(total_travel_times)),
        "mean_steps": float(np.mean(total_steps)),
        "braess_trap_rate": float(trap_rate),
        "total_traps_taken": int(braess_traps_taken),
        "total_chords_encountered": int(total_chords_encountered),
        "mean_decision_ms": float(np.mean(decision_times)),
        "mean_slips_per_ep": float(total_slips / num_episodes)
    }


def run_stochastic_transition_study():
    print("=" * 80)
    print("STARTING STOCHASTIC TRANSITION ROBUSTNESS BENCHMARK (EPSILON_SLIP)")
    print("Sweeping slip_prob in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25] across 30 episodes")
    print("=" * 80)

    slip_probs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]

    configs = {
        "Standard-PS (K=15)": (StandardPrioritizedSweepingAgent, {"backup_budget": 15}),
        "Standard-PS (K=100)": (StandardPrioritizedSweepingAgent, {"backup_budget": 100}),
        "SL-PS (Proposed, K=15)": (SLPSAgent, {"backup_budget": 15, "beta_scale": 5.0, "alpha": 0.5, "eta": 0.1})
    }

    results = {name: {} for name in configs}

    for slip in slip_probs:
        print(f"\nEvaluating slip probability epsilon_slip = {slip:.2f}:")
        for name, (cls_obj, kwargs) in configs.items():
            res = evaluate_agent_on_env(cls_obj, kwargs, slip_prob=slip, num_episodes=30)
            results[name][str(slip)] = res
            print(f"  [{name:22s}] Latency: {res['mean_latency']:6.2f} +/- {res['std_latency']:4.2f}s | "
                  f"Trap Rate: {res['braess_trap_rate']:5.1f}% ({res['total_traps_taken']}/{res['total_chords_encountered']}) | "
                  f"Slips/ep: {res['mean_slips_per_ep']:.1f}")

    # Save JSON results
    out_json = "stochastic_transition_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to {out_json}")

    # Generate Figure 4
    plot_stochastic_transitions(results, slip_probs)


def plot_stochastic_transitions(results: dict, slip_probs: list):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "lines.linewidth": 1.8,
        "lines.markersize": 6
    })

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), dpi=300)

    colors = {
        "Standard-PS (K=15)": "#e74c3c",       # Crimson Red
        "Standard-PS (K=100)": "#e67e22",      # Pumpkin Orange
        "SL-PS (Proposed, K=15)": "#2980b9"    # Deep Blue
    }
    markers = {
        "Standard-PS (K=15)": "s",
        "Standard-PS (K=100)": "^",
        "SL-PS (Proposed, K=15)": "o"
    }

    x_vals = np.array(slip_probs)

    # -------------------------------------------------------------
    # (a) Cumulative Travel Latency vs Slip Probability
    # -------------------------------------------------------------
    ax = axes[0]
    for name, data_dict in results.items():
        means = [data_dict[str(s)]["mean_latency"] for s in slip_probs]
        stds = [data_dict[str(s)]["std_latency"] for s in slip_probs]
        ax.plot(x_vals, means, marker=markers[name], color=colors[name], label=name)
        ax.fill_between(x_vals, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                        color=colors[name], alpha=0.15)

    ax.set_title("(a) Cumulative Latency vs. Actuator Slip $\\epsilon_{\\mathrm{slip}}$")
    ax.set_xlabel("Slip Probability $\\epsilon_{\\mathrm{slip}}$")
    ax.set_ylabel("Travel Latency (s)")
    ax.set_xticks(x_vals)
    ax.set_ylim(8.5, 19.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left")

    # -------------------------------------------------------------
    # (b) Braess Trap Exploitation Rate (%) vs Slip Probability
    # -------------------------------------------------------------
    ax = axes[1]
    for name, data_dict in results.items():
        trap_rates = [data_dict[str(s)]["braess_trap_rate"] for s in slip_probs]
        ax.plot(x_vals, trap_rates, marker=markers[name], color=colors[name], label=name)

    ax.axhline(0.0, color="#27ae60", linestyle=":", linewidth=1.5, label="Optimal Immunity (0%)")
    ax.set_title("(b) Braess Trap Exploitation Rate (%)")
    ax.set_xlabel("Slip Probability $\\epsilon_{\\mathrm{slip}}$")
    ax.set_ylabel("Trap Encounter Rate (%)")
    ax.set_xticks(x_vals)
    ax.set_ylim(-5, 95)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper right")

    # -------------------------------------------------------------
    # (c) Relative Latency Overhead Delta C(eps) = C(eps) - C(0)
    # -------------------------------------------------------------
    ax = axes[2]
    for name, data_dict in results.items():
        base_lat = data_dict["0.0"]["mean_latency"]
        delta_lat = [data_dict[str(s)]["mean_latency"] - base_lat for s in slip_probs]
        ax.plot(x_vals, delta_lat, marker=markers[name], color=colors[name], label=name)

    ax.set_title(r"(c) Excess Delay $\Delta \bar{C} = \bar{C}(\epsilon) - \bar{C}(0)$")
    ax.set_xlabel("Slip Probability $\\epsilon_{\\mathrm{slip}}$")
    ax.set_ylabel("Added Travel Time (s)")
    ax.set_xticks(x_vals)
    ax.set_ylim(-1.4, 1.4)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left")

    # -------------------------------------------------------------
    # (d) Robustness Profile Under Severe Slip (eps_slip = 0.25)
    # -------------------------------------------------------------
    ax = axes[3]
    agent_names = list(results.keys())
    lat_025 = [results[n]["0.25"]["mean_latency"] for n in agent_names]
    err_025 = [results[n]["0.25"]["std_latency"] for n in agent_names]
    trap_025 = [results[n]["0.25"]["braess_trap_rate"] for n in agent_names]

    bar_pos = np.arange(len(agent_names))
    bar_width = 0.55
    bar_colors = [colors[n] for n in agent_names]

    bars = ax.bar(bar_pos, lat_025, yerr=err_025, width=bar_width, color=bar_colors,
                  edgecolor="black", linewidth=1.0, capsize=4, alpha=0.85)

    # Add text labels on top of bars
    for idx, (bar, t_rate) in enumerate(zip(bars, trap_025)):
        top_y = bar.get_height() + err_025[idx]
        ax.text(bar.get_x() + bar.get_width() / 2.0, top_y + 0.4,
                f"Trap: {t_rate:.1f}%\n$\\bar{{C}}$={lat_025[idx]:.2f}s",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    ax.set_title(r"(d) Severe Slip Regime ($\epsilon_{\mathrm{slip}} = 0.25$)")
    ax.set_xticks(bar_pos)
    ax.set_xticklabels(["Std-PS\n(K=15)", "Std-PS\n(K=100)", "SL-PS\n(Proposed)"], rotation=0)
    ax.set_ylabel("Travel Latency (s)")
    ax.set_ylim(0, 22.0)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    out_fig = "fig4_stochastic_transitions.png"
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Figure saved to {out_fig}")

    # Copy to artifact directory
    artifact_dir = r"C:\Users\ADMIN\.gemini\antigravity-cli\brain\94cfebc3-0f15-4ab1-b2a6-1c0c4c4a689a"
    if os.path.exists(artifact_dir):
        dest_fig = os.path.join(artifact_dir, "fig4_stochastic_transitions.png")
        shutil.copy2(out_fig, dest_fig)
        print(f"[OK] Figure copied to artifact directory: {dest_fig}")


if __name__ == "__main__":
    run_stochastic_transition_study()
