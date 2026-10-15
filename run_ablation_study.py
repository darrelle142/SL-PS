"""
run_ablation_study.py
=====================
Systematic Ablation Benchmark for Spectral-Laplacian Prioritized Sweeping (SL-PS):
1. Dimension 1: Role of Spectral Bellman Penalty (w/o T_SL, Priority-Only like GTA-PS)
2. Dimension 2: Contribution of SLEM Spectral Gap Modulation (sigma(P) vs Fixed beta)
3. Dimension 3: Directional Resolvent Balance (alpha in [0, 1]: Forward vs Backward)
4. Dimension 4: Regularization Sensitivity (eta in {0.01, 0.05, 0.1, 0.2, 0.5, 1.0})
5. Dimension 5: Planning Budget Efficiency (K in {0, 5, 10, 15, 30, 50, 100})

Outputs:
- ablation_results.json: Raw quantitative benchmark data
- fig3_ablation_study.png: 300 DPI, multi-panel APA 7.0 figure for LaTeX inclusion
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
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


def evaluate_agent(agent, env: CongestedGridEnv, num_episodes: int = 30) -> dict:
    total_travel_times = []
    total_steps = []
    braess_traps_taken = 0
    total_chords_encountered = 0
    decision_times = []

    for ep in range(num_episodes):
        state, info = env.reset(seed=1000 + ep)
        done = False
        ep_latency = 0.0
        steps = 0

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
            ep_latency += latency
            steps += 1
            state = next_state

        total_travel_times.append(ep_latency)
        total_steps.append(steps)

    trap_rate = (braess_traps_taken / max(1, total_chords_encountered)) * 100.0

    return {
        "mean_latency": float(np.mean(total_travel_times)),
        "std_latency": float(np.std(total_travel_times)),
        "mean_steps": float(np.mean(total_steps)),
        "braess_trap_rate": float(trap_rate),
        "total_traps_taken": int(braess_traps_taken),
        "mean_decision_ms": float(np.mean(decision_times))
    }


def run_full_ablation():
    print("=" * 80)
    print("STARTING COMPREHENSIVE ABLATION BENCHMARK FOR SL-PS (30 EPISODES)")
    print("=" * 80)

    env = CongestedGridEnv(m=3, n=6, a=1.0, b=1.0, q=1.0, p_chord=0.4, noise_std=0.02, seed=42)

    results = {}

    # -------------------------------------------------------------
    # 1. Architectural Component Ablation (Table 2 data)
    # -------------------------------------------------------------
    print("\n--- 1. Evaluating Architectural Variants ---")

    variants = {
        "Full SL-PS (Proposed)": SLPSAgent(env, backup_budget=15, beta_scale=6.0, eta=0.1, use_spectral_bellman=True, use_slem=True, alpha=0.5),
        "w/o Spectral Bellman (Priority-Only)": SLPSAgent(env, backup_budget=15, beta_scale=6.0, eta=0.1, use_spectral_bellman=False, use_slem=True, alpha=0.5),
        "w/o SLEM Modulation (Fixed beta)": SLPSAgent(env, backup_budget=15, beta_scale=6.0, eta=0.1, use_spectral_bellman=True, use_slem=False, alpha=0.5),
        "Forward-Only Priority (alpha=0.0)": SLPSAgent(env, backup_budget=15, beta_scale=6.0, eta=0.1, use_spectral_bellman=True, use_slem=True, alpha=0.0),
        "Backward-Only Priority (alpha=1.0)": SLPSAgent(env, backup_budget=15, beta_scale=6.0, eta=0.1, use_spectral_bellman=True, use_slem=True, alpha=1.0),
        "Standard-PS Baseline (K=15)": StandardPrioritizedSweepingAgent(env, backup_budget=15),
    }

    results["architecture_variants"] = {}
    for name, ag in variants.items():
        print(f"  Testing: {name} ...", end=" ", flush=True)
        res = evaluate_agent(ag, env, num_episodes=30)
        results["architecture_variants"][name] = res
        print(f"Latency = {res['mean_latency']:.2f} ± {res['std_latency']:.2f}s | Trap Rate = {res['braess_trap_rate']:.1f}%")

    # -------------------------------------------------------------
    # 2. Sensitivity to Directional Balance alpha in [0.0, 1.0]
    # -------------------------------------------------------------
    print("\n--- 2. Evaluating Directional Resolvent Balance alpha ---")
    alpha_sweep = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    results["alpha_sweep"] = {}
    for a_val in alpha_sweep:
        ag = SLPSAgent(env, backup_budget=15, beta_scale=6.0, eta=0.1, use_spectral_bellman=True, use_slem=True, alpha=a_val)
        res = evaluate_agent(ag, env, num_episodes=30)
        results["alpha_sweep"][str(a_val)] = res
        print(f"  alpha = {a_val:.1f} -> Latency = {res['mean_latency']:.2f}s | Trap Rate = {res['braess_trap_rate']:.1f}%")

    # -------------------------------------------------------------
    # 3. Sensitivity to Regularization Parameter eta
    # -------------------------------------------------------------
    print("\n--- 3. Evaluating Regularization Parameter eta ---")
    eta_sweep = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    results["eta_sweep"] = {}
    for eta_val in eta_sweep:
        ag = SLPSAgent(env, backup_budget=15, beta_scale=6.0, eta=eta_val, use_spectral_bellman=True, use_slem=True, alpha=0.5)
        res = evaluate_agent(ag, env, num_episodes=30)
        results["eta_sweep"][str(eta_val)] = res
        print(f"  eta = {eta_val:.2f} -> Latency = {res['mean_latency']:.2f}s | Trap Rate = {res['braess_trap_rate']:.1f}%")

    # -------------------------------------------------------------
    # 4. Budget Efficiency Sweep: K in {0, 5, 10, 15, 30, 50, 100}
    # -------------------------------------------------------------
    print("\n--- 4. Evaluating Budget Efficiency K (Standard-PS vs SL-PS) ---")
    K_sweep = [0, 5, 10, 15, 30, 50, 100]
    results["budget_sweep"] = {"K": K_sweep, "std_ps": [], "sl_ps": []}
    for k in K_sweep:
        ag_std = StandardPrioritizedSweepingAgent(env, backup_budget=k)
        res_std = evaluate_agent(ag_std, env, num_episodes=30)
        results["budget_sweep"]["std_ps"].append(res_std)

        ag_sl = SLPSAgent(env, backup_budget=k, beta_scale=6.0, eta=0.1, use_spectral_bellman=True, use_slem=True, alpha=0.5)
        res_sl = evaluate_agent(ag_sl, env, num_episodes=30)
        results["budget_sweep"]["sl_ps"].append(res_sl)

        print(f"  K = {k:3d} | Std-PS: Lat={res_std['mean_latency']:.2f}s, Trap={res_std['braess_trap_rate']:4.1f}% | SL-PS: Lat={res_sl['mean_latency']:.2f}s, Trap={res_sl['braess_trap_rate']:4.1f}%")

    # Save raw JSON results
    out_json = r"C:\Users\ADMIN\OneDrive\Máy tính\agents\ablation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw quantitative results to {out_json}")

    # -------------------------------------------------------------
    # 5. Generate APA 7.0 Multi-Panel Publication Figure
    # -------------------------------------------------------------
    plot_ablation_figure(results, r"C:\Users\ADMIN\OneDrive\Máy tính\agents\fig3_ablation_study.png")

    return results


def plot_ablation_figure(results, output_path):
    # Publication-ready APA 7.0 style configuration
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.titlesize": 12
    })

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), dpi=300)
    fig.subplots_adjust(hspace=0.35, wspace=0.28, left=0.08, right=0.96, top=0.92, bottom=0.08)

    # Palette (Tol / Colorblind-safe)
    c_blue = "#0077BB"
    c_cyan = "#33BBEE"
    c_teal = "#009988"
    c_orange = "#EE7733"
    c_red = "#CC3311"
    c_grey = "#888888"

    # -------------------------------------------------------------
    # Panel (a): Component Ablation Bar Chart (Latency & Trap Rate)
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
    arch_data = results["architecture_variants"]
    names_short = [
        "Full SL-PS",
        "w/o Bellman Pen.\n(Priority-Only)",
        "w/o SLEM\n(Fixed β)",
        "Forward-Only\n(α=0)",
        "Backward-Only\n(α=1)",
        "Standard-PS\n(K=15)"
    ]
    latencies = [d["mean_latency"] for d in arch_data.values()]
    stds = [d["std_latency"] for d in arch_data.values()]
    trap_rates = [d["braess_trap_rate"] for d in arch_data.values()]

    x = np.arange(len(names_short))
    width = 0.42

    bars = ax_a.bar(x, latencies, width, yerr=stds, capsize=3.5, color=c_blue, edgecolor="black", lw=0.8, alpha=0.9, label="Mean Latency (s)")
    ax_a.set_ylabel("Cumulative Travel Latency (s)", color=c_blue, fontweight="bold")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(names_short, rotation=25, ha="right", fontsize=7.8)
    ax_a.set_ylim(0, 26)
    ax_a.grid(axis="y", linestyle=":", alpha=0.5)

    # Secondary axis for trap rate
    ax_a_twin = ax_a.twinx()
    pts = ax_a_twin.plot(x, trap_rates, "s--", color=c_red, lw=1.8, markersize=6, label="Braess Trap Rate (%)")
    ax_a_twin.set_ylabel("Braess Trap Rate (%)", color=c_red, fontweight="bold")
    ax_a_twin.set_ylim(-5, 65)

    for i, tr in enumerate(trap_rates):
        ax_a_twin.text(i, tr + 2.5, f"{tr:.0f}%", ha="center", va="bottom", color=c_red, fontsize=7.8, fontweight="bold")

    ax_a.set_title("(a) Architectural Component Degradation", fontweight="bold", pad=8)

    # -------------------------------------------------------------
    # Panel (b): Directional Resolvent Balance alpha
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    alpha_data = results["alpha_sweep"]
    alphas = [float(k) for k in alpha_data.keys()]
    alpha_lats = [d["mean_latency"] for d in alpha_data.values()]
    alpha_stds = [d["std_latency"] for d in alpha_data.values()]

    ax_b.plot(alphas, alpha_lats, "o-", color=c_teal, lw=2.0, markersize=6.5, label="SL-PS Latency")
    ax_b.fill_between(alphas, np.array(alpha_lats) - np.array(alpha_stds), np.array(alpha_lats) + np.array(alpha_stds), color=c_teal, alpha=0.18)
    ax_b.axvline(0.5, color=c_orange, linestyle="--", lw=1.5, label="Default Balance (α=0.5)")
    ax_b.set_xlabel("Directional Balance Parameter α (0: Forward, 1: Backward)")
    ax_b.set_ylabel("Cumulative Travel Latency (s)")
    ax_b.set_title("(b) Bi-Directional Resolvent Tuning (α)", fontweight="bold", pad=8)
    ax_b.grid(True, linestyle=":", alpha=0.5)
    ax_b.legend(loc="upper right", frameon=True)

    # -------------------------------------------------------------
    # Panel (c): Regularization Parameter Sensitivity eta
    # -------------------------------------------------------------
    ax_c = axes[1, 0]
    eta_data = results["eta_sweep"]
    etas = [float(k) for k in eta_data.keys()]
    eta_lats = [d["mean_latency"] for d in eta_data.values()]
    eta_stds = [d["std_latency"] for d in eta_data.values()]

    ax_c.plot(etas, eta_lats, "d-", color="#882255", lw=2.0, markersize=6.5, label="SL-PS Latency")
    ax_c.fill_between(etas, np.array(eta_lats) - np.array(eta_stds), np.array(eta_lats) + np.array(eta_stds), color="#882255", alpha=0.18)
    ax_c.axvline(0.1, color=c_orange, linestyle="--", lw=1.5, label="Theoretical Optimum (η=0.1)")
    ax_c.set_xscale("log")
    ax_c.set_xlabel("Green Regularization Parameter η (log scale)")
    ax_c.set_ylabel("Cumulative Travel Latency (s)")
    ax_c.set_title("(c) Resolvent Damping Sensitivity (η)", fontweight="bold", pad=8)
    ax_c.grid(True, linestyle=":", alpha=0.5)
    ax_c.legend(loc="upper left", frameon=True)

    # -------------------------------------------------------------
    # Panel (d): Planning Budget Efficiency K Sweep
    # -------------------------------------------------------------
    ax_d = axes[1, 1]
    b_data = results["budget_sweep"]
    k_vals = b_data["K"]
    std_lats = [d["mean_latency"] for d in b_data["std_ps"]]
    sl_lats = [d["mean_latency"] for d in b_data["sl_ps"]]

    ax_d.plot(k_vals, std_lats, "s--", color=c_red, lw=2.0, markersize=6.0, label="Standard-PS (Peng & Williams)")
    ax_d.plot(k_vals, sl_lats, "o-", color=c_blue, lw=2.2, markersize=6.5, label="SL-PS (Proposed)")
    ax_d.axhline(10.6, color=c_grey, linestyle=":", lw=1.2, label="Optimal Evading Baseline")
    ax_d.set_xlabel("Bellman Backup Budget K per Step")
    ax_d.set_ylabel("Cumulative Travel Latency (s)")
    ax_d.set_title("(d) Budget Efficiency: Standard-PS vs. SL-PS", fontweight="bold", pad=8)
    ax_d.grid(True, linestyle=":", alpha=0.5)
    ax_d.legend(loc="upper right", frameon=True)

    fig.suptitle("Figure 3: Comprehensive Ablation Study of SL-PS on Congested Dynamic Grid Networks", fontsize=12.5, fontweight="bold", y=0.98)

    plt.savefig(output_path, dpi=300, facecolor="white", edgecolor="none")
    plt.close()
    print(f"Publication-ready APA 7.0 Figure successfully saved to {output_path}")


if __name__ == "__main__":
    run_full_ablation()
