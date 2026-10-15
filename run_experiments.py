"""
run_experiments.py
==================
Benchmarking and Empirical Evaluation of Model-Based Dynamic Programming Planners:
1. Greedy-Dijkstra (Baseline)
2. Standard-PS (Peng & Williams, 1993)
3. GTA-PS (Anonymous, 2025)
4. SL-PS (Proposed Spectral-Laplacian Prioritized Sweeping)
"""

import numpy as np
import time
from typing import Dict, List
from grid_braess_env import CongestedGridEnv
from planning_agents import (
    GreedyDijkstraAgent,
    StandardPrioritizedSweepingAgent,
    GTAPrioritizedSweepingAgent,
    SLPSAgent
)


def evaluate_agent(agent, env: CongestedGridEnv, num_episodes: int = 30) -> Dict[str, float]:
    total_travel_times = []
    total_steps = []
    braess_traps_taken = 0
    total_chords_encountered = 0

    for ep in range(num_episodes):
        state, info = env.reset(seed=1000 + ep)
        done = False
        ep_latency = 0.0
        steps = 0

        # Reset agent value function per episode for fair online replanning comparison
        if hasattr(agent, "V"):
            agent.V = {
                (i, j): float((env.m - i) + (env.n - j)) * env.grid.b
                for i in range(env.m + 1)
                for j in range(env.n + 1)
            }
            agent.V[(env.m, env.n)] = 0.0

        while not done:
            action = agent.select_action(state, info["active_chord"])

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
        "braess_trap_rate": trap_rate,
        "total_traps_taken": braess_traps_taken
    }


def run_benchmark():
    grid_configs = [
        {"name": "Grid 3x6 (Aspect Ratio 2:1)", "m": 3, "n": 6, "p_chord": 0.4},
        {"name": "Grid 4x8 (Aspect Ratio 2:1)", "m": 4, "n": 8, "p_chord": 0.4}
    ]

    num_episodes = 30
    print("=" * 80)
    print("STARTING BENCHMARK: SPECTRAL-LAPLACIAN DYNAMIC PROGRAMMING (SL-PS)")
    print(f"Number of episodes per configuration: {num_episodes}")
    print("=" * 80)

    for cfg in grid_configs:
        print(f"\n>>> Running Configuration: {cfg['name']} (m={cfg['m']}, n={cfg['n']}, p_chord={cfg['p_chord']})")
        env = CongestedGridEnv(
            m=cfg["m"],
            n=cfg["n"],
            a=1.0,
            b=1.0,
            q=1.0,
            p_chord=cfg["p_chord"],
            noise_std=0.02,
            seed=42
        )

        agents = [
            GreedyDijkstraAgent(env),
            StandardPrioritizedSweepingAgent(env, backup_budget=15),
            GTAPrioritizedSweepingAgent(env, backup_budget=15),
            SLPSAgent(env, backup_budget=15, beta_scale=6.0)
        ]

        print("-" * 80)
        print(f"{'Algorithm':<28} | {'Mean Travel Time':<18} | {'Steps':<8} | {'Braess Traps Taken':<18}")
        print("-" * 80)

        for agent in agents:
            t0 = time.time()
            metrics = evaluate_agent(agent, env, num_episodes=num_episodes)
            elapsed = time.time() - t0

            name = agent.name
            latency_str = f"{metrics['mean_latency']:.2f} ± {metrics['std_latency']:.2f}"
            steps_str = f"{metrics['mean_steps']:.1f}"
            trap_str = f"{metrics['total_traps_taken']} ({metrics['braess_trap_rate']:.1f}%)"

            print(f"{name:<28} | {latency_str:<18} | {steps_str:<8} | {trap_str:<18}")

        print("-" * 80)


if __name__ == "__main__":
    run_benchmark()
