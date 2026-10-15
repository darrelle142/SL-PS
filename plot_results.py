"""
plot_results.py
===============
Generates publication-quality figures for idea.tex and walkthrough:
1. fig1_potential_and_braess_chords.png: Potential field heatmap and Braess chord vectors.
2. fig2_benchmark_comparison.png: Empirical benchmark performance across algorithms.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from spectral_braess_core import GridNetwork


def plot_potential_field():
    m, n = 3, 6
    grid = GridNetwork(m=m, n=n, a=1.0, q=1.0)
    chords_info = grid.find_all_chords()
    braess_chords = chords_info["braess_capable"]
    shortcuts = chords_info["shortcut"]

    phi_matrix = np.zeros((m + 1, n + 1))
    for i in range(m + 1):
        for j in range(n + 1):
            phi_matrix[i, j] = grid.phi_base[grid.node_to_idx[(i, j)]]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    # Plot potential contours / heatmap
    c = ax.imshow(phi_matrix, cmap="YlGnBu", origin="upper", aspect="equal", extent=[-0.5, n + 0.5, m + 0.5, -0.5])
    cbar = plt.colorbar(c, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Electrical Potential $\\phi(i, j)$", fontsize=11)

    # Plot base grid lines
    for i in range(m + 1):
        for j in range(n):
            ax.annotate("", xy=(j + 1, i), xytext=(j, i),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1, alpha=0.5))
    for i in range(m):
        for j in range(n + 1):
            ax.annotate("", xy=(j, i + 1), xytext=(j, i),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1, alpha=0.5))

    # Plot node dots
    for i in range(m + 1):
        for j in range(n + 1):
            ax.plot(j, i, "o", color="#333333", markersize=6)

    # Mark Source and Sink
    ax.plot(0, 0, "s", color="lime", markersize=12, label="Source $s=(0,0)$", zorder=5)
    ax.plot(n, m, "s", color="magenta", markersize=12, label="Sink $t=(m,n)$", zorder=5)

    # Plot sample Braess chords (Red dashed)
    for idx, (u, v, V_uv, k) in enumerate(braess_chords[:6]):
        lbl = "Braess Chords ($V_{uv} < 0, k > 0$)" if idx == 0 else ""
        ax.annotate("", xy=(v[1], v[0]), xytext=(u[1], u[0]),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2.2, ls="--",
                                    connectionstyle="arc3,rad=-0.2"), label=lbl, zorder=6)

    # Plot sample Shortcuts (Blue solid)
    for idx, (u, v, V_uv, k) in enumerate(shortcuts[:3]):
        lbl = "Benign Shortcuts ($V_{uv} > 0$)" if idx == 0 else ""
        ax.annotate("", xy=(v[1], v[0]), xytext=(u[1], u[0]),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=1.8,
                                    connectionstyle="arc3,rad=0.2"), label=lbl, zorder=6)

    ax.set_title(f"Electrical Potential Field $\\phi$ & Braess Chords on $G_{{{m},{n}}}$", fontsize=13, fontweight="bold")
    ax.set_xlabel("Grid Coordinate $j$ (Columns)", fontsize=11)
    ax.set_ylabel("Grid Coordinate $i$ (Rows)", fontsize=11)
    # Custom legend handles
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", label="Source $s=(0,0)$", markerfacecolor="lime", markersize=10),
        Line2D([0], [0], marker="s", color="w", label="Sink $t=(m,n)$", markerfacecolor="magenta", markersize=10),
        Line2D([0], [0], color="red", lw=2, ls="--", label="Braess Chords ($V_{uv} < 0$)"),
        Line2D([0], [0], color="blue", lw=2, label="Shortcuts ($V_{uv} > 0$)")
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig("fig1_potential_and_braess_chords.png")
    plt.close()
    print("Saved fig1_potential_and_braess_chords.png")


def plot_benchmark_comparison():
    algorithms = ["Greedy-Dijkstra", "Standard-PS", "GTA-PS", "SL-PS\n(Proposed)"]
    # Grid 3x6 results
    mean_lat_3x6 = [11.27, 12.42, 12.30, 11.17]
    std_lat_3x6 = [0.89, 3.62, 3.70, 0.73]
    trap_rate_3x6 = [0.0, 50.0, 50.0, 0.0]

    # Grid 4x8 results
    mean_lat_4x8 = [14.46, 14.48, 14.51, 14.43]
    std_lat_4x8 = [1.06, 2.32, 2.24, 0.68]
    trap_rate_4x8 = [0.0, 20.0, 20.0, 0.0]

    x = np.arange(len(algorithms))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Subplot 1: Travel Time (Mean +- Std)
    bars1 = ax1.bar(x - width/2, mean_lat_3x6, width, yerr=std_lat_3x6, capsize=5, label="Grid 3x6", color="#4C72B0")
    bars2 = ax1.bar(x + width/2, mean_lat_4x8, width, yerr=std_lat_4x8, capsize=5, label="Grid 4x8", color="#55A868")

    ax1.set_ylabel("Mean Travel Time (s)", fontsize=11)
    ax1.set_title("Navigation Latency & Variance Across Planners", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, fontsize=9)
    ax1.legend(loc="upper left")
    ax1.set_ylim(8, 18)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    # Subplot 2: Braess Trap Exploitation Rate (%)
    bars3 = ax2.bar(x - width/2, trap_rate_3x6, width, label="Grid 3x6", color="#C44E52")
    bars4 = ax2.bar(x + width/2, trap_rate_4x8, width, label="Grid 4x8", color="#8172B3")

    ax2.set_ylabel("Braess Trap Encounter Rate (%)", fontsize=11)
    ax2.set_title("Braess Deceptive Trap Exploitation Rate (%)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, fontsize=9)
    ax2.legend(loc="upper right")
    ax2.set_ylim(0, 60)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    # Annotate zero trap rates for SL-PS
    ax2.annotate("0% Traps\n(Immune)", xy=(3 - width/2, 2), xytext=(3 - width/2, 12),
                 ha="center", fontsize=8, fontweight="bold", color="darkgreen",
                 arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5))

    plt.tight_layout()
    plt.savefig("fig2_benchmark_comparison.png")
    plt.close()
    print("Saved fig2_benchmark_comparison.png")


if __name__ == "__main__":
    plot_potential_field()
    plot_benchmark_comparison()
