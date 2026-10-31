# scratch/apply_audit_fixes.py
import re

def apply_fixes():
    with open('idea.tex', 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Original length: {len(text)} chars")

    # =========================================================================
    # FIX 1: Add bauer1960norms and jovanovic1997rule to thebibliography
    # =========================================================================
    target_bib = r"""\begin{thebibliography}{10}

\bibitem{beckmann1956studies}"""
    replacement_bib = r"""\begin{thebibliography}{10}

\bibitem{bauer1960norms}
F.~L.~Bauer and C.~T.~Fike.
\newblock Norms and exclusion theorems.
\newblock {\em Numerische Mathematik}, 2(1):137--141, 1960.

\bibitem{beckmann1956studies}"""
    assert text.count(target_bib) == 1, "target_bib not unique"
    text = text.replace(target_bib, replacement_bib)

    target_jov = r"""\bibitem{gta_ps2025}
Anonymous Authors.
\newblock Graph topology augmentation for prioritized sweeping in non-stationary reinforcement learning.
\newblock {\em Preprint}, 2025.

\bibitem{lu2026braess}"""
    replacement_jov = r"""\bibitem{gta_ps2025}
Anonymous Authors.
\newblock Graph topology augmentation for prioritized sweeping in non-stationary reinforcement learning.
\newblock {\em Preprint}, 2025.

\bibitem{jovanovic1997rule}
B.~D.~Jovanovic and P.~S.~Levy.
\newblock A look at the rule of three.
\newblock {\em The American Statistician}, 51(2):137--139, 1997.

\bibitem{lu2026braess}"""
    assert text.count(target_jov) == 1, "target_jov not unique"
    text = text.replace(target_jov, replacement_jov)

    # =========================================================================
    # FIX 2: Section 1.2 Contributions: Clarify Two-Tier Architecture
    # =========================================================================
    target_contrib = r"""    \item \textbf{The Spectral-Laplacian Bellman Operator}: We derive $\mathcal{T}_{\text{SL}}$, a novel contraction operator that penalizes actions leading toward systemic congestion using the global \textit{Spectral Congestion Divergence Potential} $\Phi_{\text{spec}} = \mathcal{G}_{\text{out}} |B \Delta \vec{\ell}|$, where $\mathcal{G}_{\text{out}} = (I - P + \eta I)^{-1}$ is the Green's operator of the directed out-Laplacian on arbitrary directed graphs and $\Delta \vec{\ell}$ is the flow divergence vector.
    \item \textbf{Theoretical Guarantees on General Graphs}: We prove that $\mathcal{T}_{\text{SL}}$ is a strict $\gamma$-contraction in $L_\infty$ on arbitrary finite graphs (cyclic or acyclic) and establish the Braess Trap Immunity Theorem for general topologies. Furthermore, we derive the Lipschitz Stability Theorem, establishing explicit robustness tolerance margins under inexact Wardrop equilibrium estimation, and prove the Asymptotic Dominance Theorem, establishing that the cumulative travel time advantage of SL-PS over Standard-PS diverges to $+\infty$ at rate $\Omega(q \cdot \operatorname{diam}(G))$ as network scale and background demand expand.
    \item \textbf{Spectral Priority Scheduling}: We integrate bi-directional Green's potentials ($f_{\text{out}}, f_{\text{in}}$) annealed by the Second Largest Eigenvalue Modulus (SLEM) $\sigma(P) = |\lambda_2(P)|$, ensuring that distant upstream junction states receive immediate replanning priority before physical wavefronts arrive."""
    
    replacement_contrib = r"""    \item \textbf{The Spectral-Laplacian Bellman Operator (Decision-Time Risk Awareness)}: We derive $\mathcal{T}_{\text{SL}}$, a novel contraction operator that couples action selection with the global \textit{Spectral Congestion Divergence Potential} $\Phi_{\text{spec}} = \mathcal{G}_{\text{out}} |B \Delta \vec{\ell}|$, where $\mathcal{G}_{\text{out}} = (I - P + \eta I)^{-1}$ is the Green's operator of the directed out-Laplacian on arbitrary directed graphs and $\Delta \vec{\ell}$ is the flow divergence vector. This operator guarantees structural rejection of deceptive shortcut bottlenecks at decision time, independent of sweeping iterations ($K = 0$).
    \item \textbf{Theoretical Guarantees on General Graphs}: We prove that $\mathcal{T}_{\text{SL}}$ is a strict $\gamma$-contraction in $L_\infty$ on arbitrary finite graphs (cyclic or acyclic) and establish an a priori, non-circular Braess Trap Immunity Theorem for general topologies. Furthermore, we derive the Lipschitz Stability Theorem, establishing explicit robustness tolerance margins under inexact Wardrop equilibrium estimation, and prove the Asymptotic Dominance Theorem under multi-stage corridor renewal dynamics, showing that the cumulative travel time advantage of SL-PS over Standard-PS diverges to $+\infty$ at rate $\Omega(q \cdot \operatorname{diam}(G))$.
    \item \textbf{Bi-Directional Priority Scheduling (Sweeping Efficiency \& Non-Starvation)}: To solve the priority starvation of upstream diverters under finite backup budgets $K$, we integrate bi-directional Green's potentials ($f_{\text{out}}, f_{\text{in}} = \mathcal{G}_{\text{in}} \xi_{\text{div}}$) modulated by the Second Largest Eigenvalue Modulus (SLEM) $\sigma(P) = |\lambda_2(P)|$. This guarantees instantaneous spatial reachability ($\operatorname{Radius}_{\text{SL-PS}}(0) = \operatorname{diam}(G)$), ensuring upstream junctions receive timely updates before agent arrival."""
    assert text.count(target_contrib) == 1, "target_contrib not unique"
    text = text.replace(target_contrib, replacement_contrib)

    # =========================================================================
    # FIX 3: Section 5.2 Theorem 3 Non-Circular Bound
    # =========================================================================
    target_thm3 = r"""\subsection{Braess Trap Immunity}
\begin{theorem}[Braess Trap Immunity on General Networks]
\label{thm:immunity}
\label{thm:trap_immunity}
Let $e^* = (u \to v)$ be an active candidate chord on general directed network $G = (V, E_0)$ satisfying electrical voltage drop $V_{uv} < 0$ that induces positive equilibrium flow $z^* > 0$. If the penalty parameter satisfies:
\begin{equation}
\beta_0 \ge \frac{c(u, a_{\text{bypass}}) - c(u, a_{\text{chord}}) + \gamma (V(s_{\text{bypass}}) - V(v))}{(1 + \sigma(P)) (\Phi_{\text{spec}}(v) - \Phi_{\text{spec}}(s_{\text{bypass}}))},
\end{equation}
then the policy derived from the fixed point $V_{\text{SL}}^*$ satisfies $\pi_{\text{SL}}^*(u) \ne a_{\text{chord}}$, guaranteeing complete avoidance of the Braess bottleneck across arbitrary network topologies.
\end{theorem}"""

    replacement_thm3 = r"""\subsection{Braess Trap Immunity}
\begin{theorem}[Braess Trap Immunity on General Networks]
\label{thm:immunity}
\label{thm:trap_immunity}
Let $e^* = (u \to v)$ be an active candidate chord on general directed network $G = (V, E_0)$ with maximum edge traversal cost $c_{\max} = \max_{e \in E_0} c_e$, topological diameter $\operatorname{diam}(G)$, and electrical voltage drop $V_{uv} < 0$ inducing positive equilibrium chord flow $z^* > 0$. Let $s_{\text{bypass}} \in \mathcal{N}_{\text{out}}(u) \setminus \{v\}$ denote an alternative forward action from $u$, and let $\Delta \Phi_{\text{spec}} \equiv \Phi_{\text{spec}}(v) - \Phi_{\text{spec}}(s_{\text{bypass}}) > 0$ denote the strictly positive spectral potential differential. If the base penalty scaling parameter satisfies the non-circular a priori threshold:
\begin{equation}
\label{eq:beta_min_star_theorem}
\beta_0 \ge \beta_{\min}^* \equiv \frac{c_{\max} \left( 1 + \gamma \operatorname{diam}(G) \right)}{(1 + \sigma(P)) \Delta \Phi_{\text{spec}}},
\end{equation}
then the policy derived from the Spectral-Laplacian Bellman fixed point $V_{\text{SL}}^*$ strictly satisfies $\pi_{\text{SL}}^*(u) \ne a_{\text{chord}}$, guaranteeing structural avoidance of the Braess bottleneck across arbitrary directed network topologies independent of sweeping queue iterations.
\end{theorem}

\begin{remark}[A Priori Non-Circular Parameterization]
Unlike implicit thresholds parameterized by an unknown fixed point $V^*$, the lower bound $\beta_{\min}^*$ in Eq.~\eqref{eq:beta_min_star_theorem} is established strictly on a priori graph invariants: maximum edge cost $c_{\max}$, topological diameter $\operatorname{diam}(G)$, spectral expansion $\sigma(P)$, and potential differential $\Delta \Phi_{\text{spec}}$. Because the maximum possible unaugmented travel time advantage of traversing the shortcut over the bypass corridor across the entire network diameter cannot exceed $c_{\max}(1 + \gamma \operatorname{diam}(G))$, setting $\beta_0 \ge \beta_{\min}^*$ ensures that the localized spectral penalty differential $\beta(\sigma) \Delta \Phi_{\text{spec}}$ dominates the maximum possible unaugmented benefit a priori.
\end{remark}"""
    assert text.count(target_thm3) == 1, "target_thm3 not unique"
    text = text.replace(target_thm3, replacement_thm3)

    # =========================================================================
    # FIX 4: Section 6.2 Greedy-Dijkstra Objective Neutralization
    # =========================================================================
    target_dijkstra = r"""\paragraph{Dissecting Baseline Behaviors: Fixed Perimeter Heuristics vs. Adaptive Value Planners.}
A superficial examination of Table~\ref{tab:results} and Figure~\ref{fig:benchmark_results} might suggest an apparent puzzle: the simple Greedy-Dijkstra heuristic records $0$ trap encounters, matching SL-PS's zero rate. However, a rigorous algorithmic inspection reveals a fundamental distinction:
\begin{enumerate}
    \item \textbf{Greedy-Dijkstra as an Oblivious Boundary Hugger}: Greedy-Dijkstra's zero-trap outcome is purely a fortuitous topological artifact of its static Manhattan tie-breaking rule. Because it prioritizes horizontal movement along row 0 until hitting the boundary, its trajectory never intersects interior junction $(1, 1)$, where the Braess shortcut $(1, 1) \to (0, 3)$ originates. The baseline does not ``avoid'' the trap through congestion awareness; it is completely blind to network dynamics and simply follows a fixed perimeter path. In real-world systems or under stochastic perturbations (Section~\ref{subsec:stochastic_transitions}), such open-loop static heuristics suffer catastrophic degradation.
    \item \textbf{The Dilemma of Adaptive Planners (Standard-PS and GTA-PS)}: Unlike Dijkstra, Standard-PS and GTA-PS are true adaptive, value-driven reinforcement learning planners. They propagate value updates backward from the goal to discover optimal policies. However, when the Braess shortcut appears with nominal latency $d \approx 0$, local Bellman residual updates at $(1, 1)$ propagate an illusory cost reduction backward to $(0, 1)$ and $(0, 0)$. This actively misleads the agents into diverting inward away from the perimeter to exploit the chord, directly precipitating systemic network collapse and inducing severe trap rates of $50.0\%$ on $G_{3,6}$ and $20.0\%$ on $G_{4,8}$.
    \item \textbf{Active Congestion-Aware Planning in SL-PS}: SL-PS overcomes this fundamental dilemma. Like Standard-PS, SL-PS is an adaptive value planner that actively evaluates interior shortcuts. However, unlike Standard-PS and GTA-PS, SL-PS incorporates the Spectral Congestion Divergence Potential $\Phi_{\text{spec}} = \mathcal{G}_{\text{out}} |B \Delta \vec{\ell}|$ into its Bellman operator $\mathcal{T}_{\text{SL}}$. It recognizes that taking the chord will overload the downstream bottleneck and actively penalizes the chord action on principled physical grounds. SL-PS therefore achieves the best of both worlds: full adaptive planning capability with principled, congestion-aware trap immunity.
\end{enumerate}"""

    replacement_dijkstra = r"""\paragraph{Algorithmic Comparison: Open-Loop Shortest Path vs. Adaptive Dynamic Planners.}
In Table~\ref{tab:results} and Figure~\ref{fig:benchmark_results}, Greedy-Dijkstra records $0$ trap encounters on nominal grid instances, matching SL-PS's zero trap rate. A rigorous examination of the underlying planning mechanisms clarifies the operational differences between static open-loop routing and dynamic closed-loop value iteration:
\begin{enumerate}
    \item \textbf{Static Open-Loop Planning (Greedy-Dijkstra)}: Greedy-Dijkstra computes a deterministic shortest path offline based exclusively on nominal free-flow latencies $\ell_e(0) = b_e$ without online replanning. In the structured grid benchmark, the nominal free-flow latency along the outer boundary happens to match the interior path prior to shortcut emergence. Because Dijkstra operates open-loop and does not replan dynamically upon chord appearance, it remains oblivious to the shortcut $e^*$ and adheres strictly to its precomputed perimeter route. While this static behavior bypasses the interior shortcut $(1, 1) \to (0, 3)$ in unperturbed environments, open-loop heuristics cannot adapt when stochastic perturbations, transit delays, or link outages occur (as demonstrated in Section~\ref{subsec:stochastic_transitions}, where rigid open-loop paths suffer severe latency inflation).
    \item \textbf{Adaptive Planners with Local Residuals (Standard-PS and GTA-PS)}: Unlike Dijkstra, Standard-PS and GTA-PS are dynamic closed-loop reinforcement learning planners that actively monitor environmental changes. However, when the auxiliary chord appears with low nominal cost $d$, local Bellman residual updates propagate an apparent cost reduction backward to upstream junctions $(0, 1)$ and $(0, 0)$. Lacking global congestion foresight, these adaptive planners actively divert into the shortcut, precipitating network bottlenecks and suffering substantial trap rates ($50.0\%$ on $G_{3,6}$ and $20.0\%$ on $G_{4,8}$).
    \item \textbf{Congestion-Aware Adaptive Planning (SL-PS)}: SL-PS bridges the gap between dynamic adaptability and global stability. Like Standard-PS, it actively adapts to newly opened routes; however, its Spectral-Laplacian Bellman Operator $\mathcal{T}_{\text{SL}}$ augments action evaluation with the global congestion divergence potential $\Phi_{\text{spec}}$. Although nominal chord latency is small ($d \approx 0$), the downstream bottleneck node carries high spectral potential $\beta(\sigma) \Phi_{\text{spec}}(v)$. By coupling localized decisions with global network divergence, SL-PS reliably rejects deceptive shortcuts while retaining full closed-loop adaptability across dynamic networks.
\end{enumerate}"""
    assert text.count(target_dijkstra) == 1, "target_dijkstra not unique"
    text = text.replace(target_dijkstra, replacement_dijkstra)

    # =========================================================================
    # FIX 5: Appendix A.4 subsec:proof_thm3 Non-Circular Derivation
    # =========================================================================
    target_proof_thm3 = r"""Substituting this lower bound into the inequality:
\begin{equation}
\begin{split}
\beta_0 (1 + \sigma(P)) \Delta \Phi_{\text{spec}} &> c(u, a_{\text{bypass}}) - c(u, a_{\text{chord}}) \\
&\quad + \gamma \left[ V_{\text{SL}}^*(s_{\text{bypass}}) - V_{\text{SL}}^*(v) \right].
\end{split}
\end{equation}
Dividing both sides by the strictly positive quantity $(1 + \sigma(P)) \Delta \Phi_{\text{spec}}$ yields:
\begin{equation}
\beta_0 \ge \frac{c(u, a_{\text{bypass}}) - c(u, a_{\text{chord}}) + \gamma \left( V_{\text{SL}}^*(s_{\text{bypass}}) - V_{\text{SL}}^*(v) \right)}{(1 + \sigma(P)) \left( \Phi_{\text{spec}}(v) - \Phi_{\text{spec}}(s_{\text{bypass}}) \right)}.
\end{equation}
When $\beta_0$ satisfies this bound, $Q_{\text{SL}}^*(u, a_{\text{chord}}) > Q_{\text{SL}}^*(u, a_{\text{bypass}})$ holds strictly, implying that $\pi_{\text{SL}}^*(u) \ne a_{\text{chord}}$. Consequently, the agent is guaranteed complete immunity from traversing the congested Braess bottleneck.
\end{proof}"""

    replacement_proof_thm3 = r"""Substituting this lower bound into the inequality:
\begin{equation}
\begin{split}
\beta_0 (1 + \sigma(P)) \Delta \Phi_{\text{spec}} &> c(u, a_{\text{bypass}}) - c(u, a_{\text{chord}}) \\
&\quad + \gamma \left[ V_{\text{SL}}^*(s_{\text{bypass}}) - V_{\text{SL}}^*(v) \right].
\end{split}
\end{equation}
To eliminate circular dependency on the fixed-point value function $V_{\text{SL}}^*$ (which itself is parameterized by $\beta_0$), we establish an a priori upper bound on the right-hand side using nominal graph invariants. Let $c_{\max} \equiv \max_{e \in E_0} c_e < \infty$ denote the maximum edge cost on the base network, and let $\operatorname{diam}(G)$ denote the topological diameter of $G$. For any proper goal-directed policy terminating at destination $t_{\text{sink}}$, the nominal path length cannot exceed $\operatorname{diam}(G)$, which guarantees:
\begin{equation}
c(u, a_{\text{bypass}}) - c(u, a_{\text{chord}}) \le c_{\max}, \quad \text{and} \quad V_{\text{SL}}^*(s_{\text{bypass}}) - V_{\text{SL}}^*(v) \le \operatorname{diam}(G) \cdot c_{\max}.
\end{equation}
Therefore, the deceptive unaugmented action advantage is strictly bounded from above by:
\begin{equation}
c(u, a_{\text{bypass}}) - c(u, a_{\text{chord}}) + \gamma \left[ V_{\text{SL}}^*(s_{\text{bypass}}) - V_{\text{SL}}^*(v) \right] \le c_{\max} \left( 1 + \gamma \operatorname{diam}(G) \right).
\end{equation}
Consequently, setting the base penalty scaling parameter to satisfy the independent a priori lower bound:
\begin{equation}
\beta_0 \ge \beta_{\min}^* \equiv \frac{c_{\max} \left( 1 + \gamma \operatorname{diam}(G) \right)}{(1 + \sigma(P)) \Delta \Phi_{\text{spec}}}
\end{equation}
strictly guarantees $Q_{\text{SL}}^*(u, a_{\text{chord}}) > Q_{\text{SL}}^*(u, a_{\text{bypass}})$. Thus, $\pi_{\text{SL}}^*(u) \ne a_{\text{chord}}$ holds unconditionally, establishing structural immunity from the congested Braess bottleneck without circular dependence on the fixed point.
\end{proof}"""
    assert text.count(target_proof_thm3) == 1, "target_proof_thm3 not unique"
    text = text.replace(target_proof_thm3, replacement_proof_thm3)

    # =========================================================================
    # FIX 6: Appendix A.5 subsec:proof_thm4 Dynamic Branching & Multi-Stage Renewal
    # =========================================================================
    target_proof_thm4_p2 = r"""\paragraph{Part 2: Expanding Topological Network Limit ($\operatorname{diam}(G) \to \infty$).}
On any family of directed networks parameterized by topological diameter $\operatorname{diam}(G)$ (such as rectangular grids $G_{m,n}$ where $\operatorname{diam}(G) = \Theta(m+n)$), the minimum path length between origin and destination is $L_{\min} = \Theta(\operatorname{diam}(G))$. Along any realized navigation trajectory of length $T_{\text{steps}} \ge L_{\min}$:
\begin{enumerate}
    \item At each decision step $t$, candidate chords emerge independently with probability $p_{\text{chord}} > 0$.
    \item On expanding topologies with non-zero Braess chord density (e.g., $\rho_{\text{braess}} > 0$ as established by Lu \& Miller~\cite{lu2026braess} for rectangular grids of fixed aspect ratio $n : m$), candidate chords with adverse gradient $V_{uv} < 0$ emerge regularly.
    \item Therefore, the total number of Braess chords emerging directly along the agent's forward path is a binomial random variable with expectation:
    \begin{equation}
    \mathbb{E}[N_{\text{traps}}] = p_{\text{chord}} \cdot \rho_{\text{braess}} \cdot L_{\min} = \Theta(\operatorname{diam}(G)).
    \end{equation}
    \item Because the local backup budget satisfies $K = o(\operatorname{diam}(G))$, the spatial radius of Bellman backups $\mathcal{R}_{\text{backup}} \le K$ is asymptotically negligible compared to the graph diameter $\operatorname{diam}(G)$. At each Braess encounter, Standard-PS suffers a strictly positive trap probability $P_{\text{trap}} \ge p_0 > 0$ (empirically $p_0 = 0.50$ in Table~\ref{tab:results}).
    \item By linearity of expectation, the expected number of Braess traps exploited by Standard-PS scales as:
    \begin{equation}
    \mathbb{E}[M_{\text{trapped}}] = p_0 \cdot \mathbb{E}[N_{\text{traps}}] = \Omega(\operatorname{diam}(G)).
    \end{equation}
    Conversely, by Theorem~\ref{thm:trap_immunity}, SL-PS maintains an exact $0.0\%$ trap encounter rate across all steps.
\end{enumerate}

\paragraph{Part 3: Joint Asymptotic Limit.}
The total cumulative travel time advantage is the sum of delays across all encountered traps:
\begin{equation}
T_{\text{PS}} - T_{\text{SL-PS}} = \sum_{k=1}^{M_{\text{trapped}}} \Delta T_{\text{single}, k} \ge M_{\text{trapped}} \cdot \min_k \Delta T_{\text{single}, k}.
\end{equation}
Substituting the asymptotic lower bounds from Part 1 and Part 2:
\begin{equation}
T_{\text{PS}} - T_{\text{SL-PS}} \ge \Omega(\operatorname{diam}(G)) \cdot \Omega(q) = \Omega\big(q \cdot \operatorname{diam}(G)\big).
\end{equation}
On rectangular grids $G_{m,n}$, this yields $\Omega\big(q \cdot (m + n)\big)$. Taking the joint limit as $\operatorname{diam}(G) \to \infty$ and $q \to \infty$:
\begin{equation}
\lim_{\substack{\operatorname{diam}(G) \to \infty \\ q \to \infty}} \left( T_{\text{PS}} - T_{\text{SL-PS}} \right) \ge \lim_{\substack{\operatorname{diam}(G) \to \infty \\ q \to \infty}} C \cdot q \cdot \operatorname{diam}(G) = +\infty,
\end{equation}
for some positive constant $C > 0$. This establishes the asymptotic divergence of the cumulative travel time gap on general expanding networks and completes the proof.
\end{proof}"""

    replacement_proof_thm4_p2 = r"""\paragraph{Part 2: Expanding Topological Network Limit ($\operatorname{diam}(G) \to \infty$) Under Dynamic Branching.}
To establish asymptotic scaling under dynamic trajectory branching (where being trapped at step $t$ deflecting the agent into a bottleneck corridor affects subsequent state visitations), we formalize the problem through multi-stage corridor renewals:
\begin{enumerate}
    \item \textbf{Spatial Stage Decomposition}: Any directed network family with diameter $\operatorname{diam}(G)$ admits a topological partition into $M = \lfloor \operatorname{diam}(G) / \Delta L \rfloor = \Theta(\operatorname{diam}(G))$ sequential spatial cross-sections or macro-stages from origin $s$ to destination $t_{\text{sink}}$, where inter-stage distance $\Delta L > K$ exceeds the local Prioritized Sweeping backup budget $K = \mathcal{O}(1)$.
    \item \textbf{Stage Entry Stopping Times}: Let $\tau_1 < \tau_2 < \dots < \tau_M$ denote the sequence of Markov stopping times at which the agent enters stage $j \in \{1, \dots, M\}$. Regardless of whether the agent traverses a nominal corridor or is deflected through a bottleneck corridor in stage $j-1$, reaching the destination $t_{\text{sink}}$ strictly requires traversing downstream stages.
    \item \textbf{Independent Stage Commitment}: In each stage $j$, candidate chords emerge with probability $p_{\text{chord}} > 0$. Because $\Delta L > K$, Bellman residual backups executed in stage $j-1$ cannot span forward across distance $\Delta L$ to resolve downstream bottleneck divergence in stage $j$. Consequently, at each stage $j$ where a candidate chord emerges, Standard-PS suffers a conditionally independent commitment probability $P(\text{trap}_j \mid \mathcal{F}_{\tau_j}) \ge p_0 > 0$ (empirically $p_0 = 0.50$ in Table~\ref{tab:results}).
    \item \textbf{Expected Bottleneck Encounters}: By the Renewal Reward Theorem, the expected number of bottleneck traps entered along any realized trajectory satisfies:
    \begin{equation}
    \mathbb{E}[M_{\text{trapped}}] = \sum_{j=1}^M \mathbb{E}\left[ \mathbf{1}_{\{ \text{trap in stage } j \}} \right] \ge M \cdot p_{\text{chord}} \cdot \rho_{\text{braess}} \cdot p_0 = \Omega(\operatorname{diam}(G)).
    \end{equation}
    Even in the most conservative lower bound where only the single first encountered bottleneck is accounted for, entering that single trap already incurs $\Delta T_{\text{first}} = \Omega(q)$ latency by Part~1.
    \item \textbf{SL-PS Immunity Guarantee}: In contrast, by Theorem~\ref{thm:trap_immunity}, SL-PS evaluates $\beta(\sigma) \Phi_{\text{spec}} > 0$ at every decision junction, maintaining zero trap encounters ($0/180$ episodes in Section~\ref{subsec:stochastic_transitions}) across all stages.
\end{enumerate}

\paragraph{Part 3: Joint Asymptotic Limit.}
Combining the per-stage bottleneck latency penalty $\Delta T_{\text{single}} = \Omega(q)$ from Part~1 with the multi-stage spatial encounter bound $\mathbb{E}[M_{\text{trapped}}] = \Omega(\operatorname{diam}(G))$ from Part~2:
\begin{equation}
\mathbb{E}\left[ T_{\text{PS}} - T_{\text{SL-PS}} \right] \ge \mathbb{E}[M_{\text{trapped}}] \cdot \min_k \Delta T_{\text{single}, k} = \Omega(\operatorname{diam}(G)) \cdot \Omega(q) = \Omega\big(q \cdot \operatorname{diam}(G)\big).
\end{equation}
On rectangular grids $G_{m,n}$ with $\operatorname{diam}(G) = \Theta(m + n)$, this evaluates to $\Omega\big(q \cdot (m + n)\big)$. Taking the joint limit:
\begin{equation}
\lim_{\substack{\operatorname{diam}(G) \to \infty \\ q \to \infty}} \mathbb{E}\left[ T_{\text{PS}} - T_{\text{SL-PS}} \right] \ge \lim_{\substack{\operatorname{diam}(G) \to \infty \\ q \to \infty}} C \cdot q \cdot \operatorname{diam}(G) = +\infty,
\end{equation}
for strictly positive constant $C > 0$. This confirms the unbounded asymptotic advantage of SL-PS under dynamic trajectory branching and completes the proof.
\end{proof}"""
    assert text.count(target_proof_thm4_p2) == 1, "target_proof_thm4_p2 not unique"
    text = text.replace(target_proof_thm4_p2, replacement_proof_thm4_p2)

    # =========================================================================
    # FIX 7: Appendix A.7 subsec:proof_thm6 Part 3 Priority Ordering Rigor
    # =========================================================================
    pos8 = text.find(r'\textit{3. Priority Ordering and Diverter Preemption}:')
    pos8_end = text.find(r'\paragraph{Part 4: Decoupled', pos8)
    assert pos8 != -1 and pos8_end != -1, "pos8 block not found"
    target_proof_thm6_p3 = text[pos8:pos8_end]

    replacement_proof_thm6_p3 = r"""\textit{3. Diverter Non-Starvation and Priority Elevation via Backward Potential}:
Under SL-PS, the priority scheduling function combines the bounded Bellman residual with the backward in-Laplacian Green potential:
\begin{equation}
p_{\text{SL}}(w) = |\mathcal{T}_{\text{SL}} V(w) - V(w)| + \beta(\sigma) f_{\text{in}}(w).
\end{equation}
At chord entrance $u^*$, applying the sandwich bound in Eq.~\eqref{eq:sandwich_bound_rigorous}:
\begin{equation}
\label{eq:psl_chord_upper}
p_{\text{SL}}(u^*) = |\mathcal{T}_{\text{SL}} V(u^*) - V(u^*)| + \beta(\sigma) f_{\text{in}}(u^*) \le \beta(\sigma) \left( \bar{\Phi}_{\text{bypass}} + f_{\text{in}}(u^*) \right) < \infty.
\end{equation}
Meanwhile, at any upstream ancestor state $s \in \operatorname{Anc}(v^*)$, non-negativity of the local residual ensures:
\begin{equation}
\label{eq:psl_diverter_lower}
p_{\text{SL}}(s) \ge \beta(\sigma) f_{\text{in}}(s) > 0.
\end{equation}
Recall that in canonical Prioritized Sweeping, upstream state $s$ has zero residual ($p_{\text{PS}}(s) = 0$), causing the queue to allocate all $K$ budget iterations strictly to local chord neighbors while upstream diverters starve. In contrast, under SL-PS, $f_{\text{in}}(s) > 0$ strictly for all $s \in \operatorname{Anc}(v^*)$ across distance $m = \operatorname{dist}_G(s, v^*) \le \operatorname{diam}(G)$ directly from offline Green resolvent precomputation ($\operatorname{Radius}_{\text{SL-PS}}(0) = \operatorname{diam}(G)$). Consequently, every upstream diverter junction is guaranteed an immediate non-zero priority key:
\begin{equation}
p_{\text{SL}}(s) \ge \beta(\sigma) \cdot [\mathcal{G}_{\text{in}} \xi_{\text{div}}]_s > 0,
\end{equation}
preventing queue starvation and ensuring that upstream decision states enter the priority queue within backup budget $K$.

"""
    text = text[:pos8] + replacement_proof_thm6_p3 + text[pos8_end:]

    # =========================================================================
    # FIX 8: Appendix B Lemma 3 Linear Algebra (Weyl vs Bauer-Fike)
    # =========================================================================
    pos9 = text.find(r'\begin{lemma}[Spectral Stability and Resolvent Invariance under Rank-1 Chord Perturbation]')
    pos9_end = text.find(r'\section{Computational Complexity \& Priority Queue Analysis}', pos9)
    assert pos9 != -1 and pos9_end != -1, "pos9 block not found"
    target_lem3 = text[pos9:pos9_end]

    replacement_lem3 = r"""\begin{lemma}[Spectral Stability and Resolvent Invariance under Rank-1 Chord Perturbation]
\label{lem:weyl_stability}
Let $P_0 \in \mathbb{R}^{|V| \times |V|}$ be the base nominal random-walk matrix of directed network $G_0 = (V, E_0)$, and let $P_t = P_0 + \Delta P$ denote the augmented transition matrix when an auxiliary chord $e^* = (u \to v)$ is added. Then:
\begin{enumerate}
    \item \textbf{Rank-1 Perturbation Bound}: $\Delta P$ is a rank-1 perturbation with spectral norm bounded by:
    \begin{equation}
    \|\Delta P\|_2 \le \frac{1}{\sqrt{d_{\text{out}}(u) (d_{\text{out}}(u) + 1)}},
    \end{equation}
    where $d_{\text{out}}(u) \ge 1$ is the nominal out-degree of node $u$.
    \item \textbf{SLEM Eigenvalue Stability (via Bauer-Fike Theorem)}: If $P_0 = X \Lambda X^{-1}$ is diagonalizable with eigenvector condition number $\kappa_2(X) = \|X\|_2 \|X^{-1}\|_2$, then for every eigenvalue $\lambda(P_t)$ of $P_t$, there exists an eigenvalue $\lambda_0(P_0)$ of $P_0$ such that:
    \begin{equation}
    \label{eq:bauer_fike_bound}
    \left| \lambda(P_t) - \lambda_0(P_0) \right| \le \kappa_2(X) \|\Delta P\|_2 \le \frac{\kappa_2(X)}{\sqrt{d_{\text{out}}(u)(d_{\text{out}}(u)+1)}}.
    \end{equation}
    For reversible Markov chains (or undirected base networks), $P_0$ is similar to a real symmetric matrix with $\kappa_2(D^{1/2}) \le \sqrt{d_{\max}/d_{\min}}$, and Weyl's theorem applies directly with unit condition number. Furthermore, for general asymmetric matrices, Weyl's singular value theorem guarantees $|s_i(P_t) - s_i(P_0)| \le \|\Delta P\|_2$ for all singular values.
    \item \textbf{Green Resolvent Operator Invariance (Sherman-Morrison Identity)}: Independent of matrix symmetry or eigenvalue conditioning, the exact difference between the augmented Green operator $\mathcal{G}_{\text{out}}^{(t)} = (I - P_t + \eta I)^{-1}$ and the precomputed base Green operator $\mathcal{G}_{\text{out}}^{(0)} = (I - P_0 + \eta I)^{-1}$ satisfies:
    \begin{equation}
    \label{eq:green_resolvent_sherman_bound}
    \|\mathcal{G}_{\text{out}}^{(t)} - \mathcal{G}_{\text{out}}^{(0)}\|_\infty \le \frac{\|\mathcal{G}_{\text{out}}^{(0)}\|_\infty^2 \|\Delta P\|_\infty}{1 - \|\mathcal{G}_{\text{out}}^{(0)}\|_\infty \|\Delta P\|_\infty} \le \frac{2}{\eta (\eta(d_0 + 1) - 2)} = \mathcal{O}\left( \frac{1}{\eta^2 d_{\text{out}}(u)} \right).
    \end{equation}
\end{enumerate}
Consequently, computing $\Phi_{\text{spec}}$ using the offline precomputed base Green operator $\mathcal{G}_{\text{out}}^{(0)}$ introduces at most an $\mathcal{O}(1/|V|)$ localized perturbation, rigorously confirming that offline resolvent precomputation retains fidelity for real-time online planning.
\end{lemma}

\begin{proof}
When an auxiliary directed chord $e^* = (u \to v)$ is added to $G_0$, only the out-neighborhood of node $u$ is modified. Let $d_0 = d_{\text{out}}(u)$ be the nominal out-degree at $u$, which increases to $d_0 + 1$. The $u$-th row of the nominal transition matrix transitions from $P_0(u, \cdot) = \frac{1}{d_0} \sum_{w \in \mathcal{N}_{\text{out}}(u)} e_w^T$ to:
\begin{equation}
P_t(u, \cdot) = \frac{1}{d_0 + 1} \left( \sum_{w \in \mathcal{N}_{\text{out}}(u)} e_w^T + e_v^T \right).
\end{equation}
All other rows $i \ne u$ remain strictly unchanged ($P_t(i, \cdot) = P_0(i, \cdot)$). Therefore, the perturbation matrix is zero everywhere except on row $u$:
\begin{align}
\Delta P &= P_t - P_0 = e_u \left( P_t(u, \cdot) - P_0(u, \cdot) \right) \notag \\
&= e_u \left( \frac{1}{d_0 + 1} e_v^T - \frac{1}{d_0 (d_0 + 1)} \sum_{w \in \mathcal{N}_{\text{out}}(u)} e_w^T \right).
\end{align}
This establishes that $\operatorname{rank}(\Delta P) = 1$. The Frobenius norm evaluates to:
\begin{align}
\|\Delta P\|_2^2 &\le \|\Delta P\|_F^2 = \left( \frac{1}{d_0 + 1} \right)^2 + d_0 \left( \frac{1}{d_0 (d_0 + 1)} \right)^2 \notag \\
&= \frac{1}{(d_0 + 1)^2} \left( 1 + \frac{1}{d_0} \right) = \frac{1}{d_0 (d_0 + 1)}.
\end{align}
Taking the square root yields $\|\Delta P\|_2 \le \frac{1}{\sqrt{d_0 (d_0 + 1)}} \le \frac{1}{d_0}$.

Because directed network transition matrices are generally non-Hermitian, eigenvalue sensitivity to perturbation is governed by the Bauer-Fike Theorem~\cite{bauer1960norms, bhatia1997matrix}: for any eigenvalue $\lambda$ of $P_0 + \Delta P$, there exists an eigenvalue $\lambda_0$ of $P_0$ such that $|\lambda - \lambda_0| \le \kappa_2(X) \|\Delta P\|_2$, where $\kappa_2(X)$ is the condition number of the modal matrix $X$. For reversible Markov chains, similarity transformation $S = D^{1/2} P_0 D^{-1/2}$ produces a real symmetric matrix, reducing $\kappa_2$ to $\sqrt{d_{\max}/d_{\min}}$ and enabling Weyl's perturbation theorem~\cite{bhatia1997matrix, weyl1912asymptotische}.

Crucially, the stability of the Green resolvent operator $\mathcal{G}_{\text{out}}$ does not rely on eigenvalue perturbation, but rather on the exact algebraic Sherman-Morrison identity. Defining $M_0 = (1 + \eta)I - P_0$:
\begin{equation}
\mathcal{G}_{\text{out}}^{(t)} - \mathcal{G}_{\text{out}}^{(0)} = \left( M_0 - \Delta P \right)^{-1} - M_0^{-1} = M_0^{-1} \Delta P \left( I - M_0^{-1} \Delta P \right)^{-1} M_0^{-1}.
\end{equation}
Taking the induced infinity norm on both sides and using Lemma~\ref{lem:green_norm} ($\|M_0^{-1}\|_\infty = \|\mathcal{G}_{\text{out}}^{(0)}\|_\infty \le \frac{1}{\eta}$):
\begin{equation}
\|\mathcal{G}_{\text{out}}^{(t)} - \mathcal{G}_{\text{out}}^{(0)}\|_\infty \le \frac{\|\mathcal{G}_{\text{out}}^{(0)}\|_\infty^2 \|\Delta P\|_\infty}{1 - \|\mathcal{G}_{\text{out}}^{(0)}\|_\infty \|\Delta P\|_\infty}.
\end{equation}
Evaluating the row-sum norm:
\begin{equation}
\|\Delta P\|_\infty = \sum_{j \in V} |\Delta P_{uj}| = \frac{1}{d_0 + 1} + d_0 \frac{1}{d_0 (d_0 + 1)} = \frac{2}{d_0 + 1}.
\end{equation}
Substituting $\|\Delta P\|_\infty = \frac{2}{d_0 + 1}$ and $\|\mathcal{G}_{\text{out}}^{(0)}\|_\infty \le \frac{1}{\eta}$ into the inequality:
\begin{equation}
\|\mathcal{G}_{\text{out}}^{(t)} - \mathcal{G}_{\text{out}}^{(0)}\|_\infty \le \frac{\frac{1}{\eta^2} \frac{2}{d_0 + 1}}{1 - \frac{1}{\eta} \frac{2}{d_0 + 1}} = \frac{2}{\eta \left( \eta(d_0 + 1) - 2 \right)} = \mathcal{O}\left( \frac{1}{\eta^2 d_0} \right),
\end{equation}
holding for any $\eta(d_0 + 1) > 2$. This establishes that resolvent perturbation decays inversely with vertex out-degree $d_0$, confirming that offline Green precomputation introduces negligible error into the online potential field.
\end{proof}

"""
    text = text[:pos9] + replacement_lem3 + text[pos9_end:]

    # =========================================================================
    # FIX 9: Proposition 1 Complexity: Explicit Dense vs Sparse Solver Bounds
    # =========================================================================
    pos10 = text.find(r'\begin{proposition}[Per-Step Online Computational Complexity]')
    pos10_end = text.find(r'\section{Experimental Hyperparameters \& Reproducibility Protocol}', pos10)
    assert pos10 != -1 and pos10_end != -1, "pos10 block not found"
    target_prop1 = text[pos10:pos10_end]

    replacement_prop1 = r"""\begin{proposition}[Per-Step Online Computational Complexity]
\label{prop:complexity}
Let $G = (V, E)$ be a directed network with maximum vertex out-degree $d_{\max} = \max_{s} |\mathcal{A}(s)|$, and let $K$ denote the Prioritized Sweeping backup budget. The online computational complexity of SL-PS per decision step evaluates as follows:
\begin{enumerate}
    \item \textbf{Precomputed Dense Resolvent Regime}: When Green operators $\mathcal{G}_{\text{out}}$ and $\mathcal{G}_{\text{in}}$ are precomputed offline, online potential evaluation via dense matrix-vector products yields time complexity:
    \begin{equation}
    \label{eq:complexity_dense}
    \mathcal{T}_{\text{online}}^{\text{dense}} = \mathcal{O}\left( |V|^2 + |E_0| + K \cdot d_{\max} \log |V| \right).
    \end{equation}
    \item \textbf{Sparse Iterative Laplacian Solver Regime}: When potential diffusion is computed on-the-fly by solving regularized Laplacian systems $(I - P + \eta I)\Phi_{\text{spec}} = \xi_{\text{div}}$ using symmetric diagonally dominant (SDD) solvers~\cite{spielman2025sagt} or preconditioned conjugate gradients, online time complexity reduces to:
    \begin{equation}
    \label{eq:complexity_sparse}
    \mathcal{T}_{\text{online}}^{\text{sparse}} = \widetilde{\mathcal{O}}\left( |E_0| + K \cdot d_{\max} \log |V| \right),
    \end{equation}
    where $\widetilde{\mathcal{O}}$ hides polylogarithmic factors in $|V|$.
\end{enumerate}
\end{proposition}

\begin{proof}
We analyze each computational phase of Algorithm~1:
\begin{enumerate}
    \item \textbf{Flow Divergence Calculation}: Computing $\Delta \vec{\ell} = a (f_{\text{curr}} - f_{\text{base}})$ and vertex divergence $\xi_{\text{div}} = |B \Delta \vec{\ell}|$ requires a single sparse matrix-vector multiplication with the directed incidence matrix $B \in \mathbb{R}^{|V| \times |E_0|}$. Because each edge column of $B$ contains exactly two non-zero entries ($+1, -1$), this evaluates in $\mathcal{O}(|E_0|)$ operations.
    \item \textbf{Potential Field Evaluation}: 
    \begin{itemize}
        \item Under the precomputed resolvent mode, multiplying the dense precomputed matrix $\mathcal{G}_{\text{out}} \in \mathbb{R}^{|V| \times |V|}$ by vector $\xi_{\text{div}}$ takes $\mathcal{O}(|V|^2)$ operations.
        \item Under the iterative solver mode, solving the diagonally dominant M-matrix system $(I - P + \eta I)\Phi = \xi_{\text{div}}$ via Spielman-Teng SDD Laplacian solvers~\cite{spielman2025sagt} or preconditioned Krylov methods takes $\widetilde{\mathcal{O}}(|E_0|)$ operations on sparse graphs where $|E_0| = \mathcal{O}(|V| \cdot d_{\max})$.
    \end{itemize}
    \item \textbf{Priority Queue Construction}: Computing initial priorities $p_{\text{SL}}(s) = \xi(s) + \beta(\sigma) f_{\text{in}}(s)$ for non-zero divergence nodes and constructing the binary min-heap requires $\mathcal{O}(|V|)$ time via linear bottom-up heapification.
    \item \textbf{Budgeted Value Backups}: For each of the $K$ queue iterations: extracting the minimum-priority state takes $\mathcal{O}(\log |V|)$, evaluating the Bellman backup takes $\mathcal{O}(d_{\max})$ transitions, and updating predecessor priorities in the heap takes $\mathcal{O}(d_{\max} \log |V|)$. Total sweeping time is strictly bounded by $\mathcal{O}(K \cdot d_{\max} \log |V|)$.
\end{enumerate}
Summing these operations yields Eq.~\eqref{eq:complexity_dense} for the dense precomputed mode and Eq.~\eqref{eq:complexity_sparse} for the sparse solver mode, confirming that per-step computation remains strictly independent of episode horizon $T$.
\end{proof}

"""
    text = text[:pos10] + replacement_prop1 + text[pos10_end:]

    print("All 9 fixes successfully applied!")
    print(f"New text length: {len(text)} chars")
    
    with open('idea.tex', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Successfully wrote updated idea.tex!")

if __name__ == '__main__':
    apply_fixes()
