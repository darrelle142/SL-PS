# scratch/check_targets.py

with open('idea.tex', 'r', encoding='utf-8') as f:
    text = f.read()

def check(name, target):
    c = text.count(target)
    print(f"[{name}] count = {c}")
    if c != 1:
        print(f"  ERROR: {name} count is not 1!")

# 1. target_bib
target_bib = r"""\begin{thebibliography}{10}

\bibitem{beckmann1956studies}"""
check("target_bib", target_bib)

# 2. target_jov
target_jov = r"""\bibitem{gta_ps2025}
Anonymous Authors.
\newblock Graph topology augmentation for prioritized sweeping in non-stationary reinforcement learning.
\newblock {\em Preprint}, 2025.

\bibitem{lu2026braess}"""
check("target_jov", target_jov)

# 3. target_contrib
target_contrib = r"""    \item \textbf{The Spectral-Laplacian Bellman Operator}: We derive $\mathcal{T}_{\text{SL}}$, a novel contraction operator that penalizes actions leading toward systemic congestion using the global \textit{Spectral Congestion Divergence Potential} $\Phi_{\text{spec}} = \mathcal{G}_{\text{out}} |B \Delta \vec{\ell}|$, where $\mathcal{G}_{\text{out}} = (I - P + \eta I)^{-1}$ is the Green's operator of the directed out-Laplacian on arbitrary directed graphs and $\Delta \vec{\ell}$ is the flow divergence vector.
    \item \textbf{Theoretical Guarantees on General Graphs}: We prove that $\mathcal{T}_{\text{SL}}$ is a strict $\gamma$-contraction in $L_\infty$ on arbitrary finite graphs (cyclic or acyclic) and establish the Braess Trap Immunity Theorem for general topologies. Furthermore, we derive the Lipschitz Stability Theorem, establishing explicit robustness tolerance margins under inexact Wardrop equilibrium estimation, and prove the Asymptotic Dominance Theorem, establishing that the cumulative travel time advantage of SL-PS over Standard-PS diverges to $+\infty$ at rate $\Omega(q \cdot \operatorname{diam}(G))$ as network scale and background demand expand.
    \item \textbf{Spectral Priority Scheduling}: We integrate bi-directional Green's potentials ($f_{\text{out}}, f_{\text{in}}$) annealed by the Second Largest Eigenvalue Modulus (SLEM) $\sigma(P) = |\lambda_2(P)|$, ensuring that distant upstream junction states receive immediate replanning priority before physical wavefronts arrive."""
check("target_contrib", target_contrib)

# 4. target_thm3
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
check("target_thm3", target_thm3)

# 5. target_dijkstra
target_dijkstra = r"""\paragraph{Dissecting Baseline Behaviors: Fixed Perimeter Heuristics vs. Adaptive Value Planners.}
A superficial examination of Table~\ref{tab:results} and Figure~\ref{fig:benchmark_results} might suggest an apparent puzzle: the simple Greedy-Dijkstra heuristic records $0$ trap encounters, matching SL-PS's zero rate. However, a rigorous algorithmic inspection reveals a fundamental distinction:
\begin{enumerate}
    \item \textbf{Greedy-Dijkstra as an Oblivious Boundary Hugger}: Greedy-Dijkstra's zero-trap outcome is purely a fortuitous topological artifact of its static Manhattan tie-breaking rule. Because it prioritizes horizontal movement along row 0 until hitting the boundary, its trajectory never intersects interior junction $(1, 1)$, where the Braess shortcut $(1, 1) \to (0, 3)$ originates. The baseline does not ``avoid'' the trap through congestion awareness; it is completely blind to network dynamics and simply follows a fixed perimeter path. In real-world systems or under stochastic perturbations (Section~\ref{subsec:stochastic_transitions}), such open-loop static heuristics suffer catastrophic degradation.
    \item \textbf{The Dilemma of Adaptive Planners (Standard-PS and GTA-PS)}: Unlike Dijkstra, Standard-PS and GTA-PS are true adaptive, value-driven reinforcement learning planners. They propagate value updates backward from the goal to discover optimal policies. However, when the Braess shortcut appears with nominal latency $d \approx 0$, local Bellman residual updates at $(1, 1)$ propagate an illusory cost reduction backward to $(0, 1)$ and $(0, 0)$. This actively misleads the agents into diverting inward away from the perimeter to exploit the chord, directly precipitating systemic network collapse and inducing severe trap rates of $50.0\%$ on $G_{3,6}$ and $20.0\%$ on $G_{4,8}$.
    \item \textbf{Active Congestion-Aware Planning in SL-PS}: SL-PS overcomes this fundamental dilemma. Like Standard-PS, SL-PS is an adaptive value planner that actively evaluates interior shortcuts. However, unlike Standard-PS and GTA-PS, SL-PS incorporates the Spectral Congestion Divergence Potential $\Phi_{\text{spec}} = \mathcal{G}_{\text{out}} |B \Delta \vec{\ell}|$ into its Bellman operator $\mathcal{T}_{\text{SL}}$. It recognizes that taking the chord will overload the downstream bottleneck and actively penalizes the chord action on principled physical grounds. SL-PS therefore achieves the best of both worlds: full adaptive planning capability with principled, congestion-aware trap immunity.
\end{enumerate}"""
check("target_dijkstra", target_dijkstra)

# 6. target_proof_thm3
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
check("target_proof_thm3", target_proof_thm3)

# 7. target_proof_thm4_p2
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
check("target_proof_thm4_p2", target_proof_thm4_p2)

# 8. target_proof_thm6_p3
target_proof_thm6_p3 = r"""\textit{3. Priority Ordering and Diverter Preemption}:
Under SL-PS, the priority scheduling function is defined as:
\begin{equation}
p_{\text{SL}}(w) = |\mathcal{T}_{\text{SL}} V(w) - V(w)| + \beta(\sigma) f_{\text{in}}(w).
\end{equation}
At chord entrance $u^*$, applying the sandwich bound in Eq.~\eqref{eq:sandwich_bound_rigorous}:
\begin{equation}
\label{eq:psl_chord_upper}
p_{\text{SL}}(u^*) = |\mathcal{T}_{\text{SL}} V(u^*) - V(u^*)| + \beta(\sigma) f_{\text{in}}(u^*) \le \beta(\sigma) \left( \bar{\Phi}_{\text{bypass}} + f_{\text{in}}(u^*) \right).
\end{equation}
At upstream diverter state $s$, non-negativity of the local Bellman residual yields:
\begin{equation}
\label{eq:psl_diverter_lower}
p_{\text{SL}}(s) = |\mathcal{T}_{\text{SL}} V(s) - V(s)| + \beta(\sigma) f_{\text{in}}(s) \ge \beta(\sigma) f_{\text{in}}(s).
\end{equation}
Comparing Eq.~\eqref{eq:psl_diverter_lower} and Eq.~\eqref{eq:psl_chord_upper}, the condition $p_{\text{SL}}(s) \ge p_{\text{SL}}(u^*)$ is guaranteed whenever:
\begin{equation}
\beta(\sigma) f_{\text{in}}(s) \ge \beta(\sigma) \left( \bar{\Phi}_{\text{bypass}} + f_{\text{in}}(u^*) \right) \iff f_{\text{in}}(s) - f_{\text{in}}(u^*) \ge \bar{\Phi}_{\text{bypass}}.
\end{equation}
More generally, in slow-mixing networks where $\sigma(P) = |\lambda_2(P)| \to 1$, the SLEM-modulated multiplier satisfies $\beta(\sigma) = \beta_0 (1 + \sigma(P)) \ge 2 \beta_0$. For any target diverter state $s$ with $f_{\text{in}}(s) > 0$, setting base multiplier:
\begin{equation}
\label{eq:beta_zero_star}
\beta_0 \ge \beta_0^* \equiv \frac{|\mathcal{T}_{\text{SL}} V(u^*) - V(u^*)|}{f_{\text{in}}(s)} = \frac{\beta(\sigma) \Phi_{\text{spec}}(s_{\text{bypass}})}{f_{\text{in}}(s)}
\end{equation}
strictly ensures $p_{\text{SL}}(s) \ge p_{\text{SL}}(u^*)."""
check("target_proof_thm6_p3", target_proof_thm6_p3)

# 9. target_lem3
target_lem3 = r"""\begin{lemma}[Spectral Stability and Resolvent Invariance under Rank-1 Chord Perturbation]
\label{lem:weyl_stability}
Let $P_0 \in \mathbb{R}^{|V| \times |V|}$ be the base nominal random-walk matrix of directed network $G_0 = (V, E_0)$, and let $P_t = P_0 + \Delta P$ denote the augmented transition matrix when an auxiliary chord $e^* = (u \to v)$ is added. Then:
\begin{enumerate}
    \item \textbf{Rank-1 Perturbation Bound}: $\Delta P$ is a rank-1 perturbation with spectral norm bounded by:
    \begin{equation}
    \|\Delta P\|_2 \le \frac{1}{\sqrt{d_{\text{out}}(u) (d_{\text{out}}(u) + 1)}},
    \end{equation}
    where $d_{\text{out}}(u) \ge 1$ is the nominal out-degree of node $u$.
    \item \textbf{SLEM Eigenvalue Stability (via Weyl's Theorem)}: The shift in the Second Largest Eigenvalue Modulus $\sigma(P) = |\lambda_2(P)|$ satisfies:
    \begin{equation}
    \left| \sigma(P_t) - \sigma(P_0) \right| \le \|\Delta P\|_2 \le \mathcal{O}\left( \frac{1}{|V|^{1/2}} \right).
    \end{equation}
    \item \textbf{Green's Resolvent Operator Invariance (via Sherman-Morrison)}: The difference between the augmented Green operator $\mathcal{G}_{\text{out}}^{(t)} = (I - P_t + \eta I)^{-1}$ and the precomputed base Green operator $\mathcal{G}_{\text{out}}^{(0)} = (I - P_0 + \eta I)^{-1}$ satisfies:
    \begin{equation}
    \|\mathcal{G}_{\text{out}}^{(t)} - \mathcal{G}_{\text{out}}^{(0)}\|_\infty \le \frac{\|\mathcal{G}_{\text{out}}^{(0)}\|_\infty^2 \|\Delta P\|_\infty}{1 - \|\mathcal{G}_{\text{out}}^{(0)}\|_\infty \|\Delta P\|_\infty} = \mathcal{O}\left( \frac{1}{\eta^2 d_{\text{out}}(u)} \right).
    \end{equation}
\end{enumerate}
Consequently, computing $\Phi_{\text{spec}}$ using the offline precomputed base Green operator $\mathcal{G}_{\text{out}}^{(0)}$ introduces at most an $\mathcal{O}(1/|V|)$ perturbation to the potential field, rigorously justifying offline precomputation for real-time online planning.
\end{lemma}

\begin{proof}
When an auxiliary directed chord $e^* = (u \to v)$ is added to $G_0$, only the out-neighborhood of node $u$ is modified. Let $d_0 = d_{\text{out}}(u)$ be the nominal out-degree at $u$, which increases to $d_0 + 1$. The $u$-th row of the nominal transition matrix, originally given by $P_0(u, \cdot) = \frac{1}{d_0} \sum_{w \in \mathcal{N}_{\text{out}}(u)} e_w^T$, transitions to:
\begin{equation}
P_t(u, \cdot) = \frac{1}{d_0 + 1} \left( \sum_{w \in \mathcal{N}_{\text{out}}(u)} e_w^T + e_v^T \right).
\end{equation}
All other rows $i \ne u$ remain strictly unchanged ($P_t(i, \cdot) = P_0(i, \cdot)$). Therefore, the perturbation matrix is zero everywhere except on row $u$:
\begin{align}
\Delta P &= P_t - P_0 = e_u \left( P_t(u, \cdot) - P_0(u, \cdot) \right) \notag \\
&= e_u \left( \frac{1}{d_0 + 1} e_v^T - \frac{1}{d_0 (d_0 + 1)} \sum_{w \in \mathcal{N}_{\text{out}}(u)} e_w^T \right).
\end{align}
This establishes that $\operatorname{rank}(\Delta P) = 1$. The Frobenius norm (which bounds the spectral norm $\|\cdot\|_2$) evaluates to:
\begin{align}
\|\Delta P\|_2^2 &\le \|\Delta P\|_F^2 = \left( \frac{1}{d_0 + 1} \right)^2 + d_0 \left( \frac{1}{d_0 (d_0 + 1)} \right)^2 \notag \\
&= \frac{1}{(d_0 + 1)^2} \left( 1 + \frac{1}{d_0} \right) = \frac{1}{d_0 (d_0 + 1)}.
\end{align}
Taking the square root yields $\|\Delta P\|_2 \le \frac{1}{\sqrt{d_0 (d_0 + 1)}} \le \frac{1}{d_0}$.

By Weyl's Perturbation Theorem~\cite{bhatia1997matrix, weyl1912asymptotische}, for any matrix perturbation $A + E$, the singular values and eigenvalues of the symmetric part satisfy $|\lambda_i(P_t) - \lambda_i(P_0)| \le \|\Delta P\|_2$. For a network with $N = |V|$ vertices, local node degree $d_0 \ge 1$ implies that the influence of a single row modification on the global spectrum is bounded by $\mathcal{O}(1/\sqrt{N})$.

Finally, by the resolvent identity and the Sherman-Morrison rank-1 update formula, defining $M_0 = (1 + \eta)I - P_0$:
\begin{equation}
\mathcal{G}_{\text{out}}^{(t)} - \mathcal{G}_{\text{out}}^{(0)} = \left( M_0 - \Delta P \right)^{-1} - M_0^{-1} = M_0^{-1} \Delta P \left( I - M_0^{-1} \Delta P \right)^{-1} M_0^{-1}.
\end{equation}
Taking the induced infinity norm on both sides and using Lemma~\ref{lem:green_norm} ($\|M_0^{-1}\|_\infty = \|\mathcal{G}_{\text{out}}^{(0)}\|_\infty \le \frac{1}{\eta}$):
\begin{equation}
\|\mathcal{G}_{\text{out}}^{(t)} - \mathcal{G}_{\text{out}}^{(0)}\|_\infty \le \frac{\|\mathcal{G}_{\text{out}}^{(0)}\|_\infty^2 \|\Delta P\|_\infty}{1 - \|\mathcal{G}_{\text{out}}^{(0)}\|_\infty \|\Delta P\|_\infty}.
\end{equation}
Since $\|\Delta P\|_\infty = \sum_j |\Delta P_{uj}| = \frac{2}{d_0 + 1},"""
check("target_lem3", target_lem3)

# 10. target_prop1
target_prop1 = r"""\begin{proposition}[Per-Step Online Computational Complexity]
\label{prop:complexity}
Let $G = (V, E)$ be a directed network with maximum vertex out-degree $d_{\max} = \max_{s} |\mathcal{A}(s)|$ (e.g., $d_{\max} \le 3$ for planar grids, $d_{\max} \le 5$ for municipal road topologies), and let $K$ denote the Prioritized Sweeping backup budget. Under precomputed Green operators $\mathcal{G}_{\text{out}}$ and $\mathcal{G}_{\text{in}}$, the online time complexity of SL-PS per decision step is:
\begin{equation}
\mathcal{T}_{\text{online}} = \mathcal{O}\left( |E_0| + |V| + K \cdot d_{\max} \log |V| \right) = \mathcal{O}(|E_0| + |V| + K \log |V|).
\end{equation}
\end{proposition}

\begin{proof}
We analyze each phase of Algorithm~1:
\begin{enumerate}
    \item \textbf{Flow Divergence Calculation}: Computing $\Delta \vec{\ell} = a (f_{\text{curr}} - f_{\text{base}})$ and the divergence vector $\xi_{\text{div}} = |B \Delta \vec{\ell}|$ requires a single sparse matrix-vector multiplication with the incidence matrix $B$. Since each edge has exactly two non-zero entries in $B$, this takes $\mathcal{O}(|E_0|)$ operations.
    \item \textbf{Global Green Diffusion}: Evaluating $\Phi_{\text{spec}} = \mathcal{G}_{\text{out}} \xi_{\text{div}}$ requires a matrix-vector product. Because $\mathcal{G}_{\text{out}}$ is precomputed offline, this dense operation requires $\mathcal{O}(|V|^2)$ operations, which reduces to $\mathcal{O}(|V| \cdot d)$ when solving the linear system $(I - P + \eta I) \Phi = \xi_{\text{div}}$ via sparse preconditioned conjugate gradient.
    \item \textbf{Priority Queue Construction}: Computing initial priorities $p(s) = \xi(s) + \beta f_{\text{in}}(s)$ and inserting non-zero states into the binary min-heap requires $\mathcal{O}(|V| \log |V|)$ in the worst case, or $\mathcal{O}(|V|)$ via linear-time heapify.
    \item \textbf{Budgeted Value Backups}: At each of the $K$ iterations, extracting the top state takes $\mathcal{O}(\log |V|)$, evaluating the Bellman backup takes $\mathcal{O}(d)$, and pushing predecessor updates takes $\mathcal{O}(d \log |V|)$. Total backup time is $\mathcal{O}(K \cdot d \log |V|)$.
\end{enumerate}
Summing these components, total online execution time is $\mathcal{O}(|E_0| + |V| + K \log |V|),"""
check("target_prop1", target_prop1)
