# coding: utf-8
import sys

with open('idea.tex', 'r', encoding='utf-8') as f:
    text = f.read()

print('Initial length:', len(text))

# 1. Update Theorem 6 Item 1 in Section 5.5
old_thm6_item1 = r"In contrast, SL-PS evaluates the backward potential $f_{\text{in}}(s) = (\mathcal{G}_{\text{in}} \xi_{\text{div}})(s) > 0$ directly via regularized Green resolvents, achieving global reachability instantaneously at zero sweeping budget:"
new_thm6_item1 = r"In contrast, SL-PS evaluates the regularized Green resolvents ($\Phi_{\text{spec}}(s) = (\mathcal{G}_{\text{out}} \xi_{\text{div}})(s) > 0$ and $f_{\text{in}}(s) = (\mathcal{G}_{\text{in}} \xi_{\text{div}})(s) > 0$) directly via offline matrix inversion, achieving global reachability instantaneously at zero sweeping budget:"

assert text.count(old_thm6_item1) == 1, "Failed to match old_thm6_item1"
text = text.replace(old_thm6_item1, new_thm6_item1)
print("Updated Theorem 6 Item 1.")

# 2. Update Part 2 of Proof of Theorem 6 in Appendix A.7
idx_p2 = text.rfind(r'\paragraph{Part 2: Instantaneous Global Reachability')
idx_p3 = text.rfind(r'\paragraph{Part 3: Rigorous Sandwich-Bound')
assert idx_p2 != -1 and idx_p3 != -1 and idx_p2 < idx_p3, "Failed to locate Part 2 boundaries"

new_part2_text = r"""\paragraph{Part 2: Instantaneous Global Reachability via Regularized Green Resolvents.}
In SL-PS, global reachability from upstream ancestor state $s \in \operatorname{Anc}(v^*)$ to downstream bottleneck $v^*$ at zero sweeping budget ($K = 0$) is guaranteed through the regularized Green resolvent operators.

Let $v^*$ be the bottleneck node carrying strictly positive Wardrop flow divergence $\xi_{\text{div}}(v^*) = \Delta_0 > 0$. Because network $G$ is connected and $s \in \operatorname{Anc}(v^*)$, there exists at least one directed path of length $m = \operatorname{dist}_G(s, v^*) \le \operatorname{diam}(G)$:
\begin{equation}
s = w_0 \to w_1 \to \dots \to w_m = v^*.
\end{equation}
Along this path, the forward policy random-walk transition probability at each step satisfies $P_{w_i, w_{i+1}} = \frac{1}{d_{\text{out}}(w_i)} \ge \frac{1}{d_{\text{out}}^{\max}} > 0$. Therefore, the probability that an $m$-step random walk initiated at $s$ terminates at bottleneck $v^*$ is strictly positive:
\begin{equation}
\label{eq:random_walk_path_bound}
\mathbb{P}(\operatorname{Walk}_m = v^* \mid \operatorname{Walk}_0 = s) = [P^m]_{s, v^*} \ge \prod_{i=0}^{m-1} \frac{1}{d_{\text{out}}(w_i)} \ge \left( \frac{1}{d_{\text{out}}^{\max}} \right)^m > 0.
\end{equation}

By Lemma~\ref{lem:green_norm} (Appendix~\ref{sec:appendix_spectral}), the out-Laplacian Green operator $\mathcal{G}_{\text{out}} = (I - P + \eta I)^{-1}$ admits the exact row-stochastic Neumann series $\mathcal{G}_{\text{out}} = \sum_{k=0}^\infty \frac{1}{(1+\eta)^{k+1}} P^k$. Because all terms in this expansion are entrywise non-negative and $\xi_{\text{div}} \ge 0$, the $m$-th term provides a strict lower bound on the forward spectral divergence potential at ancestor $s$:
\begin{equation}
\label{eq:phi_spec_strictly_positive}
\Phi_{\text{spec}}(s) = [\mathcal{G}_{\text{out}} \xi_{\text{div}}]_s = \sum_{k=0}^\infty \frac{1}{(1+\eta)^{k+1}} [P^k \xi_{\text{div}}](s) \ge \frac{1}{(1+\eta)^{m+1}} [P^m]_{s, v^*} \Delta_0 \ge \frac{\Delta_0}{(1+\eta)^{m+1} (d_{\text{out}}^{\max})^m} > 0.
\end{equation}

\begin{remark}[Mathematical Rigor: Resolvent Duality and Column-Stochasticity]
We address a subtle but essential mathematical distinction: in general directed networks, transition matrix $P$ is row-stochastic ($\sum_j P_{ij} = 1$), but generally \textit{not} column-stochastic ($\sum_i P_{ij} \ne 1$). Consequently, the in-degree diagonal matrix $D_{\text{in}} = \operatorname{diag}(P^T \mathbf{1})$ satisfies $D_{\text{in}} \ne I$. The in-Laplacian resolvent $\mathcal{G}_{\text{in}} = (D_{\text{in}} - P^T + \eta I)^{-1}$ therefore cannot be expanded as a simple scalar series $((1+\eta)I - P^T)^{-1}$. 
Instead, as proven in Lemma~\ref{lem:green_norm} (Appendix~\ref{sec:appendix_spectral}), defining $D_\eta \equiv D_{\text{in}} + \eta I > 0$, $\mathcal{G}_{\text{in}}$ admits the exact diagonally scaled M-matrix Neumann series:
\begin{equation}
\label{eq:gin_m_matrix_expansion}
\mathcal{G}_{\text{in}} = \sum_{k=0}^\infty (D_\eta^{-1} P^T)^k D_\eta^{-1} \ge 0,
\end{equation}
which converges unconditionally since $\|D_\eta^{-1} P^T\|_\infty < 1$. Because $P_{w_i, w_{i+1}} > 0$ along path $s \to \dots \to v^*$, the transposed entry $[P^T]_{v^*, s}$ connects the bottleneck to upstream ancestor $s$ with strictly positive weight:
\begin{equation}
[\mathcal{G}_{\text{in}}]_{v^*, s} \ge \frac{1}{(d_{\text{in}}^{\max} + \eta)^{m+1}} \prod_{i=0}^{m-1} P_{w_i, w_{i+1}} > 0.
\end{equation}
Moreover, path reachability $\mathbb{P}(\operatorname{Walk}_m = v^* \mid \operatorname{Walk}_0 = s)$ is physically governed by forward progression from $s$ to $v^*$, which is captured directly and exactly by $\Phi_{\text{spec}}(s) = [\mathcal{G}_{\text{out}} \xi_{\text{div}}]_s > 0$ via the row-stochastic Neumann series of $\mathcal{G}_{\text{out}}$, completely independent of column-stochasticity.
\end{remark}

In the SL-PS priority queue scheduling rule (Section~\ref{sec:priority_scheduling}), the diffused spectral priority key combines both channels:
\begin{equation}
\tau(s) = (1 - \alpha) f_{\text{out}}(s) + \alpha f_{\text{in}}(s), \quad \text{where } f_{\text{out}} = \mathcal{G}_{\text{out}}(\xi_{\text{res}} + \Phi_{\text{spec}}), \quad f_{\text{in}} = \mathcal{G}_{\text{in}}(\xi_{\text{res}} + \Phi_{\text{spec}}).
\end{equation}
Because $\Phi_{\text{spec}}(s) > 0$ and both Green operators are entrywise non-negative ($\mathcal{G}_{\text{out}} \ge 0, \mathcal{G}_{\text{in}} \ge 0$), $\tau(s) > 0$ holds strictly for all ancestor states $s \in \operatorname{Anc}(v^*)$ across arbitrary distances $m \le \operatorname{diam}(G)$, evaluated directly through offline Green matrix inversion prior to online planning ($K = 0$). Hence:
\begin{equation}
\operatorname{Radius}_{\text{SL-PS}}(0) = \operatorname{diam}(G),
\end{equation}
establishing Eq.~\eqref{eq:sl_reachability_radius} and completing Part 2.

"""

text = text[:idx_p2] + new_part2_text + text[idx_p3:]
print("Updated Part 2 in Appendix A.7.")

# 3. Update Appendix B Lemma 1 and Proof
idx_sec = text.rfind(r'\label{sec:appendix_spectral}')
idx_lem1 = text.find(r'\begin{lemma}[Positivity and $L_\infty$ Resolvent Norm Bound]', idx_sec)
idx_lem2 = text.find(r'\begin{lemma}[Spectral Scale Invariance', idx_lem1)
assert idx_lem1 != -1 and idx_lem2 != -1, "Failed to locate Lemma 1 in Appendix B"

new_lemma1_text = r"""\begin{lemma}[Positivity, Duality, and Uniform $L_\infty$ Resolvent Norm Bounds]
\label{lem:green_norm}
Let $P \in \mathbb{R}^{|V| \times |V|}$ be a row-stochastic transition matrix ($\sum_j P_{ij} = 1, P_{ij} \ge 0$), and let $\eta > 0$ be the spectral regularization parameter. Define the policy-induced regularized out-Laplacian and in-Laplacian Green operators:
\begin{equation}
\mathcal{G}_{\text{out}} = (I - P + \eta I)^{-1}, \quad \mathcal{G}_{\text{in}} = (D_{\text{in}} - P^T + \eta I)^{-1},
\end{equation}
where $D_{\text{in}} = \operatorname{diag}(P^T \mathbf{1})$. Then both operators satisfy:
\begin{enumerate}
    \item \textbf{Entrywise Non-Negativity}: $\mathcal{G}_{\text{out}, ij} \ge 0$ and $\mathcal{G}_{\text{in}, ij} \ge 0$ for all $i, j \in V$.
    \item \textbf{Exact Uniform Operator Norm Bounds}: $\|\mathcal{G}_{\text{out}}\|_\infty = \frac{1}{\eta}$ and $\|\mathcal{G}_{\text{in}}\|_\infty = \frac{1}{\eta}$.
    \item \textbf{Exact Resolvent Series Representations}: $\mathcal{G}_{\text{out}}$ admits the exact row-stochastic Neumann series:
    \begin{equation}
    \label{eq:g_out_neumann}
    \mathcal{G}_{\text{out}} = \sum_{k=0}^\infty \frac{1}{(1 + \eta)^{k+1}} P^k,
    \end{equation}
    while $\mathcal{G}_{\text{in}}$ admits the diagonally scaled M-matrix Neumann series:
    \begin{equation}
    \label{eq:g_in_neumann}
    \mathcal{G}_{\text{in}} = \sum_{k=0}^\infty (D_\eta^{-1} P^T)^k D_\eta^{-1},
    \end{equation}
    where $D_\eta \equiv D_{\text{in}} + \eta I = \operatorname{diag}(d_{\text{in}}(u) + \eta)$ is strictly positive diagonal, holding rigorously on arbitrary directed graphs without requiring $P$ to be column-stochastic.
\end{enumerate}
\end{lemma}

\begin{proof}
We analyze each Green operator independently.

\paragraph{1. Analysis of $\mathcal{G}_{\text{out}}$}:
Rewrite the out-Laplacian resolvent as a scaled matrix operator:
\begin{equation}
\mathcal{G}_{\text{out}} = \left( (1 + \eta) I - P \right)^{-1} = \frac{1}{1 + \eta} \left( I - \frac{1}{1 + \eta} P \right)^{-1}.
\end{equation}
Define $\alpha \equiv \frac{1}{1 + \eta}$. Since $\eta > 0$, we have $0 < \alpha < 1$. Because $P$ is row-stochastic, its induced infinity norm is $\|P\|_\infty = \max_i \sum_j |P_{ij}| = 1$. The spectral radius satisfies $\rho(\alpha P) \le \|\alpha P\|_\infty = \alpha < 1$.

Therefore, the Neumann series for $(I - \alpha P)^{-1}$ converges absolutely and unconditionally:
\begin{equation}
\mathcal{G}_{\text{out}} = \alpha \sum_{k=0}^\infty (\alpha P)^k = \sum_{k=0}^\infty \frac{1}{(1 + \eta)^{k+1}} P^k.
\end{equation}
Since $P_{ij} \ge 0$, every power $P^k$ has non-negative entries ($P^k_{ij} \ge 0$). Since $(1+\eta)^{-(k+1)} > 0$, every entry of $\mathcal{G}_{\text{out}}$ is a sum of non-negative real numbers, establishing $\mathcal{G}_{\text{out}, ij} \ge 0$ for all $i, j \in V$.

Because $P \mathbf{1} = \mathbf{1}$, we have $P^k \mathbf{1} = \mathbf{1}$ for all $k \ge 0$. Evaluating the row sums of $\mathcal{G}_{\text{out}}$:
\begin{equation}
\mathcal{G}_{\text{out}} \mathbf{1} = \sum_{k=0}^\infty \frac{1}{(1 + \eta)^{k+1}} P^k \mathbf{1} = \left( \sum_{k=0}^\infty \frac{1}{(1 + \eta)^{k+1}} \right) \mathbf{1} = \frac{1}{\eta} \mathbf{1}.
\end{equation}
Since all entries of $\mathcal{G}_{\text{out}}$ are non-negative, the induced infinity norm evaluates exactly to:
\begin{equation}
\|\mathcal{G}_{\text{out}}\|_\infty = \max_{u \in V} [\mathcal{G}_{\text{out}} \mathbf{1}]_u = \frac{1}{\eta}.
\end{equation}

\paragraph{2. Analysis of $\mathcal{G}_{\text{in}}$}:
In general directed graphs, $P$ is row-stochastic but not necessarily column-stochastic ($\sum_i P_{ij} \ne 1$). Consequently, $D_{\text{in}} = \operatorname{diag}(P^T \mathbf{1}) \ne I$, so $(D_{\text{in}} - P^T + \eta I) \ne (1+\eta)I - P^T$. We establish the mathematical properties of $\mathcal{G}_{\text{in}}$ through non-singular M-matrix theory.

Define the diagonal matrix $D_\eta \equiv D_{\text{in}} + \eta I$. For any node $u \in V$, the diagonal entry satisfies $[D_\eta]_{uu} = d_{\text{in}}(u) + \eta \ge \eta > 0$. Hence $D_\eta$ is strictly positive and invertible with $D_\eta^{-1} = \operatorname{diag}\left( \frac{1}{d_{\text{in}}(u) + \eta} \right)$. Factoring $D_\eta$ from the left yields:
\begin{equation}
D_{\text{in}} - P^T + \eta I = D_\eta - P^T = D_\eta \left( I - D_\eta^{-1} P^T \right).
\end{equation}
Define the normalized backward transition matrix $\widetilde{P}_{\text{in}} \equiv D_\eta^{-1} P^T$. The entries of $\widetilde{P}_{\text{in}}$ are non-negative: $[\widetilde{P}_{\text{in}}]_{uv} = \frac{P_{vu}}{d_{\text{in}}(u) + \eta} \ge 0$. The row sums of $\widetilde{P}_{\text{in}}$ satisfy:
\begin{equation}
\sum_{v \in V} [\widetilde{P}_{\text{in}}]_{uv} = \sum_{v \in V} \frac{P_{vu}}{d_{\text{in}}(u) + \eta} = \frac{\sum_v P_{vu}}{d_{\text{in}}(u) + \eta} = \frac{d_{\text{in}}(u)}{d_{\text{in}}(u) + \eta} \le \frac{d_{\text{in}}^{\max}}{d_{\text{in}}^{\max} + \eta} < 1, \quad \forall u \in V.
\end{equation}
Therefore, the induced infinity norm satisfies $\|\widetilde{P}_{\text{in}}\|_\infty = \max_{u \in V} \frac{d_{\text{in}}(u)}{d_{\text{in}}(u) + \eta} < 1$. By Gelfand's formula, the spectral radius satisfies $\rho(\widetilde{P}_{\text{in}}) \le \|\widetilde{P}_{\text{in}}\|_\infty < 1$.

Consequently, $(I - \widetilde{P}_{\text{in}})^{-1}$ exists and admits the unconditionally convergent Neumann series $\sum_{k=0}^\infty (\widetilde{P}_{\text{in}})^k$. Taking the inverse of $D_\eta (I - \widetilde{P}_{\text{in}})$:
\begin{equation}
\mathcal{G}_{\text{in}} = \left( I - \widetilde{P}_{\text{in}} \right)^{-1} D_\eta^{-1} = \sum_{k=0}^\infty (\widetilde{P}_{\text{in}})^k D_\eta^{-1} = \sum_{k=0}^\infty (D_\eta^{-1} P^T)^k D_\eta^{-1}.
\end{equation}
Because every term in this expansion is a product of entrywise non-negative matrices, we have $\mathcal{G}_{\text{in}, ij} \ge 0$ for all $i, j \in V$.

Furthermore, evaluating $(D_{\text{in}} - P^T + \eta I) \mathbf{1}$:
\begin{equation}
(D_{\text{in}} - P^T + \eta I) \mathbf{1} = D_{\text{in}} \mathbf{1} - P^T \mathbf{1} + \eta \mathbf{1} = P^T \mathbf{1} - P^T \mathbf{1} + \eta \mathbf{1} = \eta \mathbf{1}.
\end{equation}
Multiplying both sides by $\mathcal{G}_{\text{in}} = (D_{\text{in}} - P^T + \eta I)^{-1}$:
\begin{equation}
\mathbf{1} = \eta \, \mathcal{G}_{\text{in}} \mathbf{1} \implies \mathcal{G}_{\text{in}} \mathbf{1} = \frac{1}{\eta} \mathbf{1}.
\end{equation}
Since all entries of $\mathcal{G}_{\text{in}}$ are non-negative, every row sum of $\mathcal{G}_{\text{in}}$ is identically equal to $\frac{1}{\eta}$. Thus:
\begin{equation}
\|\mathcal{G}_{\text{in}}\|_\infty = \max_{u \in V} [\mathcal{G}_{\text{in}} \mathbf{1}]_u = \frac{1}{\eta}.
\end{equation}
This completes the proof.
\end{proof}

"""

text = text[:idx_lem1] + new_lemma1_text + text[idx_lem2:]
print("Updated Lemma 1 in Appendix B.")

with open('idea.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print("Saved updated idea.tex! New length:", len(text))
