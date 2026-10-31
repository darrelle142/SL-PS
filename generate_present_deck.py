latex_content = r"""\documentclass[aspectratio=169, 10pt]{beamer}

% --- Typography & Encodings ---
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[english]{babel}
\usepackage{amsmath, amssymb, amsfonts, amsthm}
\usepackage{bm}
\usepackage{booktabs}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, shapes.geometric, calc}
\usepackage{tcolorbox}
\tcbuselibrary{skins}

% --- Modern Marp-Inspired Color Palette ---
\definecolor{MarpDark}{RGB}{15, 23, 42}         % Slate 900
\definecolor{MarpBlue}{RGB}{37, 99, 235}        % Royal Blue 600
\definecolor{MarpCyan}{RGB}{14, 165, 233}       % Sky Cyan 500
\definecolor{MarpEmerald}{RGB}{16, 185, 129}    % Emerald Green 500
\definecolor{MarpCrimson}{RGB}{225, 29, 72}     % Crimson Red 600
\definecolor{MarpAmber}{RGB}{245, 158, 11}      % Amber 500
\definecolor{MarpCardBg}{RGB}{248, 250, 252}    % Slate 50
\definecolor{MarpCardBorder}{RGB}{226, 232, 240}% Slate 200
\definecolor{MarpTextDark}{RGB}{30, 41, 59}     % Slate 800
\definecolor{MarpTextMuted}{RGB}{100, 116, 139} % Slate 500

% --- Clean Modern Theme Configuration ---
\setbeamertemplate{navigation symbols}{}
\setbeamercolor{background canvas}{bg=white}
\setbeamercolor{normal text}{fg=MarpTextDark}
\setbeamerfont{frametitle}{size=\small, series=\bfseries}
\setbeamercolor{frametitle}{fg=MarpDark, bg=white}

% Custom Card Environment (Marp-like box)
\newcommand{\marpcard}[3]{%
    \begin{tcolorbox}[
        colback=MarpCardBg,
        colframe=MarpCardBorder,
        coltitle=white,
        colbacktitle=#1,
        title=\textbf{#2},
        boxrule=0.5pt,
        arc=2pt,
        left=3pt, right=3pt, top=1pt, bottom=1pt,
        fonttitle=\scriptsize\bfseries
    ]
    #3
    \end{tcolorbox}%
}

\newcommand{\marpbox}[2]{%
    \begin{tcolorbox}[
        colback=MarpCardBg,
        colframe=#1,
        boxrule=0.5pt,
        arc=2pt,
        left=3pt, right=3pt, top=1pt, bottom=1pt
    ]
    #2
    \end{tcolorbox}%
}

% Custom Header and Footer with properly grouped colored rules
\setbeamertemplate{headline}{
    \vskip2pt
    \hspace{10pt}\textcolor{MarpTextMuted}{\tiny\textbf{SPECTRAL-LAPLACIAN PRIORITIZED SWEEPING (SL-PS)}}\hfill
    \textcolor{MarpTextMuted}{\tiny\textbf{THEORETICAL FOUNDATIONS \& ALGORITHMIC FRAMEWORK}}\hspace{10pt}
    \vskip1pt{\color{MarpCardBorder}\hrule height 0.4pt}\vskip1pt
}

\setbeamertemplate{footline}{
    \vskip1pt{\color{MarpCardBorder}\hrule height 0.4pt}\vskip1.5pt
    \hspace{10pt}\textcolor{MarpTextMuted}{\tiny Advanced Dynamic Programming in Congestion Games}\hfill
    \textcolor{MarpBlue}{\textbf{\insertframenumber}}\,/\,\textcolor{MarpTextMuted}{\inserttotalframenumber}\hspace{10pt}
    \vskip1.5pt
}

\begin{document}

% ==========================================
% SLIDE 1: Opening (Cover Slide)
% ==========================================
\begin{frame}[plain]
\begin{tikzpicture}[remember picture, overlay]
    \fill[MarpDark] (current page.south west) rectangle (current page.north east);
    \fill[MarpBlue] ($(current page.north west) + (0, -0.4)$) rectangle ($(current page.north east) + (0, -0.46)$);
    \node[anchor=south west, text=MarpCyan, font=\tiny\bfseries] at ($(current page.south west) + (0.8, 0.6)$) {
        MODEL-BASED REINFORCEMENT LEARNING \quad$\bullet$\quad SPECTRAL GRAPH THEORY \quad$\bullet$\quad WARDROP EQUILIBRIUM
    };
\end{tikzpicture}

\vskip18pt
\begin{centering}
    \textcolor{MarpCyan}{\scriptsize\textbf{THEORETICAL ANALYSIS \& ALGORITHMIC FRAMEWORK}}\\[8pt]
    \textcolor{white}{\textbf{\LARGE Spectral-Laplacian Prioritized Sweeping}}\\[4pt]
    \textcolor{white}{\textbf{\Large (SL-PS)}}\\[8pt]
    \textcolor{gray!40}{\small Resolving Braess's Paradox via Resolvent Value Warm-Start \& Canonical DP}\\[18pt]
    
    \textcolor{white}{\textbf{\small Author \& Research Team}}\\[2pt]
    \textcolor{gray!50}{\scriptsize Laboratory for Artificial Intelligence \& Optimization Theory}\\[10pt]
    \textcolor{MarpEmerald}{\scriptsize\textbf{Core Tracks: Model-Based RL $\times$ Spectral Graph Theory $\times$ Congestion Games}}
\end{centering}
\end{frame}

% ==========================================
% SLIDE 2: Network Congestion Architecture, Latency, Laplacians
% ==========================================
\begin{frame}{Network Congestion Architecture, Link Latency, and Graph Laplacians}
\begin{columns}[T]
\begin{column}{0.49\textwidth}
    \marpcard{MarpBlue}{Congestion Network $G = (V, E_0)$ \& Link Latency $\ell_e$}{
        \scriptsize
        \begin{itemize}\setlength\itemsep{1pt}
            \item Finite directed network $G = (V, E_0)$ with $|V| = n$ vertices and $|E_0| = m$ base edges.
            \item Each base link $e = (u \to w) \in E_0$ carries an \textbf{affine load latency}:
            \begin{equation*}
            \ell_e(f_e) = a_e f_e + b_e, \quad a_e > 0, \; b_e \ge 0
            \end{equation*}
            ($a_e$: congestion sensitivity; $b_e$: free-flow travel latency).
            \item Total background demand $q > 0$ from origin $s$ to sink $t$, creating net injection: $p = q(e_s - e_t) \in \mathbb{R}^{|V|}$.
        \end{itemize}
    }
\end{column}

\begin{column}{0.49\textwidth}
    \marpcard{MarpDark}{Graph Operators: Incidence Matrix $B$ \& Laplacian $L$}{
        \scriptsize
        \begin{itemize}\setlength\itemsep{1pt}
            \item \textbf{Directed Vertex--Edge Incidence Matrix} $B \in \mathbb{R}^{|V| \times |E_0|}$:
            \begin{equation*}
            B_{v, u \to w} = \begin{cases} +1, & \text{if } v = u \text{ (tail)} \\ -1, & \text{if } v = w \text{ (head)} \\ 0, & \text{otherwise} \end{cases}
            \end{equation*}
            \item \textbf{Conductance}: $W = \operatorname{diag}(a_e^{-1})$, \textbf{Laplacian}: $L = B W B^T \in \mathbb{R}^{|V| \times |V|}$.
            \item SPSD properties: $\operatorname{ker}(L) = \operatorname{span}(\mathbf{1})$.
            \item \textbf{Baseline Electrical Potential}: $\phi = L^+ p = q L^+ (e_s - e_t)$.
        \end{itemize}
    }
\end{column}
\end{columns}

\vskip2pt
\marpbox{MarpCyan}{
    \centering\tiny
    \textbf{Analytical Bridge}: Matrices $B$ and $L$ form an exact discrete differential calculus between edge flow space ($|E_0|$) and vertex potential space ($|V|$).
}
\end{frame}

% ==========================================
% SLIDE 3: CC-DG-MDP Environment
% ==========================================
\begin{frame}{The Congestion-Coupled Dynamic Graph MDP $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{C}, \gamma)$}
\begin{columns}[T]
\begin{column}{0.49\textwidth}
    \marpcard{MarpBlue}{State Space $\mathcal{S}$ \& Action Space $\mathcal{A}$}{
        \scriptsize
        \begin{itemize}\setlength\itemsep{1pt}
            \item \textbf{State Space $\mathcal{S}$}: Physical vertices $s_t = u \in V$.
            \item \textbf{Absorbing Destination Sink} $t_{\text{sink}} \in V$: Zero stage cost ($c(t_{\text{sink}}, \cdot) = 0$, $\mathcal{P}(t_{\text{sink}} \mid t_{\text{sink}}, \cdot) = 1$), ensuring $V^*(t_{\text{sink}}) = 0$.
            \item \textbf{Action Space $\mathcal{A}(u)$}: Active outgoing directed links:
            \begin{equation*}
            \mathcal{A}(u) = \{ v \in V \mid (u \to v) \in E_t \}
            \end{equation*}
            Unconstrained by 2D DAG grids, permitting full topological traversal across arbitrary directed graphs.
        \end{itemize}
    }
\end{column}

\begin{column}{0.49\textwidth}
    \marpcard{MarpEmerald}{Topology Dynamics $\mathcal{P}$, Stage Cost $\mathcal{C}$ \& Discount $\gamma$}{
        \scriptsize
        \begin{itemize}\setlength\itemsep{1pt}
            \item \textbf{Stochastic Topology Dynamics $\mathcal{P}$}: Shortcut chord $e^* = (u \to v)$ emerges stochastically with probability $p_{\text{chord}}$, yielding dynamic edge set $E_t = E_0 \cup \{e^*\}$.
            \item \textbf{Shortcut Link}: Nominal affine latency $\ell^*(z) = c z + d$.
            \item \textbf{Step Cost $\mathcal{C}$}: $c(u, v)$ couples endogenous equilibrium load $f_e^*(E_t)$ with directional detour offsets.
            \item \textbf{Discount Factor $\gamma \in (0, 1)$}: Governs infinite-horizon planning.
        \end{itemize}
    }
\end{column}
\end{columns}

\vskip2pt
\marpbox{MarpDark}{
    \centering\tiny
    \textbf{Problem Formulation}: \textbf{CC-DG-MDP} formalizes a two-tier coupling: stochastic topological link emergence drives macroscopic endogenous flow re-equilibration.
}
\end{frame}

% ==========================================
% SLIDE 4: WUE & Beckmann Optimization
% ==========================================
\begin{frame}{Wardrop User Equilibrium (WUE) \& Beckmann Optimization}
\begin{columns}[T]
\begin{column}{0.49\textwidth}
    \marpcard{MarpBlue}{Wardrop User Equilibrium (WUE)}{
        \scriptsize
        \textbf{Wardrop's First Principle (1952)}: Under selfish background commuter routing, all active paths $p \in \mathcal{P}_{st}$ with non-zero flow ($h_p^* > 0$) share an identical minimal latency $\mu_{st}$:
        \begin{equation*}
        \begin{cases}
        c_p(f^*) = \sum_{e \in p} \ell_e(f_e^*) = \mu_{st}, & \text{if } h_p^* > 0, \\
        c_p(f^*) \ge \mu_{st}, & \text{if } h_p^* = 0.
        \end{cases}
        \end{equation*}
        No commuter can unilaterally switch routes to reduce latency. Total travel time: $C(f) = \sum_{e} f_e \ell_e(f_e)$.
    }
\end{column}

\begin{column}{0.49\textwidth}
    \marpcard{MarpEmerald}{Convex Beckmann Potential Optimization}{
        \scriptsize
        Equilibrium flow $f^*(E_t)$ is the unique global minimizer of the strictly convex Beckmann program (1956):
        \begin{equation*}
        \min_{f \ge 0} \Phi_{\text{Beck}}(f) \equiv \sum_{e \in E_t} \int_0^{f_e} \ell_e(x) \, dx \quad \text{s.t.} \quad B_{E_t} f = p
        \end{equation*}
        First-order KKT optimality conditions are mathematically identical to WUE!
    }
    \vskip1.5pt
    \marpcard{MarpDark}{Agent Flow Approximation (Scale Separation)}{
        \scriptsize
        $\bullet$ Macroscopic flow $f^*$ arises from thousands of selfish drivers; the ego-agent is an infinitesimal micro-vehicle.\\
        $\bullet$ Agent precomputes nominal flow $f_{\text{base}}^*$ \textbf{offline once}; upon chord emergence, it predicts flow shift $\Delta \vec{f}$ to evade traps.
    }
\end{column}
\end{columns}
\end{frame}

% ==========================================
% SLIDE 5: Cost Function Components
% ==========================================
\begin{frame}{Structural Decomposition of Step Cost Function $c(u, v)$}
\marpbox{MarpBlue}{
    \centering\scriptsize
    $c(u, v) = \ell_e(f_e^*(E_t)) + \delta_{\text{detour}} \cdot \mathbb{I}_{\text{ret}}(u \to v) + \xi_t$
}

\vskip1pt
\begin{columns}[T]
\begin{column}{0.32\textwidth}
    \marpcard{MarpCrimson}{1. Endogenous Latency $\ell_e(f_e^*)$}{
        \scriptsize
        \begin{itemize}\setlength\itemsep{0.5pt}
            \item Base edge $e \in E_0$:
            $$\ell_e(f_e^*) = a_e f_e^*(E_t) + b_e$$
            \item Shortcut chord $e^* = (u \to v)$:
            $$\ell^*(z^*) = c z^* + d$$
            \item As $z^* \to 0$, shortcut nominal latency appears cost-free ($c \approx 0$).
            \item Physical law: Swarming traffic causes downstream delays to surge!
        \end{itemize}
    }
\end{column}

\begin{column}{0.34\textwidth}
    \marpcard{MarpEmerald}{2. Retrograde Penalty $\delta_{\text{det}} \mathbb{I}_{\text{ret}}$}{
        \scriptsize
        \begin{itemize}\setlength\itemsep{0.5pt}
            \item Let $d(x) \equiv \operatorname{dist}_G(x, t_{\text{sink}})$:
            $$\mathbb{I}_{\text{ret}} = \begin{cases} 1, & d(v) \ge d(u) \\ 0, & d(v) < d(u) \end{cases}$$
            \item Eliminates zero-gain dithering oscillations on reversible networks.
            \item \textbf{Key property}: Braess chords advance forward ($d(v) < d(u)$), so $\mathbb{I}_{\text{ret}} = 0$; penalty never masks shortcut appeal!
        \end{itemize}
    }
\end{column}

\begin{column}{0.32\textwidth}
    \marpcard{MarpDark}{3. Observation Noise \& Sink}{
        \scriptsize
        \begin{itemize}\setlength\itemsep{0.5pt}
            \item \textbf{Stochastic Noise $\xi_t$}:
            $$\xi_t \sim \mathcal{N}(0, \sigma_\xi^2)$$
            Models real-time sensor measurement errors.
            \item \textbf{Destination Absorption} $t_{\text{sink}}$:
            $$c(t_{\text{sink}}, a) = 0, \quad \forall a$$
            Ensures true optimal value $V^*(t_{\text{sink}}) = 0$.
        \end{itemize}
    }
\end{column}
\end{columns}
\end{frame}

% ==========================================
% SLIDE 6: Braess's Paradox
% ==========================================
\begin{frame}{Braess's Paradox in Congestion-Coupled Graphs}
\begin{columns}[T]
\begin{column}{0.49\textwidth}
    \marpcard{MarpCrimson}{Nature of Braess's Paradox}{
        \scriptsize
        \begin{itemize}\setlength\itemsep{0.5pt}
            \item \textbf{Paradox}: Opening a new shortcut link $e^* = (u \to v)$ with nominal cost $c \approx 0$ paradoxically \textbf{increases total travel time}:
            \begin{equation*}
            C(f_{\text{aug}}^*) > C(f_{\text{base}}^*)
            \end{equation*}
            \item \textbf{Mechanism}: Selfish drivers swarm into the shortcut, causing severe bottlenecks downstream of exit vertex $v$.
            \item \textbf{Failure of Standard-PS}: Deceived by local nominal cost $c \approx 0$, it plunges into the trap; congestion takes $\mathcal{O}(\operatorname{diam}(G))$ updates to propagate upstream.
        \end{itemize}
    }
\end{column}

\begin{column}{0.49\textwidth}
    \marpcard{MarpBlue}{Generalized Voltage Criterion (Theorem 1)}{
        \scriptsize
        \begin{itemize}\setlength\itemsep{0.5pt}
            \item Shortcut diverts flow ($z^* > 0$) if and only if:
            \begin{equation*}
            \Delta b_{uv} + V_{uv} > d
            \end{equation*}
            where $V_{uv} = \phi_u - \phi_v = q (e_s - e_t)^T L^+ (e_u - e_v)$.
            \item \textbf{Braess's paradox occurs whenever flow opposes electrical gradient}:
            \begin{equation*}
            V_{uv} = \phi_u - \phi_v < 0
            \end{equation*}
            \item Network delay increase: $\Delta C \le |V_{uv}| z^* > 0$.
            \item On grids $G_{m,n}$, $\Delta b_{uv} = k \cdot b$ ($k > 0$) and $V_{uv} < 0$.
        \end{itemize}
    }
\end{column}
\end{columns}

\vskip2pt
\marpbox{MarpAmber}{
    \centering\tiny
    \textbf{Consequence}: Standard RL planners greedily enter the deceptive shortcut and incur catastrophic cumulative delay $\Omega(q)$.
}
\end{frame}

% ==========================================
% SLIDE 7: Core Components of SL-PS
% ==========================================
\begin{frame}{Core Components of SL-PS: $\Delta \vec{f}$, $\Delta \vec{\ell}$, and Divergence $\xi_{\text{div}}$}
\begin{columns}[T]
\begin{column}{0.49\textwidth}
    \marpcard{MarpBlue}{1. Base Edge Flow Shift $\Delta \vec{f} \in \mathbb{R}^{|E_0|}$}{
        \scriptsize
        \begin{equation*}
        \Delta \vec{f} = f_{\text{aug}, E_0}^* - f_{\text{base}}^*
        \end{equation*}
        \begin{itemize}\setlength\itemsep{0pt}
            \item $f_{\text{base}}^* \in \mathbb{R}^{|E_0|}$: Base equilibrium flow on original network $E_0$.
            \item $f_{\text{aug}, E_0}^*$: Re-equilibrated flow on $E_0$ after shortcut chord $e^*$ opens.
            \item \textbf{Meaning}: Relieves load on alternate paths ($\Delta f_e < 0$) while overloading downstream bottleneck links ($\Delta f_e > 0$).
        \end{itemize}
    }
    \vskip1pt
    \marpcard{MarpEmerald}{2. Link Latency Shift $\Delta \vec{\ell} \in \mathbb{R}^{|E_0|}$}{
        \scriptsize
        \begin{equation*}
        \Delta \vec{\ell} = \operatorname{diag}(a_e) \Delta \vec{f}
        \end{equation*}
        Since $\ell_e(f_e) = a_e f_e + b_e$, link derivative is $a_e$. Latency shift is $\Delta \ell_e = a_e \Delta f_e$ (seconds).
    }
\end{column}

\begin{column}{0.49\textwidth}
    \marpcard{MarpCrimson}{3. Vertex Congestion Divergence $\xi_{\text{div}} \in \mathbb{R}^{|V|}$}{
        \scriptsize
        Projects link latency shifts onto vertices via incidence matrix $B$:
        \begin{equation*}
        \xi_{\text{div}} = |B \Delta \vec{\ell}| \in \mathbb{R}^{|V|}
        \end{equation*}
        \textbf{Explicit expansion at each vertex $v \in V$}:
        \vskip-2pt
        {\centering\scriptsize$\displaystyle\xi_{\text{div}}(v) = \left| \sum_{v \to w} a_e \Delta f_e - \sum_{u \to v} a_e \Delta f_e \right|$\par}
        \vskip1pt
        \textbf{Physical Interpretation}:
        \begin{itemize}\setlength\itemsep{0pt}
            \item $B$ serves as a \textbf{discrete divergence operator}, measuring local net latency flux imbalance.
            \item At shortcut exit $v^*$, massive convergent traffic creates a divergence shock: $\xi_{\text{div}}(v^*) = \Omega(q)$.
            \item Absolute value $|\cdot|$ provides positive source excitation for the spectral potential.
        \end{itemize}
    }
\end{column}
\end{columns}
\end{frame}

% ==========================================
% SLIDE 8: P_flow, G_out, phi_spec
% ==========================================
\begin{frame}{Flow Kernel $P_{\text{flow}}$, Green Operator $\mathcal{G}_{\text{out}}$, and Potential $\Phi_{\text{spec}}$}
\begin{columns}[T]
\begin{column}{0.49\textwidth}
    \marpcard{MarpBlue}{Flow-Preserving Markov Kernel $P_{\text{flow}}$}{
        \scriptsize
        Transition probabilities follow actual commuter choices:
        \begin{equation*}
        P_{u, v}^{\text{flow}} = \frac{f_{\text{base}, (u \to v)}^*}{\sum_{w} f_{\text{base}, (u \to w)}^*}, \quad P_{t_{\text{sink}}, t_{\text{sink}}}^{\text{flow}} = 1
        \end{equation*}
        Completely eliminates branching over-diffusion on asymmetric graph topologies.
    }
    \vskip1pt
    \marpcard{MarpEmerald}{Green Operator $\mathcal{G}_{\text{out}}$ \& Successor Representation}{
        \scriptsize
        Out-Laplacian resolvent: $\mathcal{G}_{\text{out}} = (I - P + \eta I)^{-1} = \sum_{k=0}^{\infty} \frac{P^k}{(1 + \eta)^{k+1}}$.\\
        Setting $\eta = \frac{1-\gamma}{\gamma}$ yields the exact algebraic identity:
        \begin{equation*}
        \mathcal{G}_{\text{out}} \equiv \gamma (I - \gamma P)^{-1} \equiv \gamma \mathcal{R}_\gamma \quad (\text{SR Resolvent})
        \end{equation*}
    }
\end{column}

\begin{column}{0.49\textwidth}
    \marpcard{MarpCrimson}{Spectral Congestion Potential $\Phi_{\text{spec}}$}{
        \scriptsize
        \begin{equation*}
        \Phi_{\text{spec}} \equiv \frac{1}{\gamma} \mathcal{G}_{\text{out}} \xi_{\text{div}} = (I - \gamma P)^{-1} \xi_{\text{div}} = \sum_{k=0}^\infty \gamma^k P^k \xi_{\text{div}}
        \end{equation*}
        $\bullet$ \textbf{Infinite-Horizon Projection}: Linear solve $(I - \gamma P)^{-1} \xi_{\text{div}}$ projects downstream delays upstream in $\mathcal{O}(1)$ time.\\
        $\bullet$ \textbf{Dimensionally Consistent}: Measured in seconds of delay.
    }
    \vskip1pt
    \marpcard{MarpDark}{Spectral Resolvent Value Warm-Start}{
        \scriptsize
        \begin{equation*}
        V^{(0)}(s) = V_{\text{base}}^*(s) + \beta(\sigma) \Phi_{\text{spec}}(s)
        \end{equation*}
        with $\beta(\sigma) = \beta_0 (1 + \sigma(P))$. Instantly elevates bottleneck value $V^{(0)} \implies$ \textbf{Deterministic trap immunity at zero budget ($K=0$)}!
    }
\end{column}
\end{columns}
\end{frame}

% ==========================================
% SLIDE 9: Algorithm (alg.png)
% ==========================================
\begin{frame}{Comprehensive SL-PS Algorithmic Workflow (Algorithm 1)}
\begin{columns}[c]
\begin{column}{0.46\textwidth}
    \begin{tcolorbox}[
        colback=white,
        colframe=MarpBlue,
        boxrule=0.7pt,
        arc=3pt,
        left=1pt, right=1pt, top=1pt, bottom=1pt
    ]
        \centering
        \includegraphics[width=\textwidth, height=0.75\textheight, keepaspectratio]{alg.png}
    \end{tcolorbox}
\end{column}

\begin{column}{0.52\textwidth}
    \marpcard{MarpDark}{Phase 1: Offline Precomputation}{
        \tiny
        $\bullet$ Compute $L, L^+$, potential $\phi = q L^+(e_s - e_t)$, base flow $f_{\text{base}}^*$, base value $V_{\text{base}}^*$.\\
        $\bullet$ Construct $P_{\text{flow}}$, SLEM $\sigma(P)$, and Green operators $\mathcal{G}_{\text{out}}, \mathcal{G}_{\text{in}}$.\\
        $\bullet$ Heavy matrix operations are solved entirely offline!
    }
    \vskip1.5pt
    \marpcard{MarpCrimson}{Phase 2: Instantaneous Warm-Start ($K = 0$ Shield)}{
        \tiny
        $\bullet$ Upon chord emergence $e^*$: compute $\Delta \vec{f}$, $\xi_{\text{div}} = |B (a \Delta \vec{f})|$, and $\Phi_{\text{spec}}$.\\
        $\bullet$ Warm-start value function: $V^{(0)} \leftarrow V_{\text{base}}^* + \beta(\sigma) \Phi_{\text{spec}}$.\\
        $\bullet$ \textbf{100\% Trap Immunity}: Bottleneck state value is elevated truthfully $\implies$ Agent circumvents trap with zero sweeping sweeps!
    }
    \vskip1.5pt
    \marpcard{MarpEmerald}{Phase 3: Canonical Swept Backups ($K \ge 1$ Optimizer)}{
        \tiny
        $\bullet$ Backward Green potential scheduling: $f_{\text{in}} = \mathcal{G}_{\text{in}}(\xi_{\text{res}} + \xi_{\text{div}})$.\\
        $\bullet$ Prioritizes upstream branching junctions, eliminating queue starvation.\\
        $\bullet$ Executes canonical Bellman backups $\mathcal{T}V$ to close detour gap toward Pareto optimality.
    }
\end{column}
\end{columns}
\end{frame}

% ==========================================
% SLIDE 10: Theoretical Guarantees Suite
% ==========================================
\begin{frame}{Theoretical Foundations \& Key Theorems Suite}
\begin{columns}[T]
\begin{column}{0.49\textwidth}
    \marpbox{MarpDark}{
        \textcolor{MarpBlue}{\textbf{\scriptsize Theorem 2: Strict Contraction Mapping}}
        \vskip0.5pt
        \tiny
        $\|\mathcal{T} V_1 - \mathcal{T} V_2\|_\infty \le \gamma \|V_1 - V_2\|_\infty$. Canonical Bellman operator guarantees global convergence to the true physical optimum $V^*$, uncorrupted by heuristic penalties.
    }
    \vskip1pt
    \marpbox{MarpDark}{
        \textcolor{MarpCrimson}{\textbf{\scriptsize Theorem 3: Tight Immunity \& Vanishing Threshold}}
        \vskip0.5pt
        \tiny
        Greedy policy at $K=0$ avoids Braess trap if $\beta_0 \ge \beta_{\min}^*(u) = \frac{\Delta Q_{\text{base}}(u)}{\gamma(1+\sigma(P))\Delta \Phi_{\text{spec}}(u)}$. Under heavy traffic ($q \to \infty$), $\Delta \Phi_{\text{spec}} = \Omega(q)$ forces $\beta_{\min}^* \to 0$! Any nominal $\beta_0 = 1$ is unconditionally immune.
    }
    \vskip1pt
    \marpbox{MarpEmerald}{
        \textcolor{MarpEmerald}{\textbf{\scriptsize Theorem 4: Pareto-Optimal Convergence}}
        \vskip0.5pt
        \tiny
        Value error contracts geometrically after $K$ sweeps: $\|V^{(K)} - V^*\|_\infty \le \gamma^K \epsilon \to 0$. Rapidly closes detour regret gap.
    }
\end{column}

\begin{column}{0.49\textwidth}
    \marpbox{MarpDark}{
        \textcolor{MarpCrimson}{\textbf{\scriptsize Theorem 5: Asymptotic Policy Divergence}}
        \vskip0.5pt
        \tiny
        Cumulative travel latency gap between Standard-PS and SL-PS diverges:
        $\mathbb{E}[T_{\text{PS}} - T_{\text{SL-PS}}] = \Omega(q \cdot \operatorname{diam}(G)) \to \infty$. Canonical PS fails catastrophically on large networks.
    }
    \vskip1pt
    \marpbox{MarpDark}{
        \textcolor{MarpCyan}{\textbf{\scriptsize Theorem 6: Lipschitz Fault Tolerance}}
        \vskip0.5pt
        \tiny
        Under flow estimation errors $\|\hat{f} - f^*\| \le \epsilon_f$: $\|V^{(0)} - \hat{V}^{(0)}\|_\infty \le \mathcal{L}_{\text{SL}} \epsilon_f$. Strictly positive tolerance margin $\epsilon_f^* > 0$ preserves zero-trap immunity.
    }
    \vskip1pt
    \marpbox{MarpDark}{
        \textcolor{MarpBlue}{\textbf{\scriptsize Theorem 7: Policy--Resolvent Mismatch}}
        \vskip0.5pt
        \tiny
        Resolvent Identity II: $\|\Phi(P) - \Phi(P_{\pi^*})\|_\infty \le \frac{\gamma}{(1-\gamma)^2} \|\xi\|_\infty \epsilon_P$. Flow kernel $P_{\text{flow}}$ eliminates error; unweighted $P_{\text{unif}}$ remains provably safe for $\epsilon_P < \epsilon_P^*$.
    }
\end{column}
\end{columns}

\vskip1pt
\marpbox{MarpEmerald}{
    \centering\tiny
    \textbf{Core Takeaway}: SL-PS reconciles \textbf{Instantaneous Safety} at $K=0$ with \textbf{Canonical Optimality} at $K \ge 1$ across general arbitrary networks.
}
\end{frame}

\end{document}
"""

with open("present.tex", "w", encoding="utf-8") as f:
    f.write(latex_content)

print("Updated present.tex cleanly!")
