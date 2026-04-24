"""V4.0 Uniqueness test — baseline parameterized systems.

Each baseline exposes a uniform interface::

    system = Baseline(n_params, seed)
    importance = system.parameter_importance(n_probes)   # (n_params,) float32 array

``parameter_importance`` returns a generalized Fisher-information-analogue:
the expected squared sensitivity of the system's output(s) to each internal
parameter, averaged over a bank of random probe inputs. For a stochastic
model under log-likelihood, this reduces to the empirical FIM diagonal; for
a deterministic system, it is the output-Jacobian Gram-diagonal. Either way
the dimensional content is the same and the tier-ratio analysis is apples-
to-apples across substrates.

Six baselines are implemented:

1. NeuralNetwork      — 5-layer ReLU MLP, reference system (FIM proper)
2. RandomMatrix       — Gaussian Hermitian ensemble, eigenvalue importance
3. IsingChain         — 1D Ising, magnetization susceptibility to local fields
4. HarmonicOscillator — coupled 1D harmonic chain, spring-stiffness sensitivity
5. BooleanCircuit     — random Boolean gate circuit, output flip-sensitivity
6. CellularAutomaton  — elementary CA (Rule-110), pattern-divergence sensitivity

Goal: ask whether the FIM three-tier hierarchy (Tier-1 / Tier-3 ratio,
Tier-1 share of mass) is neural-network-specific or generic information-
geometry phenomenology.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class Baseline:
    """Base class. Subclasses must implement ``parameter_importance``."""

    name: str = "baseline"

    def __init__(self, n_params: int, seed: int) -> None:
        self.n_params = int(n_params)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

    def parameter_importance(self, n_probes: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Neural network — reference / control
# ---------------------------------------------------------------------------


class NeuralNetwork(Baseline):
    """5-layer ReLU MLP on self-prediction. FIM diagonal = grad-squared average."""

    name = "neural_network"

    def __init__(self, n_params: int, seed: int, dim: int = 16) -> None:
        super().__init__(n_params, seed)
        torch.manual_seed(seed)
        # Pick hidden width that yields roughly n_params.
        # 5-layer: n_params ≈ dim*h + 3*h*h + h*dim  ≈ 3*h^2 for large h.
        h = max(8, int(np.sqrt(n_params / 3)))
        self.dim = dim
        layers: list[nn.Module] = [nn.Linear(dim, h), nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(h, h), nn.ReLU()]
        layers.append(nn.Linear(h, dim))
        self.net = nn.Sequential(*layers)
        # Train long enough for the FIM spectrum to stabilize — matches the
        # training-step count used in V1.2 seed_robustness so CVs compare
        # apples-to-apples. Shorter training produces high tier-ratio CV
        # (the distribution hasn't concentrated yet) which would mask the
        # real uniqueness signal.
        opt = torch.optim.SGD(self.net.parameters(), lr=1e-3, momentum=0.9)
        for _ in range(20000):
            x = torch.randn(64, dim)
            y = self.net(x)
            loss = 0.5 * (y - x).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    def parameter_importance(self, n_probes: int) -> np.ndarray:
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(True)
        fim: list[torch.Tensor] = [torch.zeros_like(p) for p in self.net.parameters()]
        for _ in range(n_probes):
            x = torch.randn(1, self.dim)
            loss = 0.5 * (self.net(x) - x).pow(2).mean()
            self.net.zero_grad(set_to_none=True)
            loss.backward()
            for i, p in enumerate(self.net.parameters()):
                if p.grad is not None:
                    fim[i] += p.grad.data ** 2
        flat = torch.cat([t.flatten() for t in fim]).cpu().numpy() / n_probes
        return flat.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Random Hermitian matrix ensemble
# ---------------------------------------------------------------------------


class RandomMatrix(Baseline):
    """GOE-like Hermitian matrix. Parameters = upper-triangle entries.

    Output: top-k eigenvalues. Importance(i,j) = d(sum of top-k eig) / d M_ij.
    For a Hermitian matrix this is the (i,j) entry of the projector onto the
    top-k eigenspace — a well-defined linear-response susceptibility.
    """

    name = "random_matrix"

    def __init__(self, n_params: int, seed: int) -> None:
        super().__init__(n_params, seed)
        # n_params = N*(N+1)/2 upper-triangle entries. Solve for N.
        N = int((-1 + np.sqrt(1 + 8 * n_params)) / 2)
        self.N = max(8, N)
        # Gaussian symmetric matrix, scaled so eigenvalues are O(1).
        m = self.rng.standard_normal((self.N, self.N)).astype(np.float32)
        self.M = (m + m.T) / np.sqrt(2 * self.N)

    def parameter_importance(self, n_probes: int) -> np.ndarray:
        """Importance of each upper-triangle entry = (projector onto top-k)^2.

        We generalize ``n_probes`` by averaging susceptibility over a band of
        top-k choices k ∈ {1, 2, 4, 8, ..., N/2} — each probe is one choice
        of k, consistent with the "random probe" interpretation of FIM.
        """
        k_choices = [2 ** i for i in range(int(np.log2(self.N // 2)))][: max(1, n_probes)]
        if not k_choices:
            k_choices = [1]
        eigvals, eigvecs = np.linalg.eigh(self.M)
        # Sort descending
        order = np.argsort(-eigvals)
        eigvecs = eigvecs[:, order]
        imp = np.zeros((self.N, self.N), dtype=np.float32)
        for k in k_choices:
            # Projector onto top-k eigenspace
            P = eigvecs[:, :k] @ eigvecs[:, :k].T  # (N, N)
            imp += P ** 2
        imp /= len(k_choices)
        # Extract upper triangle (including diagonal) in row-major order
        iu = np.triu_indices(self.N)
        return imp[iu].astype(np.float32)


# ---------------------------------------------------------------------------
# 3. 1D Ising chain (classical, no training)
# ---------------------------------------------------------------------------


class IsingChain(Baseline):
    """Mean-field 1D Ising. Parameters = local fields h_i (i = 1..N).

    Importance(i) = |chi_ii| = d<s_i> / d h_i = beta * (1 - <s_i>^2).
    Under a Gaussian random field ensemble this produces a heavy-tailed
    importance distribution by construction.
    """

    name = "ising_chain"

    def __init__(self, n_params: int, seed: int, beta: float = 1.0) -> None:
        super().__init__(n_params, seed)
        self.N = n_params
        self.beta = beta
        # Random external field (this is the "parameter" we're perturbing)
        self.h = self.rng.standard_normal(self.N).astype(np.float32) * 0.5
        # Random coupling J_i on bonds (fixed, not a parameter)
        self.J = self.rng.standard_normal(self.N - 1).astype(np.float32) * 0.3

    def parameter_importance(self, n_probes: int) -> np.ndarray:
        """Local susceptibility averaged over ``n_probes`` temperature samples."""
        imp = np.zeros(self.N, dtype=np.float32)
        for i in range(max(1, n_probes)):
            # Mean-field self-consistent iteration at temperature beta_i
            beta_i = self.beta * (1.0 + 0.1 * self.rng.standard_normal())
            m = np.tanh(beta_i * self.h).astype(np.float32)
            for _ in range(30):
                field = self.h.copy()
                field[1:] += self.J * m[:-1]
                field[:-1] += self.J * m[1:]
                m_new = np.tanh(beta_i * field)
                if np.max(np.abs(m_new - m)) < 1e-6:
                    m = m_new
                    break
                m = m_new
            # Local susceptibility
            chi = beta_i * (1.0 - m ** 2)
            imp += chi.astype(np.float32) ** 2
        imp /= max(1, n_probes)
        return imp


# ---------------------------------------------------------------------------
# 4. Harmonic oscillator chain
# ---------------------------------------------------------------------------


class HarmonicOscillator(Baseline):
    """1D chain of masses connected by springs. Parameters = spring stiffnesses k_i.

    Ground-state energy E_0(k_1,...,k_N) = sum_omega (hbar*omega/2) where omega
    are the normal-mode frequencies of the stiffness matrix. Importance(i) =
    (dE_0 / dk_i)^2, computable via Hellmann-Feynman.
    """

    name = "harmonic_chain"

    def __init__(self, n_params: int, seed: int) -> None:
        super().__init__(n_params, seed)
        self.N = n_params
        # Random stiffnesses around 1.0
        self.k = 1.0 + 0.3 * self.rng.standard_normal(self.N).astype(np.float32)
        self.k = np.clip(self.k, 0.1, None)  # keep positive

    def _stiffness_matrix(self) -> np.ndarray:
        """Build tridiagonal stiffness matrix from spring constants."""
        K = np.zeros((self.N + 1, self.N + 1), dtype=np.float32)
        for i in range(self.N):
            K[i, i] += self.k[i]
            K[i + 1, i + 1] += self.k[i]
            K[i, i + 1] -= self.k[i]
            K[i + 1, i] -= self.k[i]
        return K[1:, 1:]  # fix left endpoint, remove row/col 0

    def parameter_importance(self, n_probes: int) -> np.ndarray:
        """Hellmann-Feynman: dE_0/dk_i = sum_n occupation_n * <v_n | dK/dk_i | v_n> / (2 omega_n).

        Vectorized: dK/dk_i is a rank-2 sparse perturbation whose only nonzero
        entries are at (a,a), (b,b), (a,b), (b,a) where a = i-1, b = i (or
        only (0,0) for i=0). We exploit this to compute all N per-parameter
        gradients at O(N * M) cost where M = number of modes, instead of
        O(N * M * N^2) for the dense einsum.

        For N > 2000 we use ``scipy.linalg.eigh_tridiagonal`` which is
        O(N^2) vs. O(N^3) for ``np.linalg.eigh``. The stiffness matrix is
        exactly tridiagonal by construction, so no approximation is
        introduced; this just exploits the sparsity.
        """
        K = self._stiffness_matrix()
        if self.N > 2000:
            # Tridiagonal: pass the diagonal (d) and sub-diagonal (e).
            # scipy's eigh_tridiagonal returns eigenvalues + eigenvectors.
            try:
                from scipy.linalg import eigh_tridiagonal
                d = np.diag(K).astype(np.float64)
                e = np.diag(K, -1).astype(np.float64)
                eigvals, eigvecs = eigh_tridiagonal(d, e)
                eigvals = eigvals.astype(np.float32)
                eigvecs = eigvecs.astype(np.float32)
            except Exception:
                eigvals, eigvecs = np.linalg.eigh(K)
        else:
            eigvals, eigvecs = np.linalg.eigh(K)
        eigvals = np.clip(eigvals, 1e-8, None)
        omega = np.sqrt(eigvals)  # (M,)
        # eigvecs[i, n] = i-th component of n-th eigenmode
        # For i = 0: dK/dk_0 = e_0 e_0^T, so <v_n | dK | v_n> = v_n[0]^2
        # For i >= 1: dK/dk_i = (e_{i-1} - e_i)(e_{i-1} - e_i)^T, so
        #   <v_n | dK | v_n> = (v_n[i-1] - v_n[i])^2
        grad_omega = np.zeros((self.N, omega.size), dtype=np.float64)  # (N, M)
        v = eigvecs  # (M_dim, M)
        grad_omega[0, :] = v[0, :] ** 2 / (2 * omega)
        for i in range(1, self.N):
            a = i - 1
            b = i
            diff = v[a, :] - v[b, :]
            grad_omega[i, :] = diff * diff / (2 * omega)
        imp = np.zeros(self.N, dtype=np.float32)
        for _ in range(max(1, n_probes)):
            T = 0.05 * (1.0 + 0.2 * self.rng.standard_normal())
            T = float(max(T, 1e-2))
            # exp-overflow-safe Bose-Einstein
            x = omega / T
            large = x > 50.0
            n_bose = np.zeros_like(x)
            n_bose[~large] = 1.0 / (np.exp(x[~large]) - 1.0 + 1e-8)
            # in the large-x limit, n_bose ~ exp(-x) which is effectively 0
            occupation = 0.5 + n_bose  # (M,)
            dE = grad_omega @ occupation  # (N,)
            imp += (dE.astype(np.float32)) ** 2
        imp /= max(1, n_probes)
        return imp


# ---------------------------------------------------------------------------
# 5. Random Boolean circuit
# ---------------------------------------------------------------------------


class BooleanCircuit(Baseline):
    """Random Boolean circuit. Parameters = gate-selection weights w_i (soft).

    Each gate is a mixture of {AND, OR, XOR} with softmax weights w_i;
    importance(i) = average output-flip-rate under random input as a function
    of perturbing w_i. Computed via numerical finite-differences of expected
    output.
    """

    name = "boolean_circuit"

    def __init__(self, n_params: int, seed: int, n_inputs: int = 8) -> None:
        super().__init__(n_params, seed)
        # Each gate has 3 softmax weights, so n_gates = n_params // 3
        self.n_gates = max(4, n_params // 3)
        self.n_params_actual = self.n_gates * 3
        self.n_inputs = n_inputs
        # Small weights keep all softmax entries non-degenerate so that the
        # gradient w.r.t. every logit is nonzero (no dormant branches).
        self.W = 0.3 * self.rng.standard_normal((self.n_gates, 3)).astype(np.float32)
        # Random wiring: each gate takes two previous gate outputs (or input bits)
        self.wire = np.zeros((self.n_gates, 2), dtype=np.int32)
        for g in range(self.n_gates):
            upper = g + n_inputs  # can draw from inputs and prior gates
            self.wire[g, 0] = self.rng.integers(0, upper)
            self.wire[g, 1] = self.rng.integers(0, upper)

    def _forward(self, inputs: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Evaluate circuit on one input; returns output of final gate ∈ [0,1]."""
        state = np.zeros(self.n_inputs + self.n_gates, dtype=np.float32)
        state[: self.n_inputs] = inputs
        for g in range(self.n_gates):
            a = state[self.wire[g, 0]]
            b = state[self.wire[g, 1]]
            logits = W[g]
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            # Mixture of AND/OR/XOR
            andv = a * b
            orv = a + b - a * b
            xorv = a + b - 2 * a * b
            state[self.n_inputs + g] = probs[0] * andv + probs[1] * orv + probs[2] * xorv
        return state[-1:]

    def parameter_importance(self, n_probes: int) -> np.ndarray:
        """Gradient of mean output w.r.t. logits via analytic softmax.

        For each gate g, dOutput / dW[g, k] depends on where gate g is routed
        downstream. We compute this by finite differences with n_probes random
        input vectors, averaged.
        """
        eps = 1e-2
        base_W = self.W.copy()
        imp = np.zeros((self.n_gates, 3), dtype=np.float32)
        for _ in range(max(1, n_probes)):
            x = self.rng.uniform(0, 1, size=self.n_inputs).astype(np.float32)
            base_y = self._forward(x, base_W)
            for g in range(self.n_gates):
                for k in range(3):
                    W_pert = base_W.copy()
                    W_pert[g, k] += eps
                    y = self._forward(x, W_pert)
                    if np.isfinite(y).all():
                        diff = float(y[0] - base_y[0]) / eps
                        imp[g, k] += diff * diff
        imp /= max(1, n_probes)
        return imp.flatten()


# ---------------------------------------------------------------------------
# 6. Cellular automaton (elementary, Rule-110)
# ---------------------------------------------------------------------------


class CellularAutomaton(Baseline):
    """Rule-110 elementary CA. Parameters = initial-cell states s_i ∈ {0,1}.

    Importance(i) = Hamming-divergence rate when cell i is flipped. Under
    Rule-110's glider dynamics, a well-known result is that most cells have
    O(1) light-cone influence but a small fraction of cells launch gliders
    and have linear-in-t influence — a structural hierarchy not unlike a
    three-tier FIM.
    """

    name = "cellular_automaton"

    def __init__(self, n_params: int, seed: int, n_steps: int = 64) -> None:
        super().__init__(n_params, seed)
        self.N = n_params
        self.n_steps = n_steps
        # Random initial state
        self.state = (self.rng.random(self.N) < 0.5).astype(np.int8)

    @staticmethod
    def _rule110_step(state: np.ndarray) -> np.ndarray:
        left = np.roll(state, 1)
        right = np.roll(state, -1)
        pattern = (left << 2) | (state << 1) | right
        return ((110 >> pattern) & 1).astype(np.int8)

    def _evolve(self, initial: np.ndarray) -> np.ndarray:
        s = initial.copy()
        for _ in range(self.n_steps):
            s = self._rule110_step(s)
        return s

    def parameter_importance(self, n_probes: int) -> np.ndarray:
        """Influence = Hamming distance after n_steps, averaged over probes."""
        base = self.state.copy()
        # Probes = random different evolution horizons (different mental
        # "observers" of the same CA). This broadens the importance measure
        # so it is not horizon-specific.
        horizons = sorted({max(8, int(self.n_steps * (0.25 + 0.75 * self.rng.random())))
                            for _ in range(max(1, n_probes))})
        imp = np.zeros(self.N, dtype=np.float32)
        for h in horizons:
            saved_n_steps = self.n_steps
            self.n_steps = h
            base_evolved = self._evolve(base)
            for i in range(self.N):
                perturbed = base.copy()
                perturbed[i] = 1 - perturbed[i]
                evolved = self._evolve(perturbed)
                imp[i] += float(np.sum(evolved != base_evolved))
            self.n_steps = saved_n_steps
        imp /= len(horizons)
        # Square it so the units match other baselines (gradient-squared)
        return imp ** 2


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class BaselineSpec:
    name: str
    cls: type[Baseline]
    # Suggested n_params (kept small so all baselines run in seconds on CPU)
    n_params: int


REGISTRY: list[BaselineSpec] = [
    BaselineSpec("neural_network", NeuralNetwork, 3500),
    BaselineSpec("random_matrix", RandomMatrix, 3003),  # N=77, 77*78/2 = 3003
    BaselineSpec("ising_chain", IsingChain, 256),
    BaselineSpec("harmonic_chain", HarmonicOscillator, 128),
    BaselineSpec("boolean_circuit", BooleanCircuit, 384),  # 128 gates * 3 weights
    BaselineSpec("cellular_automaton", CellularAutomaton, 128),
]


def make(name: str, seed: int) -> Baseline:
    for spec in REGISTRY:
        if spec.name == name:
            return spec.cls(spec.n_params, seed)
    raise KeyError(f"unknown baseline: {name}")
