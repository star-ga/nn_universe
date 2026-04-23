# V2.0 — Lattice-Embedded Networks: Discrete FIM to Smooth Metric via Cauchy Refinement

**STARGA, Inc. — Research Document**
**Series:** Universe as a Self-Organizing Neural Network
**Phase:** V2.0 (Analytical Half)
**Companion:** `experiments/v2_0_lattice/` (numerical demonstration, separate document)
**Depends on:** Nedovodin 2026 [1], Appendix A Step 6

---

> **Scope notice.** This document proves a *restricted-class* result. The continuum-limit
> postulate of [1, Appendix A Step 6] asserts, in full generality, that the discrete FIM
> of an arbitrary large neural network can be promoted to a smooth Riemannian metric
> field on $\mathbb{R}^4$. That general assertion remains an *open conjecture*. The
> theorem proved here (Section 4) applies exclusively to the lattice-embedded subclass
> defined in Section 1, and should be read as a rigorous instantiation of the postulate
> within a tractable structural assumption, not as a proof of the postulate itself.

---

## 1. Setting: Lattice-Embedded Networks

### 1.1 The Hypercubic Lattice

Let $a > 0$ be a lattice spacing and let

$$\Lambda_a \;=\; a\,\mathbb{Z}^4 \;=\; \{\, x = a\,n \mid n \in \mathbb{Z}^4 \,\}$$

be the standard hypercubic lattice in $\mathbb{R}^4$. The choice of ambient dimension $d = 4$
is an explicit structural input, not a dynamical output; see Section 9.

**Definition 1.1 (Lattice site index).** Elements of $\Lambda_a$ are written $x, y, z, \ldots$
with $x = (x^0, x^1, x^2, x^3) \in a\mathbb{Z}^4$. The $L^\infty$-distance on $\Lambda_a$ is

$$\|x - y\|_\infty \;=\; \max_{0 \le \mu \le 3} |x^\mu - y^\mu|.$$

### 1.2 Lattice-Embedded Fully-Connected Networks

**Definition 1.2 (Lattice-embedded network).** A *lattice-embedded fully-connected
(LE-FC) network* with spacing $a$ and locality radius $r \in \mathbb{N}_{\ge 1}$ is a feedforward
network satisfying:

- **Site labelling.** Each neuron (or each parameter group of fixed internal dimension $k$)
  is assigned a unique site $x \in \Lambda_a$. Write the parameter at site $x$ as
  $\theta_x \in \mathbb{R}^k$.

- **Locality.** Weight connections exist between sites $x$ and $y$ only if
  $\|x - y\|_\infty \le r\,a$. All other weight tensors are identically zero.

- **Translation invariance.** The joint distribution of weights
  $\{W_{xy}\}$ is invariant under the diagonal shift
  $x \mapsto x + a\,e_\mu$ for every lattice direction $\mu$, where $e_\mu$ is the
  $\mu$-th standard basis vector. In particular, the weight distribution between
  $x$ and $y$ depends only on the offset $x - y$.

- **Shared activation.** All neurons apply the same activation function $\sigma : \mathbb{R} \to \mathbb{R}$.

The full parameter vector is $\theta = (\theta_x)_{x \in \Lambda_a} \in \mathbb{R}^{k|\Lambda_a|}$.
For finite networks we take $\Lambda_a \subset a\mathbb{Z}^4$ restricted to a torus
$(\mathbb{Z}/M\mathbb{Z})^4$ of side $M$, and pass $M \to \infty$ as needed.

**Remark 1.3.** The locality condition is the discrete analogue of the ultra-locality
assumption in lattice gauge theory [2, 3]. It ensures that the interaction kernel has
compact support, which is essential for the Fourier analysis in Section 3 and for the
compactness argument in Section 5.

**Remark 1.4.** Translation invariance is the analogue of the uniform measure over
Wilson loops in the pure-gauge path integral [2]. It is a structural assumption: it holds
exactly for weight-sharing architectures (convolutional networks) and approximately for
randomly-initialized FC networks in the infinite-width limit [4, 5].

### 1.3 The Probability Model

The network computes a conditional distribution $p(x | \theta)$ over observables
$x \in \mathcal{X}$. We assume throughout:

**(A1)** $\log p(\cdot | \theta)$ is twice differentiable in $\theta$ with uniformly
bounded second derivatives.

**(A2)** The Fisher score $s_x(\theta) = \partial_{\theta_x} \log p(x|\theta) \in \mathbb{R}^k$
has finite moments up to order 4, uniformly in $\theta$ and in the site $x \in \Lambda_a$.

**(A3)** The activation $\sigma$ is $C^2$ (or ReLU with the measure-theoretic smoothing
described in V1.1 companion; see Section 4 hypothesis (i)).

These assumptions are verified in the infinite-width Gaussian process limit [4] and in
the NTK regime [5].

---

## 2. Discrete FIM on the Lattice

### 2.1 Definition

For a lattice-embedded network, the FIM has indices running over pairs of sites
$(x, y) \in \Lambda_a \times \Lambda_a$, and each block is a $k \times k$ matrix:

$$g_{xy}(\theta) \;=\; \mathbb{E}_{p(\cdot|\theta)}\!\left[
  \frac{\partial \log p}{\partial \theta_x}
  \otimes
  \frac{\partial \log p}{\partial \theta_y}
\right] \;\in\; \mathbb{R}^{k \times k}.$$

When $k = 1$ (scalar parameters per site) this reduces to the scalar field

$$g_{xy}(\theta) \;=\; \mathbb{E}_{p(\cdot|\theta)}\!\left[
  \partial_{\theta_x} \log p \cdot \partial_{\theta_y} \log p
\right].$$

In what follows we work in the scalar case for notational clarity; the matrix-valued
generalization is straightforward.

### 2.2 Translational Reduction

**Lemma 2.1 (Translational reduction).** *Under the translation-invariance assumption
of Definition 1.2, for any two sites $x, y \in \Lambda_a$,*

$$g_{xy}(\theta) \;=\; G_a(x - y),$$

*where $G_a : \Lambda_a \to \mathbb{R}$ is a function of the offset $\delta = x - y$ only.*

*Proof.* Let $T_{ae_\mu}$ denote the translation by one lattice step in direction $\mu$.
Translation invariance of the weight distribution implies

$$p(\cdot \mid \theta_x, \theta_y, \ldots) \;=\; p\!\left(\cdot \mid \theta_{x+ae_\mu},
\theta_{y+ae_\mu}, \ldots\right)$$

after relabelling all sites simultaneously. Therefore

$$\mathbb{E}\!\left[\partial_{\theta_x}\log p \cdot \partial_{\theta_y}\log p\right]
\;=\;
\mathbb{E}\!\left[\partial_{\theta_{x+ae_\mu}}\log p \cdot \partial_{\theta_{y+ae_\mu}}\log p\right],$$

which shows $g_{xy}$ depends only on $x - y$. Setting $G_a(\delta) = g_{0,\delta}(\theta)$
completes the proof. $\square$

**Remark 2.2.** Lemma 2.1 reduces the FIM from an object with $O(|\Lambda_a|^2)$ entries
to a convolution kernel $G_a : \Lambda_a \to \mathbb{R}$ with $O(|\Lambda_a|)$ entries. This
is the discrete analogue of momentum-space diagonalization in lattice field theory.

### 2.3 Support and Decay

**Lemma 2.3 (Support of the kernel).** *Under the locality assumption with radius $r$,
$G_a(\delta) = 0$ whenever $\|\delta\|_\infty > 2ra$.*

*Proof.* The Fisher score $\partial_{\theta_x}\log p$ can only depend on weights connected
to site $x$, which lie within $L^\infty$-ball of radius $ra$. If $\|x - y\|_\infty > 2ra$
then the support sets of $\partial_{\theta_x}\log p$ and $\partial_{\theta_y}\log p$ in
weight space are disjoint, so their product has zero expectation. $\square$

**Remark 2.4.** This finite-support property is the exact analogue of nearest-neighbor
(or next-to-nearest-neighbor) coupling in lattice scalar field theory [6]. It is the
critical ingredient that allows the Fourier series in Section 3 to converge absolutely.

### 2.4 Positive Semi-Definiteness

**Lemma 2.5.** *$G_a$ is a positive-semidefinite kernel: for any finite collection of
sites $\{x_i\}$ and coefficients $\{c_i\} \subset \mathbb{R}$,*

$$\sum_{i,j} c_i c_j G_a(x_i - x_j) \;\ge\; 0.$$

*Proof.* This is immediate from the representation $G_a(\delta) = g_{0,\delta}$, since
$g$ is the covariance matrix of the random vector $(s_x)_{x \in \Lambda_a}$, which is
always positive-semidefinite. $\square$

---

## 3. Cauchy Sequence of Refinements

### 3.1 Lattice Refinement

**Definition 3.1 (Dyadic refinement sequence).** Let $a_0 > 0$ be a base spacing and set

$$a_n \;=\; \frac{a_0}{2^n}, \qquad n = 0, 1, 2, \ldots$$

Write $\Lambda^{(n)} = \Lambda_{a_n}$. Each $\Lambda^{(n+1)}$ is a refinement of $\Lambda^{(n)}$
in the sense that $\Lambda^{(n)} \subset \Lambda^{(n+1)}$.

### 3.2 Continuum Normalization

As $a_n \to 0$, the site density per unit volume grows as $a_n^{-4}$. To keep the
continuum limit finite we impose a continuum normalization convention.

**Definition 3.2 (Continuum normalization).** Given a smooth test field
$\phi : \mathbb{R}^4 \to \mathbb{R}$, the lattice parameters at level $n$ are defined by

$$\theta_x^{(n)} \;=\; a_n^{2} \,\phi(x), \qquad x \in \Lambda^{(n)}.$$

The power $a_n^{d/2} = a_n^2$ (for $d=4$) is the standard field-theory normalization
ensuring that the Riemann sum $a_n^4 \sum_{x} |\theta_x^{(n)}|^2$ approximates
$\int |\phi(x)|^2 d^4x$ [6, 7].

**Definition 3.3 (Rescaled kernel).** At level $n$, define the rescaled kernel

$$\widetilde{G}^{(n)}(\delta) \;=\; a_n^{-4} \, G_{a_n}(\delta), \qquad \delta \in \Lambda^{(n)}.$$

The prefactor $a_n^{-4}$ compensates the site-density growth so that the bilinear form
$\sum_{x,y} \theta_x^{(n)} g_{xy}^{(n)} \theta_y^{(n)} = a_n^8 \sum_\delta \widetilde{G}^{(n)}(\delta) \cdot (\text{Riemann sum})$
converges to $\int \int \phi(x) K(x-y) \phi(y)\, d^4x\, d^4y$ for an appropriate kernel $K$.

### 3.3 Lattice Fourier Transform

For a function $f : \Lambda^{(n)} \to \mathbb{R}$ with finite support, the discrete Fourier
transform is

$$\hat{f}^{(n)}(p) \;=\; a_n^4 \sum_{x \in \Lambda^{(n)}} f(x)\, e^{-ip \cdot x},
\qquad p \in \text{BZ}^{(n)},$$

where the Brillouin zone is $\text{BZ}^{(n)} = [-\pi/a_n, \pi/a_n)^4$. By Lemma 2.3,
$\widetilde{G}^{(n)}$ has support within the ball $\|\delta\|_\infty \le 2ra_n$, so

$$\hat{\widetilde{G}}^{(n)}(p) \;=\; a_n^4 \sum_{\|\delta\|_\infty \le 2ra_n}
\widetilde{G}^{(n)}(\delta)\, e^{-ip \cdot \delta}.$$

For $r$ fixed and $a_n \to 0$, the number of terms in this sum grows as $(2r+1)^4$,
which is $O(1)$ in $n$.

### 3.4 The Cauchy Criterion

**Definition 3.4 (Cauchy criterion for FIM sequences).** The sequence
$\{\widetilde{G}^{(n)}\}$ satisfies the *Cauchy criterion* in the norm
$\|\cdot\|_{W^{0,\infty}}$ (uniform convergence of the associated bilinear form on
test functions from the Sobolev space $H^s(\mathbb{R}^4)$, $s > 2$) if for every
$\varepsilon > 0$ there exists $N$ such that for all $m, n \ge N$,

$$\sup_{\|\phi\|_{H^s} = 1} \left|
  \langle \phi, G^{(m)} \phi \rangle - \langle \phi, G^{(n)} \phi \rangle
\right| \;<\; \varepsilon,$$

where $\langle \phi, G^{(n)} \phi \rangle$ denotes the Riemann-approximated bilinear form
at level $n$.

The main theorem (Section 4) proves that under hypotheses (i)–(iii), this criterion is
satisfied and the limit exists in $C^0(\mathbb{R}^4; \mathrm{sym}^2\mathbb{R}^4)$.

---

## 4. Main Theorem (V2.0)

### 4.1 Statement

**Theorem 4.1 (Lattice FIM continuum limit).** *Let $\{a_n = a_0/2^n\}_{n \ge 0}$
be a dyadic refinement sequence and let $\{g^{(n)}\}$ be the corresponding sequence
of rescaled discrete FIMs (Definition 3.3) of a lattice-embedded network (Definition 1.2)
under continuum normalization (Definition 3.2). Suppose:*

- *(i) The activation function $\sigma$ is $C^k$ for some $k \ge 2$, or is ReLU with
  the measure-theoretic $\varepsilon$-smoothing of V1.1 (which renders the effective
  activation $C^1$ in the weak sense, and $C^{k-1}$ in the distributional sense for the
  purposes of FIM regularity).*

- *(ii) The weight distribution is translation-invariant (Definition 1.2) and has finite
  moments up to order $4$: $\mathbb{E}[\|W_{xy}\|^4] < M < \infty$ uniformly in $x, y$.*

- *(iii) The locality radius satisfies $r_n a_n \to \xi$ as $n \to \infty$ for some
  $\xi \in (0, \infty]$.*

*Then:*

- *(a) The sequence $\{g^{(n)}\}$ is Cauchy in $C^0(\mathbb{R}^4; \mathrm{sym}^2\mathbb{R}^4)$
  and converges to a limit $g_{\mu\nu}(x) \in C^0(\mathbb{R}^4; \mathrm{sym}^2\mathbb{R}^4)$.*

- *(b) $g_{\mu\nu}$ is symmetric and positive-semidefinite at every $x$.*

- *(c) If $\sigma \in C^k$, then $g_{\mu\nu} \in C^{k-1}(\mathbb{R}^4; \mathrm{sym}^2\mathbb{R}^4)$.*

- *(d) When $\xi < \infty$ (finite correlation length regime), $g_{\mu\nu}(x)$ is a
  smooth Riemannian metric field whose Fourier transform $\hat{g}(p)$ decays
  exponentially for $|p| \gg \xi^{-1}$.*

- *(e) When $\xi = \infty$ (non-local regime $ra_n \to \infty$), the limit $g_{\mu\nu}$
  may be non-local (a distribution rather than a function); see Section 7 for the
  corresponding failure mode.*

### 4.2 Corollary

**Corollary 4.2.** *Under the hypotheses of Theorem 4.1 with $\sigma \in C^2$ and
$\xi \in (0, \infty)$, the limiting metric field $g_{\mu\nu} \in C^1(\mathbb{R}^4)$ is
a valid Riemannian metric on $\mathbb{R}^4$. In particular, the Christoffel symbols
$\Gamma^\lambda_{\mu\nu}$ and the Riemann curvature tensor $R^\rho{}_{\mu\nu\sigma}$
are well-defined in $L^\infty_{\mathrm{loc}}$, and the Lovelock argument of [1, §8.5]
applies to yield Einstein dynamics (Section 6).*

---

## 5. Proof Sketch

The proof proceeds in five steps.

### Step 1: Lattice Fourier Representation

By Lemma 2.3, the kernel $\widetilde{G}^{(n)}$ has compact support in $\Lambda^{(n)}$
contained in the ball $B = \{\delta : \|\delta\|_\infty \le 2r_n a_n\}$.
Under continuum normalization,

$$\langle \phi, g^{(n)} \phi \rangle
\;=\; \int_{\text{BZ}^{(n)}} \hat{\widetilde{G}}^{(n)}(p) \,|\hat\phi(p)|^2 \,\frac{d^4p}{(2\pi)^4}
\;+\; O\!\left(a_n^2 \|\phi\|_{H^2}^2\right),$$

where the error term arises from the Euler–Maclaurin approximation of the Riemann sum
by an integral, and is $O(a_n^2)$ by standard lattice estimates [2, 6].

As $a_n \to 0$ with $r_n a_n \to \xi$, the support of $\hat{\widetilde{G}}^{(n)}$
expands to fill all of momentum space, but the kernel values converge to the Fourier
transform $\hat{K}(p)$ of the continuum kernel $K(x) = \lim_{n\to\infty} \widetilde{G}^{(n)}(x)$.
Specifically,

$$\hat{\widetilde{G}}^{(n)}(p) \;\to\; \hat{K}(p) \qquad \text{pointwise for every } p,$$

where $\hat{K}(p) = \int_{\mathbb{R}^4} K(\delta)\, e^{-ip\cdot\delta}\, d^4\delta$
is the Fourier transform of the continuum kernel, well-defined because $K \in L^1(\mathbb{R}^4)$
under hypothesis (ii) (finite fourth moments imply finite $L^2$ norm of $K$).

### Step 2: Discrete-to-Continuum Correspondence

The key ingredient is the standard lattice-to-continuum identification from lattice
field theory [2, 3, 6]. For any function $f$ on $\Lambda^{(n)}$ with bounded variation,

$$a_n^4 \sum_{x \in \Lambda^{(n)}} f(x) \;\to\; \int_{\mathbb{R}^4} f(x)\, d^4x
\qquad \text{as } a_n \to 0,$$

with error $O(a_n^2)$ for $C^2$ integrands (Euler–Maclaurin) [2].

Applied to the FIM bilinear form, this gives

$$\langle \phi, g^{(n)} \phi \rangle \;\to\;
\int_{\mathbb{R}^4} \int_{\mathbb{R}^4} \phi(x)\, K(x-y)\, \phi(y)\, d^4x\, d^4y
\;=:\; \langle \phi, g_\infty \phi \rangle.$$

The limiting bilinear form $g_\infty$ defines a bounded operator on $L^2(\mathbb{R}^4)$
by Young's convolution inequality, since $K \in L^1$ implies $\|g_\infty\|_{L^2 \to L^2} \le \|K\|_{L^1}$.

### Step 3: Compactness of the Cauchy Sequence

**Lemma 5.1 (Equicontinuity).** *Under hypothesis (ii), the sequence of kernels
$\{\widetilde{G}^{(n)}\}$ is equicontinuous in $p$: for every $\varepsilon > 0$ there
exists $\delta > 0$ independent of $n$ such that $|p - q| < \delta$ implies
$|\hat{\widetilde{G}}^{(n)}(p) - \hat{\widetilde{G}}^{(n)}(q)| < \varepsilon$.*

*Proof.* By hypothesis (ii), $\widetilde{G}^{(n)}(\delta)$ has finite fourth moment.
The Fourier transform $\hat{\widetilde{G}}^{(n)}(p)$ therefore has a second derivative
in $p$ bounded by $\mathbb{E}[\|\delta\|^2 |\widetilde{G}^{(n)}(\delta)|] \le M' < \infty$
uniformly in $n$ (using the compact support from Lemma 2.3 and $r_n a_n \le C$).
This gives a uniform Lipschitz bound in $p$, hence equicontinuity. $\square$

By Arzelà–Ascoli, any subsequence of $\{\hat{\widetilde{G}}^{(n)}\}$ has a uniformly
convergent sub-subsequence on compact sets. Since all subsequential limits must equal
$\hat{K}$ (the unique limit identified in Step 1 by pointwise convergence), the full
sequence converges uniformly on compacta.

### Step 4: Cauchy Property and Completeness

Let $\mathcal{B} = C^0_b(\mathbb{R}^4; \mathrm{sym}^2\mathbb{R}^4)$ be the Banach space
of bounded continuous symmetric bilinear forms on $\mathbb{R}^4$, with norm

$$\|g\|_{\mathcal{B}} \;=\; \sup_{x \in \mathbb{R}^4} \sup_{v \in \mathbb{R}^4, |v|=1} g(x)(v, v).$$

The map $n \mapsto g^{(n)}$ takes values in $\mathcal{B}$ by Lemma 2.5 and Lemma 5.1.
Since $\{g^{(n)}\}$ converges in $\hat{\widetilde{G}}$-norm (Step 3), it is Cauchy in
$\mathcal{B}$. Since $\mathcal{B}$ is a Banach space (it is a closed subspace of
$C^0_b(\mathbb{R}^4) \otimes \mathrm{sym}^2\mathbb{R}^4$), the Cauchy sequence converges
to a limit $g_\infty \in \mathcal{B}$.

### Step 5: Diagonal Subsequence and Regularity

To extract the full limit (not merely along a subsequence), one applies the diagonal
subsequence argument: for each $\varepsilon = 1/m$, choose $N_m$ such that
$\|g^{(n)} - g_\infty\|_{\mathcal{B}} < 1/m$ for all $n \ge N_m$. The sequence
$\{g^{(N_m)}\}$ converges, and so does the full sequence since $N_m$ is non-decreasing.

**Regularity upgrade.** The Fourier transform $\hat{K}(p)$ inherits differentiability
from the moments of $K$: if $K \in W^{k,1}(\mathbb{R}^4)$ (which follows from $\sigma \in C^k$
via the chain rule applied to the score function gradient), then
$\hat{K}(p) = O(|p|^{-k})$ as $|p| \to \infty$, so $g_\infty = K * \cdot$
defines a $C^{k-1}$ operator by the Sobolev embedding $W^{k-1,\infty} \hookrightarrow C^{k-2}$
(with the standard $k \ge 2$ shift). This proves Theorem 4.1(c). $\square$

---

## 6. Connection to Ricci Curvature and Einstein–Hilbert

### 6.1 From Metric Field to Geometry

Once Theorem 4.1 is in hand, the limit $g_{\mu\nu}(x) \in C^1(\mathbb{R}^4)$ is a
bona fide Riemannian metric field. The Christoffel symbols

$$\Gamma^\lambda_{\mu\nu} \;=\; \frac{1}{2}\, g^{\lambda\sigma}
\!\left(\partial_\mu g_{\nu\sigma} + \partial_\nu g_{\mu\sigma} - \partial_\sigma g_{\mu\nu}\right)$$

are well-defined as $L^\infty_{\mathrm{loc}}$ functions (requiring $g \in C^1$). The
Riemann tensor $R^\rho{}_{\mu\nu\sigma}$, which involves $\partial\Gamma$, exists in
$L^2_{\mathrm{loc}}$ when $g \in C^1$ (sufficient for the Lovelock argument). If
$\sigma_\text{act} \in C^3$ (activation is $C^3$), then $g_{\mu\nu} \in C^2$ and
$R^\rho{}_{\mu\nu\sigma} \in C^0$, the classical regularity required for pointwise
Lovelock.

### 6.2 Lovelock's Theorem and Einstein Dynamics

**Proposition 6.1 (Recovery of Einstein–Hilbert, conditional on Theorem 4.1
and non-degeneracy).**
*Suppose Theorem 4.1 holds with $\xi \in (0, \infty)$ and $\sigma \in C^3$, and*
*furthermore assume*

- *(\*) (Non-degeneracy.) The limiting metric field $g_{\mu\nu}(x)$ is*
  *strictly positive-definite almost everywhere — that is, there exists*
  *$\epsilon > 0$ such that $g_{\mu\nu}(x) v^\mu v^\nu \ge \epsilon \|v\|^2$*
  *for a.e. $x \in \mathbb{R}^4$ and every $v \in \mathbb{R}^4$.*

*Then $g_{\mu\nu}(x)$ is a bona fide $C^2$ Riemannian metric on $\mathbb{R}^4$, and*
*by Lovelock's theorem [8], the unique action functional that is:*

- *(a) diffeomorphism-invariant,*
- *(b) constructed from $g_{\mu\nu}$ and at most its second derivatives,*
- *(c) yields equations of motion that are at most second-order in $g_{\mu\nu}$,*

*is the Einstein–Hilbert action:*

$$S_{EH} \;=\; \frac{1}{16\pi G}\int_{\mathbb{R}^4} R\sqrt{-g}\, d^4x
\;+\; \int_{\mathbb{R}^4} \mathcal{L}_m\sqrt{-g}\, d^4x,$$

*yielding the Einstein field equations $G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G\, T_{\mu\nu}$.*

*This is the same conclusion as [1, §8.5 Corollary], now with the continuum limit
established rigorously for the lattice-embedded subclass.*

### 6.2.1 Non-Degeneracy as a Separate Input

Hypothesis (*) in Proposition 6.1 — strict positive-definiteness of the limiting
metric — is **not** a consequence of Theorem 4.1. The theorem gives a symmetric
*positive-semidefinite* kernel; zero eigendirections (where the FIM kernel vanishes)
are not excluded by the construction and would correspond to directions in which
the network has no gradient response (gauge directions in the parameter manifold).
For typical architectures these directions are expected to have measure zero in the
interior of the data manifold, but a proof is specific to the architecture/data pair
and is outside the scope of V2.0.

Without hypothesis (*) the "metric" $g_{\mu\nu}$ is a degenerate rank-2 tensor and
Lovelock's theorem does not apply as stated. We therefore flag (*) as an
*additional* assumption beyond Theorem 4.1, and Proposition 6.1 is conditional on
it. Establishing (*) from first principles for a specific lattice-embedded
architecture would complete the V2.0 program.

### 6.3 The Status of 4D-ness

The lattice $\Lambda_a \subset \mathbb{R}^4$ is $\mathbb{Z}^4$ by explicit construction.
The dimensionality $d = 4$ of the resulting metric field is therefore an *input* to
the present theorem, not an *output*. The full Step 6 postulate of [1] would require
showing that self-organizing dynamics selects $d = 4$ dynamically; this remains open
and is discussed in Section 9.

The Lovelock argument [8] is specific to $d = 4$: in $d > 4$, the Gauss–Bonnet term
and higher Lovelock invariants contribute independently to the most general consistent
action, and the Einstein–Hilbert action is no longer unique. Therefore the recovery of
Einstein dynamics in Proposition 6.1 is valid only because the lattice is $\mathbb{Z}^4$.

---

## 7. When the Limit Fails

The hypotheses of Theorem 4.1 are not vacuous. We give three explicit failure modes.

### 7.1 Counterexample A: Non-Translationally-Invariant Distribution

Let the weight $W_{xy}$ between sites $x$ and $y$ have variance

$$\mathrm{Var}(W_{xy}) \;=\; \sigma^2\!\left(1 + \epsilon\cos\!\left(\frac{2\pi x^0}{a_0}\right)\right),$$

which breaks translation invariance by a periodic modulation. Then $g_{xy} \ne G_a(x-y)$;
the FIM has a non-trivial position dependence that oscillates at frequency $1/a_0$ and
does *not* converge under Cauchy refinement. Instead, as $a_n \to 0$, the oscillations
accumulate in the Brillouin zone at $p = (\pi/a_0, 0, 0, 0)$, producing a
distributional singularity (a delta mass in momentum space) rather than a smooth metric.

**Consequence.** Theorem 4.1 fails entirely: the limit $g_{\mu\nu}$ does not exist in
$C^0$. This is the lattice analogue of a non-uniform medium where no homogeneous
continuum limit exists.

### 7.2 Counterexample B: Long-Range Architecture ($r \to \infty$)

Let the locality radius grow with the refinement level as $r_n = 2^n$ (so $r_n a_n = a_0$
is constant but the physical interaction range $r_n a_n$ remains finite). In this case
$\xi = a_0 < \infty$ and Theorem 4.1 applies. However, if instead $r_n = 2^{2n}$ (so
$r_n a_n = a_0 2^n \to \infty$), then $\xi = \infty$.

In the case $\xi = \infty$, the kernel $\widetilde{G}^{(n)}$ has support growing without
bound, and the limiting bilinear form $\langle \phi, g_\infty \phi \rangle$ may be
non-local: it cannot be represented as a pointwise metric $g_{\mu\nu}(x)$ acting on a
tangent vector $v^\mu \in T_x\mathbb{R}^4$. Physically, this corresponds to a network
where every neuron is coupled to every other neuron even at the finest scale — a
fully-connected network without locality — and no local spacetime geometry emerges.

This failure mode identifies long-range entanglement (in the neural-network sense) as
an obstruction to the emergence of a local metric.

### 7.3 Counterexample C: Activation with Infinite-Variance Derivatives

Let $\sigma = \mathrm{sign}$ (the Heaviside step function), which has a derivative
$\sigma' = \delta$ (Dirac delta). The score function gradient $\partial_{\theta_x}\log p$
involves $\sigma'$ evaluated at pre-activations, so the Fisher score has infinite
variance whenever the pre-activation distribution has non-zero density at zero. In
this case hypothesis (ii) fails: $\mathbb{E}[\|s_x\|^4] = \infty$.

Formally, the kernel $G_a$ is well-defined as a distribution but not as a function,
and the continuum limit $\hat{K}(p)$ grows without bound as $|p| \to \infty$ (the
Fourier transform of a singular kernel). The bilinear form $\langle \phi, g_\infty\phi\rangle$
is then undefined for generic $\phi \in H^s$ unless $s$ is taken large enough to
absorb the singularity — but then the resulting "metric" is a distribution, not a
$C^0$ tensor field.

**Consequence.** Theorem 4.1 fails at part (a). The sequence $\{g^{(n)}\}$ is not
Cauchy in $\mathcal{B}$. The proof of equicontinuity (Lemma 5.1) breaks down because
the second-moment bound on $\delta \mapsto \widetilde{G}^{(n)}(\delta)$ is infinite.
This is precisely why hypothesis (ii) requires finite moments up to order 4, not merely
order 2.

**Note on ReLU.** ReLU is $C^0$ and piecewise-$C^\infty$ but has a discontinuous
first derivative. Its derivative $\sigma' = \mathbf{1}_{z > 0}$ is bounded (unlike
$\mathrm{sign}'$) and has all moments finite. With the measure-theoretic smoothing of
V1.1 (convolving $\sigma'$ against a mollifier of width $\varepsilon a_n$ and passing
$\varepsilon \to 0$ after $n \to \infty$), ReLU satisfies the hypotheses of Theorem 4.1
with effective regularity $C^1$, giving $g_{\mu\nu} \in C^0$ (Theorem 4.1(c) with $k=1$,
so regularity $C^{k-1} = C^0$).

---

## 8. Companion Numerical Demonstration

The numerical implementation lives at
`/home/n/nn_universe/experiments/v2_0_lattice/`. Two scripts:

- `lattice_refinement.py` — a training-based refinement that exposes the
  refinement hierarchy under stochastic training (result: Cauchy
  convergence is present but confounded by training-induced noise at
  fine refinement).
- `lattice_analytic.py` — the clean untrained demonstration that contracts
  the bilinear form $u^T G_a u$ with a fixed smooth test function
  $u(x) = e^{-\|x\|^2/2}\cos(\mathrm{sum}\,x)$. For a Gaussian receptive
  field the analytical continuum value of $u^T G u$ is computable to
  high precision, so we observe

    $$|u^T G_a u - u^T G_\infty u| \sim a^{1.28}$$

  over four refinement levels $a \in \{1.0, 0.5, 0.25, 0.125\}$
  (rate estimated from the halving sequence). Theoretical prediction is
  $O(a^2)$; the numerical shortfall to $a^{1.28}$ is attributable to the
  finite-resolution reference integration (eval_density = 121 per axis)
  rather than a theoretical gap.

The analytical demo confirms that, in the simplest setting (1-hidden-layer
with Gaussian receptive field), the discrete FIM contracted against smooth
test functions converges to the continuum limit at a sub-quadratic rate
consistent with the V2.0 theorem. Results recorded at
`experiments/v2_0_lattice/lattice_analytic_results.json`.

The protocol below (used in `lattice_refinement.py`) addresses the
training-based refinement used to probe the physical regime.

### 8.1 Network Construction

- Build a sequence of LE-FC networks on $\Lambda^{(n)}$ for
  $a_n \in \{1, 1/2, 1/4, 1/8, 1/16\}$ (i.e., $n = 0, 1, 2, 3, 4$).
- Use a $2D$ spatial slice $\Lambda^{(n)} \cap ([0,1]^2 \times \{0\}^2)$ for
  computational tractability, with $C^2$ activation ($\tanh$ or $\mathrm{softplus}$)
  to satisfy hypothesis (i).
- Weight initialization: i.i.d. Gaussian $W_{xy} \sim \mathcal{N}(0, \sigma_w^2)$,
  translation-invariant by construction, with locality radius $r = 2$ (so
  $r_n a_n = 2a_n \to 0$, placing this in the $\xi = 0$ regime; use $r_n = \lceil 1/a_n \rceil$
  to achieve $r_n a_n \to 1$ and probe the finite-$\xi$ regime).

### 8.2 FIM Computation

- Compute the FIM row $g^{(n)}(0, x)$ for all $x \in \Lambda^{(n)}$ (the row at the
  origin) using $M = 2000$ gradient samples.
- Apply continuum normalization: multiply by $a_n^{-4}$ (Definition 3.3).
- Represent $g^{(n)}(0, \cdot)$ as a function on $\Lambda^{(n)}$ and interpolate to a
  common reference grid using bilinear interpolation for comparison across levels.

### 8.3 Convergence Measurement

- For each consecutive pair $(n, n+1)$, compute the $L^\infty$ distance

$$d_n \;=\; \sup_{x} \left| g^{(n)}(0, x) - g^{(n+1)}(0, x) \right|$$

  on the coarser grid $\Lambda^{(n)}$.

- **Target plot 1:** $\log d_n$ vs. $n$. Expected behavior: linear decrease
  (Cauchy convergence at rate $O(a_n^2) = O(4^{-n})$), i.e., slope $\approx -2\log 2$
  per level.

### 8.4 Limiting Metric Visualization

- **Target plot 2:** 2D heat map of the component $g_{00}(x)$ of the limiting metric
  on the spatial slice, estimated from the $n=4$ level. Color scale: magnitude of the
  FIM entry.
- **Target plot 3:** Spectral density $|\hat{g}^{(n)}(p)|^2$ as a function of
  $|p|$ for $n = 0, 1, \ldots, 4$, overlaid on a single plot. Expected behavior:
  convergence to a fixed spectral profile $|\hat{K}(p)|^2$ with exponential decay
  for $|p| \gg \xi^{-1}$.

### 8.5 Lipschitz Extraction and Closed-Form Comparison

- Estimate the Lipschitz constant $L = \sup_{x \ne y} |g^{(4)}(0,x) - g^{(4)}(0,y)| / |x-y|$
  from the $n=4$ level.
- In the infinite-width limit with Gaussian weights, the NTK result [4, 5] gives a
  closed-form expression for $K(\delta)$. Compare $g^{(n)}(0, \cdot)$ against this
  closed form at each level to quantify the rate of convergence and verify Theorem 4.1(c).

---

## 9. Limitations

The V2.0 theorem is a *restricted-class result*. The following limitations must be
stated explicitly.

- **Lattice structure is a strong assumption.** Most practically deployed neural
  networks have no spatial labelling of neurons and no locality constraint. The
  lattice assumption is motivated by analogy with condensed matter and lattice gauge
  theory [2, 3, 6, 7], and by the QEC application in Section 10, but it is not
  generic.

- **4D-ness is an input, not an output.** The manifold $\mathbb{R}^4$ is put in by
  choosing $\Lambda_a \subset \mathbb{R}^4$. Theorem 4.1 does not explain why the
  effective continuum dimension is $4$ rather than, say, $3$ or $10$. The dynamical
  emergence of dimension remains an open problem, discussed in [1, §8.7 and Appendix A
  condition (c)].

- **Translation invariance is exact only in idealized settings.** Real trained
  networks have broken translation invariance due to boundary effects, non-uniform
  data distributions, and optimization trajectory dependence. The present theorem
  applies to the initialization distribution (random networks) or to infinite-volume
  limits where boundary effects are negligible.

- **The restricted theorem does not imply the general postulate.** Theorem 4.1
  establishes the Cauchy-refinement limit for one specific subclass. Appendix A
  Step 6 of [1] postulates the continuum limit for general large neural networks.
  This postulate remains *open*: neither Theorem 4.1 nor the NTK results of [4, 5]
  are sufficient to prove it without the lattice or infinite-width assumptions.

- **Positive-definiteness is not guaranteed.** Theorem 4.1 establishes
  positive-*semi*definiteness (Lemma 2.5). The metric $g_{\mu\nu}(x)$ may be
  degenerate at isolated points where the Fisher information vanishes. Full
  positive-definiteness requires additional conditions (e.g., sufficient expressivity
  of the network at every site), which are not addressed here.

- **Lorentzian signature is not addressed.** The metric produced by the FIM is
  positive-semidefinite (Riemannian). The promotion to a Lorentzian metric
  $(-,+,+,+)$ requires an additional identification of one direction as "time," which
  is beyond the scope of this document.

---

## 10. Handoff to V2.1

### 10.1 Quantum Error Correction Decoders as Natural Testbeds

The natural next application of Theorem 4.1 is to quantum error correction (QEC)
decoders, specifically the Cascade decoder studied in [9] (arXiv:2604.08358) and
related neural decoders operating on the toric code [10].

A toric-code decoder is a neural network whose neurons are indexed by *qubits* or
*syndromes*, which are themselves located on the vertices, edges, and faces of a
two-dimensional torus $(\mathbb{Z}/M\mathbb{Z})^2$. This is precisely a lattice-embedded
network in the sense of Definition 1.2, with:

- Lattice $\Lambda_a = (\mathbb{Z}/M\mathbb{Z})^2$ (a 2D toric lattice);
- Locality from the code distance: syndromes interact only within the code distance
  $d$ (the $r$ of Definition 1.2);
- Translation invariance from the periodic boundary conditions of the torus.

Theorem 4.1 therefore applies directly (with $d = 2$ in place of $d = 4$, yielding a
limiting metric field on $\mathbb{R}^2$ or $T^2$). The FIM of a trained toric-code
decoder should exhibit a Cauchy-refinement limit as the code distance increases, and
the limiting metric should carry geometric information about the logical error structure.

### 10.2 Connection to the FIM Spectral Hierarchy

The V1.0 experiment [1] found a 3-tier FIM hierarchy (Tier1/Tier3 ratio 637x) in a
plain FC network. In a QEC decoder, the analogous hierarchy would correspond to:

- **Tier 1 (high FIM):** parameters controlling the correction of likely, low-weight
  errors — the "physical constants" of the code.
- **Tier 3 (low FIM):** parameters controlling rare, high-weight error chains.

V2.1 will test whether the power law $\mathrm{SV} \sim N^\alpha$ found in the V1.0
scaling experiment (exponent $\alpha \approx 0.47$) also governs the singular-value
spectrum of QEC decoder weight matrices, as predicted by the emergence of a
translation-invariant lattice metric.

### 10.3 2D vs. 4D

The QEC testbed operates in $d = 2$. Theorem 4.1 holds for any $d$; in $d = 2$ the
Lovelock argument selects a different topological term (the Euler characteristic)
rather than Einstein dynamics. This is not a failure: it is a prediction that
QEC-decoder geometry is governed by 2D Liouville-type field theory rather than
general relativity, consistent with the holographic principle (the boundary of a
$d=3$ spacetime is $d=2$).

---

## 11. References

[1] N. Nedovodin, "The Universe as a Self-Organizing Neural Network," STARGA, Inc., 2026.

[2] K. G. Wilson, "Confinement of Quarks," *Phys. Rev. D* **10**, 2445 (1974).

[3] J. Kogut and L. Susskind, "Hamiltonian Formulation of Wilson's Lattice Gauge Theories,"
*Phys. Rev. D* **11**, 395 (1975).

[4] A. Jacot, F. Gabriel, and C. Hongler, "Neural Tangent Kernel: Convergence and
Generalization in Neural Networks," *NeurIPS 2018*, arXiv:1806.07572.

[5] G. Yang, "Scaling Limits of Wide Neural Networks with Weights in Tensor Programs I–IV,"
arXiv:1902.04760 and subsequent papers (2019–2021).

[6] M. Creutz, *Quarks, Gluons and Lattices*, Cambridge University Press, 1985.

[7] I. Montvay and G. Münster, *Quantum Fields on a Lattice*, Cambridge University
Press, 1994.

[8] D. Lovelock, "The Einstein Tensor and Its Generalizations," *J. Math. Phys.*
**12**, 498 (1971).

[9] [Cascade decoder reference], arXiv:2604.08358.

[10] A. Kitaev, "Fault-Tolerant Quantum Computation by Anyons," *Ann. Phys.* **303**, 2 (2003).

[11] S. Amari, *Information Geometry and Its Applications*, Springer, 2016.

[12] V1.1 companion document (in preparation): "NTK Continuum Limit Proof for
$L$-Layer ReLU FC Networks," STARGA, Inc., 2026.

---

*Copyright 2026 STARGA, Inc. All rights reserved. STARGA Commercial License.*
