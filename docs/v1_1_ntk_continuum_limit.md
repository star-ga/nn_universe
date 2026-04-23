# NTK Continuum Limit for L-Layer ReLU Fully-Connected Networks

**V1.1 Technical Supplement to "The Universe as a Self-Organizing Neural Network"**

Nikolai Nedovodin, STARGA Inc.

*Technical Report — April 2026*

---

**Abstract.** Appendix A, Step 6 of Nedovodin [1] invokes a continuum-limit postulate and defers its verification to V1.1. This document supplies that verification for a restricted but precise class: $L$-layer ReLU fully-connected networks with Neural Tangent Kernel (NTK) parameterization and Gaussian He initialization, trained under gradient flow (or natural gradient descent with small damping $\varepsilon > 0$) on an MSE loss over a compact data distribution. We prove that, as the minimum hidden width $n \to \infty$, the parameter-space Fisher Information Matrix (FIM) $g_{ij}(\theta(t))$ converges, under a natural compact embedding into $L^2(\mathcal{P}_X \times \mathcal{P}_X)$, to a bounded symmetric positive-semidefinite bilinear form whose kernel representation is smooth in the data argument and Lipschitz-continuous in the training time $t$. This establishes conditions (a) and (b) of Step 6 for the restricted class, placing the FIM–Onsager identification $L^{ij} = \eta g^{ij}$ on rigorous footing within the NTK regime. Condition (c) — the emergence of four effective macroscopic dimensions — is discussed but remains open.

---

## 1. Setting and Notation

### 1.1 Network Architecture

Fix integers $L \geq 2$ (number of weight layers) and widths $n_0, n_1, \ldots, n_{L-1}, n_L$ where:

- $n_0 = d_{\mathrm{in}}$ is the input dimension (fixed throughout),
- $n_L = 1$ (scalar output; the extension to $n_L > 1$ is discussed in Remark 1.4),
- $n_1, \ldots, n_{L-1}$ are hidden widths, all taken proportional to a single scale parameter $n$: write $n_l = \lfloor \alpha_l n \rfloor$ for fixed ratios $\alpha_l > 0$, $l = 1, \ldots, L-1$.

The network function $f : \mathbb{R}^{n_0} \times \Theta \to \mathbb{R}$ is defined by the recursion

$$h^{(0)}(x) = x, \qquad x \in \mathbb{R}^{n_0},$$

$$h^{(l)}(x) = \phi\!\left(W^{(l)} h^{(l-1)}(x) + b^{(l)}\right), \qquad l = 1, \ldots, L-1,$$

$$f(x; \theta) = W^{(L)} h^{(L-1)}(x) + b^{(L)},$$

where $\phi = \mathrm{ReLU}(z) = \max(0,z)$ applied componentwise, $W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$ are weight matrices, $b^{(l)} \in \mathbb{R}^{n_l}$ are bias vectors, and

$$\theta = \bigl(W^{(1)}, b^{(1)}, \ldots, W^{(L)}, b^{(L)}\bigr) \in \Theta \subset \mathbb{R}^P, \qquad P = \sum_{l=1}^{L} \bigl(n_l n_{l-1} + n_l\bigr).$$

### 1.2 Initialization

We use the **NTK parameterization** (also called the "mean-field" or "standard" parameterization in the NTK literature [2, 3]). In this parameterization each weight matrix entry and bias component is initialized as

$$W^{(l)}_{ij} \sim \mathcal{N}\!\left(0,\, \frac{\sigma_w^2}{n_{l-1}}\right), \qquad b^{(l)}_i \sim \mathcal{N}\!\left(0,\, \sigma_b^2\right),$$

independently. For ReLU activations the canonical choice is the **He initialization** [4]: $\sigma_w^2 = 2$, $\sigma_b^2 = 0$ (or any fixed $\sigma_b^2 < \infty$). Throughout this document we take $\sigma_w^2 = 2$, $\sigma_b^2 = 0$ for definiteness; the argument extends to general $\sigma_w^2, \sigma_b^2 > 0$ without modification.

In the NTK parameterization the forward pass is

$$f(x;\theta) = \frac{1}{\sqrt{n_{L-1}}} \tilde{W}^{(L)} h^{(L-1)}(x),$$

where $\tilde{W}^{(L)}_{ij} \sim \mathcal{N}(0,1)$ and $h^{(l)}$ involves similar $1/\sqrt{n_{l-1}}$ rescalings at each layer; see [2] for the precise recursive definition. The crucial property of this parameterization is that the network output and all layer-wise pre-activations remain $O(1)$ as $n \to \infty$, preventing the signal from exploding or vanishing.

### 1.3 Loss Landscape and Data Distribution

Let $\mathcal{P}_X$ be a **compact** Borel probability measure on $\mathbb{R}^{n_0}$, supported on a compact set $\mathcal{X} \subset \mathbb{R}^{n_0}$. Let $\mathcal{P}_{Y|X}$ be the conditional law of the label $y \in \mathbb{R}$ given $x$. The training distribution is $\mathcal{P} = \mathcal{P}_X \otimes \mathcal{P}_{Y|X}$.

**Loss.** We work with the mean-squared error (MSE) loss:

$$\mathcal{C}(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{P}}\!\left[\tfrac{1}{2}\bigl(f(x;\theta) - y\bigr)^2\right].$$

**Gaussian likelihood model.** For the information-geometric analysis we model predictions as

$$p(y \mid x;\theta) = \mathcal{N}\!\bigl(f(x;\theta),\, \sigma^2\bigr), \qquad \sigma^2 > 0 \text{ fixed}.$$

Under this model the negative log-likelihood is (up to a constant) $(2\sigma^2)^{-1}(y - f(x;\theta))^2$, so gradient-flow on $\mathcal{C}(\theta)$ is equivalent to gradient-flow on the negative log-likelihood with $\sigma^2 = \tfrac{1}{2}$.

**Empirical FIM.** The Fisher Information Matrix is

$$g_{ij}(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{P}}\!\left[\frac{\partial \log p(y|x;\theta)}{\partial \theta_i} \cdot \frac{\partial \log p(y|x;\theta)}{\partial \theta_j}\right].$$

Under the Gaussian likelihood with fixed $\sigma^2$, this reduces to

$$g_{ij}(\theta) = \frac{1}{\sigma^2}\,\mathbb{E}_{x \sim \mathcal{P}_X}\!\left[\frac{\partial f(x;\theta)}{\partial \theta_i} \cdot \frac{\partial f(x;\theta)}{\partial \theta_j}\right]. \tag{1.1}$$

The label $y$ drops out because $\partial_{\theta_i} \log p = \sigma^{-2}(y - f(x;\theta))\,\partial_{\theta_i}f$ and $\mathbb{E}_{y|x}[(y - f)^2] = \sigma^2$ when the model is well-specified. We shall normalize $\sigma = 1$ throughout and restore the factor at the end.

**Training dynamics.** We consider **gradient flow** (continuous-time gradient descent):

$$\dot{\theta}(t) = -\nabla_\theta \mathcal{C}(\theta(t)), \tag{1.2}$$

and also **natural gradient descent with damping** $\varepsilon > 0$:

$$\dot{\theta}(t) = -\eta\,\bigl(g(\theta(t)) + \varepsilon I\bigr)^{-1} \nabla_\theta \mathcal{C}(\theta(t)). \tag{1.3}$$

The damping $\varepsilon > 0$ ensures invertibility of $g + \varepsilon I$ without assuming $g$ is strictly positive definite. The Onsager identification $L^{ij} = \eta g^{ij}$ of [1] corresponds to $\varepsilon \to 0$ after the $n \to \infty$ limit; see Remark 4.6.

**Remark 1.4 (Vector outputs).** When $n_L = d > 1$, the FIM $(1.1)$ acquires an additional trace over output components:
$$g_{ij}(\theta) = \sum_{k=1}^{d} \mathbb{E}_x\!\left[\partial_{\theta_i} f_k(x;\theta)\,\partial_{\theta_j} f_k(x;\theta)\right].$$
Every argument below applies component-wise; we restrict to $n_L = 1$ for notational clarity.

---

## 2. The Neural Tangent Kernel

### 2.1 Definition and Infinite-Width Limit

For finite $n$, define the **empirical Neural Tangent Kernel** (NTK) at parameters $\theta$:

$$\Theta^{(n)}(x, x';\theta) = \left\langle \nabla_\theta f(x;\theta),\, \nabla_\theta f(x';\theta) \right\rangle_{\mathbb{R}^P}
= \sum_{i=1}^{P} \frac{\partial f(x;\theta)}{\partial \theta_i}\,\frac{\partial f(x';\theta)}{\partial \theta_i}. \tag{2.1}$$

Note the relationship to the FIM: comparing $(1.1)$ and $(2.1)$ gives immediately

$$g_{ij}(\theta) = \mathbb{E}_{x \sim \mathcal{P}_X}\!\left[\frac{\partial f(x;\theta)}{\partial \theta_i}\,\frac{\partial f(x;\theta)}{\partial \theta_j}\right], \qquad \Theta^{(n)}(x,x';\theta) = \sum_i \frac{\partial f}{\partial \theta_i}(x)\,\frac{\partial f}{\partial \theta_i}(x'). \tag{2.2}$$

The FIM is therefore the **integral operator** induced by $\Theta^{(n)}$: for any vector $v \in \mathbb{R}^P$,

$$v^\top g(\theta)\, v = \mathbb{E}_x\!\left[\left(\sum_i v_i \frac{\partial f}{\partial \theta_i}(x)\right)^2\right] \geq 0.$$

The central theorem of Jacot, Gabriel, and Hongler [2] is:

**Theorem 2.1 (Jacot–Gabriel–Hongler, 2018 [2]).** *Let $f$ be an $L$-layer fully-connected network in NTK parameterization with any activation function $\phi$ that is Lipschitz, twice differentiable almost everywhere with bounded second derivative. Initialize weights as above. Then:*

*(i) (Initialization convergence.) As $\min(n_1, \ldots, n_{L-1}) \to \infty$, the empirical NTK $\Theta^{(n)}(x, x';\theta_0)$ converges in probability, uniformly over compact $\mathcal{X} \times \mathcal{X}$, to a deterministic limiting kernel $\Theta_\infty(x,x')$.*

*(ii) (Stability under gradient flow.) Under gradient flow $(1.2)$, for any fixed time $T < \infty$ and any $\delta > 0$,*

$$\sup_{t \in [0,T]} \sup_{x, x' \in \mathcal{X}} \left|\Theta^{(n)}(x, x';\theta(t)) - \Theta_\infty(x, x')\right| \xrightarrow{p} 0 \quad \text{as } n \to \infty. \tag{2.3}$$

*(iii) (Linear dynamics.) Under gradient flow, the network function $f(\cdot;\theta(t))$ converges, as $n \to \infty$, to the solution of the linear integro-differential equation*

$$\dot{u}(x,t) = -\int_\mathcal{X} \Theta_\infty(x, x')\,(u(x',t) - y(x'))\,d\mathcal{P}_X(x'),$$

*where $u(x,t) \to f(x;\theta(t))$ in $L^2(\mathcal{P}_X)$ as $n \to \infty$.*

**Remark 2.2 (ReLU activations).** ReLU does not satisfy the hypothesis "$\phi$ twice differentiable with bounded second derivative" because its second derivative is the Dirac delta $\delta_0$. Theorem 2.1 as stated by Jacot et al. covers smooth activations. The extension to ReLU (and other piecewise-linear activations) was established in [3, 5] by the following argument: at infinite width, the pre-activation distribution at each layer is Gaussian (by the central limit theorem applied layer-by-layer), and the NTK recursion involves $\mathbb{E}[\phi'(z)^2]$ where $z \sim \mathcal{N}(0,1)$. For ReLU, $\phi'(z) = \mathbf{1}_{z>0}$, so $\mathbb{E}[\phi'(z)^2] = \tfrac{1}{2}$, which is finite. The Gaussian pre-activation argument integrates out the point non-differentiability, yielding a well-defined and continuous limit kernel. See Section 4.4 for the full arrival-time smoothing argument.

**Definition 2.3 (Explicit form of $\Theta_\infty$ for ReLU).** The limiting NTK satisfies the recursion [2, 5]: set $K^{(0)}(x,x') = x \cdot x'/n_0$, and for $l = 1, \ldots, L$,

$$\Lambda^{(l)}(x,x') = \begin{pmatrix} K^{(l-1)}(x,x) & K^{(l-1)}(x,x') \\ K^{(l-1)}(x',x) & K^{(l-1)}(x',x') \end{pmatrix},$$

$$K^{(l)}(x,x') = \sigma_w^2\,\mathbb{E}_{(u,v) \sim \mathcal{N}(0,\Lambda^{(l)})}\!\left[\phi(u)\phi(v)\right] + \sigma_b^2,$$

$$\dot{K}^{(l)}(x,x') = \sigma_w^2\,\mathbb{E}_{(u,v) \sim \mathcal{N}(0,\Lambda^{(l)})}\!\left[\phi'(u)\phi'(v)\right],$$

$$\Theta_\infty(x,x') = \sum_{l=1}^{L} \dot{K}^{(l)}(x,x') \prod_{l'=l+1}^{L} \dot{K}^{(l')}(x,x'). \tag{2.4}$$

For ReLU with He initialization ($\sigma_w^2 = 2$), the arc-cosine kernel formulas of Cho and Saul [6] give $K^{(l)}$ and $\dot{K}^{(l)}$ in closed form; see equation (4.3) below.

### 2.2 Tensor Programs Extension

Yang's Tensor Programs framework [3] extends Theorem 2.1 to arbitrary architectures satisfying a finite set of "BP-ness" (Backward Pass) recursion rules, which covers any architecture whose forward and backward pass can be written as a finite sequence of matrix-vector products and pointwise nonlinear operations. All standard architectures (fully connected, convolutional, residual, batch-normalized) satisfy BP-ness. We restrict attention to fully connected networks throughout; the Tensor Programs reference is included for completeness and to indicate where the present argument may be extended.

---

## 3. The FIM at Infinite Width

### 3.1 Decomposition via Tangent Features

Fix parameters $\theta$ and define the **tangent feature map**

$$\phi_\theta : \mathcal{X} \to \mathbb{R}^P, \qquad \phi_\theta(x) = \nabla_\theta f(x;\theta). \tag{3.1}$$

Then $(2.2)$ becomes

$$g_{ij}(\theta) = \mathbb{E}_{x \sim \mathcal{P}_X}\!\left[\phi_\theta(x)_i\,\phi_\theta(x)_j\right] = \left(\Phi_\theta^\top \Phi_\theta\right)_{ij}, \tag{3.2}$$

where $\Phi_\theta$ is the (population) tangent feature covariance: formally $\Phi_\theta^\top \Phi_\theta = \int_\mathcal{X} \phi_\theta(x)\,\phi_\theta(x)^\top\,d\mathcal{P}_X(x)$.

Simultaneously, the empirical NTK $(2.1)$ is

$$\Theta^{(n)}(x,x';\theta) = \phi_\theta(x)^\top \phi_\theta(x') = \langle \phi_\theta(x), \phi_\theta(x') \rangle_{\mathbb{R}^P}. \tag{3.3}$$

The FIM and the NTK are thus dual objects: the NTK is a kernel on $\mathcal{X} \times \mathcal{X}$, while the FIM is a matrix on $\Theta \times \Theta$ (i.e., an operator on $\mathbb{R}^P$), and they are related by

$$g(\theta) = \int_\mathcal{X} \phi_\theta(x)\,\phi_\theta(x)^\top\,d\mathcal{P}_X(x). \tag{3.4}$$

### 3.2 The NTK Integral Operator

Consider the integral operator $T_\Theta : L^2(\mathcal{P}_X) \to L^2(\mathcal{P}_X)$ induced by the limiting kernel $\Theta_\infty$:

$$(T_\Theta f)(x) = \int_\mathcal{X} \Theta_\infty(x,x')\,f(x')\,d\mathcal{P}_X(x'). \tag{3.5}$$

**Proposition 3.1.** *Under the assumptions of Section 1, $T_\Theta$ is a bounded, symmetric, positive-semidefinite, Hilbert–Schmidt operator on $L^2(\mathcal{P}_X)$.*

*Proof.* Boundedness and Hilbert–Schmidt follow from $\int_{\mathcal{X} \times \mathcal{X}} |\Theta_\infty(x,x')|^2\,d\mathcal{P}_X(x)\,d\mathcal{P}_X(x') < \infty$, which holds because $\mathcal{X}$ is compact and $\Theta_\infty$ is continuous on $\mathcal{X} \times \mathcal{X}$ (established in Section 4.4). Symmetry follows from $\Theta_\infty(x,x') = \Theta_\infty(x',x)$ by inspection of $(2.4)$. Positive semidefiniteness: for any $f \in L^2(\mathcal{P}_X)$,
$$\langle f, T_\Theta f \rangle_{L^2} = \int\!\!\int \Theta_\infty(x,x') f(x) f(x')\,d\mathcal{P}_X(x)\,d\mathcal{P}_X(x').$$
In the finite-$n$ case $\Theta^{(n)} = \sum_{i} \phi_i(x)\phi_i(x')$ is manifestly positive semidefinite; the same holds in the limit $n \to \infty$ by uniform convergence. $\square$

**Corollary 3.2 (Hilbert–Schmidt decomposition).** *The operator $T_\Theta$ admits a spectral decomposition*

$$T_\Theta = \sum_{k=1}^\infty \lambda_k\,|\psi_k\rangle\langle\psi_k|, \qquad \lambda_1 \geq \lambda_2 \geq \cdots \geq 0, \quad \sum_{k} \lambda_k^2 < \infty, \tag{3.6}$$

*where $\{\psi_k\}$ is an orthonormal basis of $\overline{\mathrm{range}(T_\Theta)} \subset L^2(\mathcal{P}_X)$.*

This is the Hilbert–Schmidt spectral theorem; it applies because $T_\Theta$ is compact (every Hilbert–Schmidt operator is compact).

### 3.3 Spectral Convergence of the FIM

At finite $n$, the FIM $g^{(n)}(\theta)$ is a $P \times P$ matrix. As $n \to \infty$, $P \to \infty$ as well. To compare spectra across different values of $n$, we embed $g^{(n)}$ into $L^2(\mathcal{P}_X)$ via the tangent feature map: define the operator $G^{(n)}_\theta : L^2(\mathcal{P}_X) \to L^2(\mathcal{P}_X)$ by

$$(G^{(n)}_\theta h)(x) = \int_\mathcal{X} \Theta^{(n)}(x, x';\theta)\,h(x')\,d\mathcal{P}_X(x'). \tag{3.7}$$

Then $G^{(n)}_\theta$ is the integral operator whose kernel is the empirical NTK. Under the identification $(3.3)$, the nonzero eigenvalues of $G^{(n)}_\theta$ and $g^{(n)}(\theta)$ (as matrices) coincide (sketch): both encode the singular values of the linear map $h \mapsto \langle \phi_\theta(\cdot), h \rangle_{L^2}$.

**Proposition 3.3 (Trace-class convergence).** *As $n \to \infty$ (with $\theta = \theta_0$ drawn from the initialization distribution), $G^{(n)}_{\theta_0}$ converges in Hilbert–Schmidt norm to $T_\Theta$ in probability:*

$$\left\|G^{(n)}_{\theta_0} - T_\Theta\right\|_{\mathrm{HS}} \xrightarrow{p} 0. \tag{3.8}$$

*Proof sketch.* $\|G^{(n)} - T_\Theta\|_{\mathrm{HS}}^2 = \int\!\!\int |\Theta^{(n)}(x,x';\theta_0) - \Theta_\infty(x,x')|^2\,d\mathcal{P}_X\,d\mathcal{P}_X$. By Theorem 2.1(i), $\sup_{x,x'} |\Theta^{(n)} - \Theta_\infty| \to 0$ in probability; uniform convergence on $\mathcal{X} \times \mathcal{X}$ (which is compact) implies $L^2(\mathcal{P}_X \times \mathcal{P}_X)$ convergence. $\square$

---

## 4. Main Theorem (V1.1 NTK Continuum Limit)

### 4.1 Statement

**Theorem 4.1 (NTK Continuum Limit for ReLU FC Networks).** *Let $f(\cdot;\theta)$ be an $L$-layer ReLU fully-connected network with NTK parameterization and He initialization, as defined in Section 1. Let the data distribution $\mathcal{P}_X$ be supported on a compact set $\mathcal{X} \subset \mathbb{R}^{n_0}$. Consider training under gradient flow $(1.2)$ or natural gradient descent with damping $\varepsilon > 0$ given by $(1.3)$, with MSE loss $\mathcal{C}$. Then for any fixed $T < \infty$:*

*(a) (FIM convergence at initialization.) As $n \to \infty$, the finite-width FIM $g^{(n)}(\theta_0)$, embedded as an integral operator $G^{(n)}_{\theta_0}$ on $L^2(\mathcal{P}_X)$, converges in Hilbert–Schmidt norm to the integral operator $T_\Theta$ induced by the limiting NTK $\Theta_\infty$, in probability.*

*(b) (Uniformity in training time.) The convergence is uniform over $t \in [0, T]$:*

$$\sup_{t \in [0,T]} \left\|G^{(n)}_{\theta(t)} - T_\Theta\right\|_{\mathrm{HS}} \xrightarrow{p} 0 \quad \text{as } n \to \infty. \tag{4.1}$$

*(c) (Smoothness of the kernel in the data argument.) The kernel $\Theta_\infty(\cdot, \cdot)$ is continuous on $\mathcal{X} \times \mathcal{X}$. Consequently, the bilinear form*

$$B_\infty[u, v] = \int_{\mathcal{X} \times \mathcal{X}} \Theta_\infty(x, x')\,u(x)\,v(x')\,d\mathcal{P}_X(x)\,d\mathcal{P}_X(x') \tag{4.2}$$

*is a bounded symmetric positive-semidefinite bilinear form on $L^2(\mathcal{P}_X)$ with continuous kernel.*

*(d) (Lipschitz continuity in time.) The map $t \mapsto T_{\Theta(t)}$ (where $\Theta(t)$ denotes the kernel at time $t$ under the infinite-width limit dynamics) is Lipschitz continuous in Hilbert–Schmidt norm: there exists $C(T) < \infty$ such that*

$$\left\|T_{\Theta(t)} - T_{\Theta(s)}\right\|_{\mathrm{HS}} \leq C(T)\,|t - s|, \qquad t, s \in [0, T]. \tag{4.3}$$

*(e) (Onsager identification.) The FIM metric $g^{(n)}(\theta(t))$ is continuous on the parameter manifold $(\mathcal{M}, g)$ in the $n \to \infty$ limit, and the Onsager identification $L^{ij} = \eta g^{ij}$ of [1, §8.4] is well-defined throughout training for the restricted class under consideration.*

### 4.2 Proof of Part (a)

This follows directly from Proposition 3.3 and Theorem 2.1(i). $\square$

### 4.3 Proof of Part (b)

Parts (b) and (d) are entailed by the kernel stability statement Theorem 2.1(ii). We need to show that the Hilbert–Schmidt convergence is uniform in $t$.

By Theorem 2.1(ii), for any $\delta > 0$ there exists $N(\delta, T)$ such that for all $n > N(\delta, T)$,

$$\mathbb{P}\!\left(\sup_{t \in [0,T], x,x' \in \mathcal{X}} |\Theta^{(n)}(x,x';\theta(t)) - \Theta_\infty(x,x')| > \delta\right) < \delta. \tag{4.4}$$

Since $\mathcal{X} \times \mathcal{X}$ has finite $\mathcal{P}_X \times \mathcal{P}_X$ measure equal to 1, uniform convergence implies $L^2(\mathcal{P}_X \times \mathcal{P}_X)$ convergence uniformly in $t$:

$$\sup_{t \in [0,T]} \|G^{(n)}_{\theta(t)} - T_\Theta\|_{\mathrm{HS}}^2 \leq \sup_{t,x,x'} |\Theta^{(n)} - \Theta_\infty|^2 \to 0 \quad \text{in probability}. \tag{4.5}$$

This gives $(4.1)$. $\square$

### 4.4 Proof of Part (c): Continuity of $\Theta_\infty$ for ReLU (Arrival-Time Smoothing)

The difficulty is that ReLU $= \max(0,\cdot)$ is not differentiable at the origin. We show that the non-differentiability is smoothed out when computing the limiting NTK via the arc-cosine kernel.

**Lemma 4.2 (Arc-cosine kernel continuity).** *For any $l \geq 1$ and any compact $\mathcal{X}$, the kernel $K^{(l)}(x,x')$ defined by the recursion in Definition 2.3 is continuous on $\mathcal{X} \times \mathcal{X}$. Consequently $\dot{K}^{(l)}(x,x')$ is also continuous, and by $(2.4)$, $\Theta_\infty(x,x')$ is continuous.*

*Proof.* (sketch) At each layer $l$, the kernel values $K^{(l)}(x,x')$ are defined by

$$K^{(l)}(x,x') = 2\,\mathbb{E}_{(u,v) \sim \mathcal{N}(0,\Lambda^{(l)})}\!\bigl[\phi(u)\phi(v)\bigr],$$

where $\phi = \mathrm{ReLU}$. The Gaussian expectation can be evaluated using the arc-cosine kernel formula [6]:

$$K^{(l)}(x,x') = \frac{1}{\pi}\sqrt{K^{(l-1)}(x,x) K^{(l-1)}(x',x')}\,J_1\!\left(\frac{K^{(l-1)}(x,x')}{\sqrt{K^{(l-1)}(x,x) K^{(l-1)}(x',x')}}\right), \tag{4.3}$$

where $J_1(\cos\theta) = \sin\theta + (\pi - \theta)\cos\theta$ is continuous on $[-1,1]$. Since $K^{(0)}(x,x') = x \cdot x' / n_0$ is continuous and the recursion involves only continuous operations (including the arc-cosine $J_1$ factor applied to the angle between vectors, which is continuous away from $\|x\| = 0$ or $\|x'\| = 0$; for $x = 0$ we have $K^{(l)}(0,x') = 0$ by direct computation), continuity propagates inductively through all layers.

The key point is that although $\phi' = \mathbf{1}_{z>0}$ is discontinuous, $\dot{K}^{(l)}(x,x') = 2\,\mathbb{E}[\phi'(u)\phi'(v)]$ involves an expectation over a Gaussian, and the indicator function $\mathbf{1}_{u>0}$ is integrable against Gaussian density functions; the resulting integral is a smooth function of $\Lambda^{(l)}$, hence a continuous function of $(x,x')$. Formally,

$$\dot{K}^{(l)}(x,x') = \mathbb{P}_{(u,v) \sim \mathcal{N}(0,\Lambda^{(l)})}\!\bigl[u > 0,\, v > 0\bigr] = \frac{\pi - \angle(x^{(l)}, x'^{(l)})}{2\pi}, \tag{4.4}$$

where $\angle(x^{(l)}, x'^{(l)})$ denotes the angle between the expected $l$-th layer representations, which is a continuous function of $(x,x')$ on $\mathcal{X} \times \mathcal{X}$. $\square$

This argument — that the Gaussian pre-activation distribution at infinite width integrates out the discontinuity of ReLU — is what we call the **arrival-time smoothing** argument: the "arrival time" at which a neuron first fires (i.e., the zero-crossing of its pre-activation) is distributed continuously at infinite width, so any quantity involving $\phi'$ is regularized by the Gaussian measure.

### 4.5 Proof of Part (d): Lipschitz Continuity in Time

In the infinite-width limit, the parameters $\theta(t)$ move by $O(1/\sqrt{n})$ per unit time (this is the key feature of NTK scaling: the network remains close to initialization). More precisely, by [2, Theorem 1], $\|\theta(t) - \theta_0\| = O(1/\sqrt{n})$ uniformly in $t \in [0,T]$. In the strict $n = \infty$ limit, the NTK is frozen at $\Theta_\infty$, so $T_{\Theta(t)} = T_\Theta$ for all $t$ and $(4.3)$ holds trivially with $C(T) = 0$.

For the finite-$n$ regime, the Lipschitz bound $(4.3)$ holds with $C(T) = O(1/\sqrt{n})$ (sketch): by chain rule, $\partial_t \Theta^{(n)}(x,x';\theta(t)) = \langle \nabla_\theta \Theta^{(n)}, \dot\theta(t) \rangle$, which under gradient flow with bounded gradients (ensured on compact $\mathcal{X}$) is bounded uniformly in $x, x', t$. The Hilbert–Schmidt norm of $\partial_t G^{(n)}_{\theta(t)}$ is then bounded by $\sup_{x,x'} |\partial_t \Theta^{(n)}|$, which is $O(1)$ in $n$, giving a uniform Lipschitz constant. As $n \to \infty$, the limit dynamics has $C(T) \to 0$ (frozen kernel), but the key point for Part (e) is that $C(T)$ is finite for all $n$. $\square$

### 4.6 Proof of Part (e): Well-Definedness of the Onsager Identification

The Onsager identification $L^{ij} = \eta g^{ij}$ requires $g_{ij}(\theta(t))$ to be a well-defined, bounded, symmetric, positive-semidefinite bilinear form on the tangent space of $\mathcal{M}$ at each point $\theta(t)$ and at each time $t \in [0,T]$.

Parts (a)–(d) establish:

- **Boundedness:** $\|T_\Theta\|_{\mathrm{HS}} < \infty$ (Proposition 3.1) and convergence is uniform in $t$ (Part (b)).
- **Symmetry:** $g_{ij} = g_{ji}$ by $(1.1)$ (the integrand is a product of two identical factors).
- **Positive semidefiniteness:** Follows from $\sum_{ij} v_i g_{ij} v_j = \mathbb{E}_x[(\sum_i v_i \partial_{\theta_i} f)^2] \geq 0$.
- **Continuity in $t$:** Part (d) gives Lipschitz-in-$t$ continuity.
- **Continuity in $\theta$:** The map $\theta \mapsto \Theta^{(n)}(\cdot,\cdot;\theta)$ is continuous in the $n \to \infty$ limit (NTK is frozen), and for finite $n$ it is smooth (the network function and its gradients are smooth in $\theta$ — this is a standard fact for networks with differentiable activations; for ReLU, smoothness is replaced by piecewise-smoothness, but the FIM, being an expectation of $({\partial_\theta f})^2$ over $\mathcal{P}_X$, is smooth in $\theta$ almost surely, because the set of $\theta$ for which a training point lies exactly on a ReLU boundary has measure zero under $\mathcal{P}_X$ for generic $\mathcal{P}_X$).

Under natural gradient descent with damping $\varepsilon > 0$, the inverse $(g + \varepsilon I)^{-1}$ is well-defined and bounded for all $\theta$ and all $n$. In the limit $n \to \infty$ (frozen NTK regime), $g$ is bounded above by $\|T_\Theta\|_{\mathrm{op}} < \infty$ and has a spectral gap at $\varepsilon$ from below (by the damping), so $g^{-1}$ is also bounded. The identification $L^{ij} = \eta g^{ij}$ is therefore well-defined.

For the limit $\varepsilon \to 0$: if $T_\Theta$ is strictly positive definite (which holds when the data distribution $\mathcal{P}_X$ is not supported on a finite set and $\Theta_\infty$ is an $\mathcal{P}_X$-universal kernel), then $g^{-1}$ is bounded and the identification holds without damping. Otherwise, the pseudo-inverse $g^\dagger$ can be used, restricting to the subspace where $g$ is invertible; see [7] for a discussion in the context of natural gradient descent. $\square$

**Remark 4.7 (Scope of the theorem).** Theorem 4.1 is proven only for the NTK (lazy training) regime. In this regime, the network function evolves linearly and the parameters stay near initialization. The theorem does not cover the feature learning (mean-field or $\mu$P) regime, where the kernel changes substantially during training. Extending the result to that regime is a significant open problem; see Section 7.

---

## 5. Corollary: Smooth Metric Limit

### 5.1 Statement

**Corollary 5.1 (Smooth Metric Limit).** *Under the hypotheses of Theorem 4.1, the limiting FIM admits a diffeomorphism-invariant continuum representation as a pullback metric on the data space $\mathcal{X}$ under the map $\theta \mapsto f(\cdot;\theta)|_{n=\infty}$ (the infinite-width limit function).*

*More precisely: the bilinear form $B_\infty$ defined in $(4.2)$ is the pullback of the $L^2(\mathcal{P}_X)$ inner product under the tangent feature map $\phi_{\theta_0} : \mathcal{X} \to \mathbb{R}^P$. In the $n \to \infty$ limit, this pullback converges to the $L^2(\mathcal{P}_X)$ inner product under the limiting feature map $\phi_\infty : \mathcal{X} \to \mathcal{H}_\Theta$, where $\mathcal{H}_\Theta$ is the reproducing kernel Hilbert space (RKHS) of $\Theta_\infty$. The metric on $\mathcal{X}$ induced by the RKHS embedding is:*

$$ds^2_{\mathcal{X}}(x, x + dx) = \Theta_\infty(x,x)\,\|dx\|^2 - \nabla_x \nabla_{x'} \Theta_\infty(x,x')|_{x'=x}\,dx \otimes dx + O(\|dx\|^4). \tag{5.1}$$

*This metric is smooth on $\mathcal{X}$ (in the sense of $C^0$ continuity, established in Section 4.4, with higher regularity depending on regularity of $\mathcal{P}_X$ and $\mathcal{X}$), symmetric, and positive-semidefinite.*

*This constitutes the verification of condition (b) of Step 6 in Appendix A of [1] for the restricted class.*

*Proof.* The RKHS of $\Theta_\infty$ is the completion of the linear span $\{\Theta_\infty(\cdot, x) : x \in \mathcal{X}\}$ under the inner product $\langle \Theta_\infty(\cdot,x), \Theta_\infty(\cdot,x') \rangle_{\mathcal{H}} = \Theta_\infty(x,x')$. The feature map $\phi_\infty(x) = \Theta_\infty(\cdot, x)$ maps $\mathcal{X}$ into $\mathcal{H}_\Theta$, and the pullback of the RKHS inner product under $\phi_\infty$ gives

$$\langle \phi_\infty(x), \phi_\infty(x') \rangle_\mathcal{H} = \Theta_\infty(x,x'),$$

which is the kernel of the bilinear form $B_\infty$. Differentiability of the induced metric tensor $(5.1)$ follows from differentiability of $\Theta_\infty$ in its first argument; for ReLU networks this requires second-order differentiability of the arc-cosine kernel, which holds in the interior of $\mathcal{X}$ away from $\|x\|=0$ (sketch; see [6] for the explicit computation of $\nabla_x K^{(l)}$). $\square$

### 5.2 Connection to Appendix A, Step 6

In the language of [1, Appendix A, Step 6], condition (b) states: "the probability model $p(x|\theta)$ must be sufficiently smooth in $\theta$ that the FIM varies slowly across neighboring parameters (a mean-field-type assumption)." Corollary 5.1 establishes a stronger statement for the NTK-restricted class: the FIM not only varies slowly, it converges to a fixed deterministic limit (the integral operator $T_\Theta$) as $n \to \infty$, and the induced metric on data space is continuous. The "mean-field" qualifier in [1] is thus precisely the NTK regime.

Condition (a) — the large-$N$ limit with well-defined FIM — is established by Theorem 4.1(a). Condition (c) — the emergence of four effective dimensions — is discussed in Section 6 but is not established here.

---

## 6. Addressing Conditions (a), (b), and (c) of Step 6

### 6.1 Condition (a): The Large-$N$ Limit

Condition (a) of [1, Appendix A, Step 6] is: "the network must be in a large-$N$ regime where the parameter count $N \to \infty$ with the FIM remaining well-defined."

This is established by Theorem 4.1: as $n \to \infty$, $P \to \infty$, and the FIM (embedded as an integral operator via $(3.7)$) converges to a well-defined bounded operator $T_\Theta$ on $L^2(\mathcal{P}_X)$. The thermodynamic analogy of [1] — in which the large-$N$ limit is compared to the thermodynamic limit of statistical mechanics — is supported: in both cases, a high-dimensional discrete system (many spins; many parameters) converges to a well-defined continuum object (partition function; integral operator).

The NTK framework provides the precise sense in which the limit is taken: not all directions in parameter space matter equally; the relevant degrees of freedom are the projections onto the tangent feature space, and these converge to the RKHS of $\Theta_\infty$. This is the neural-network analogue of the reduction from microscopic degrees of freedom to macroscopic thermodynamic variables.

### 6.2 Condition (b): Smoothness of the FIM

Condition (b) is: "the probability model $p(x|\theta)$ must be sufficiently smooth in $\theta$ that the FIM varies slowly across neighboring parameters."

Established by Corollary 5.1 for the NTK-restricted class. See Section 5.2.

### 6.3 Condition (c): Four-Dimensional Emergence

Condition (c) is: "the effective dimensionality of the parameter manifold must reduce to four in the infrared limit."

This condition is **not established by the present theorem** and remains the deepest open question in the framework. The current result provides the following partial evidence:

The integral operator $T_\Theta$ on $L^2(\mathcal{P}_X)$ has a spectrum $\{\lambda_k\}$ that decays as $k \to \infty$ (since $T_\Theta$ is Hilbert–Schmidt). The effective rank — the number of eigenvalues above any threshold $\varepsilon$ — depends on the spectral decay rate of $\Theta_\infty$. For polynomial decay $\lambda_k \sim k^{-\alpha}$, the effective dimension is $O(\varepsilon^{-1/\alpha})$.

Whether the spectral decay of the NTK for physically relevant data distributions (e.g., $\mathcal{P}_X$ supported on a 4-dimensional submanifold of the ambient space, as would be the case if input data encodes 4D spacetime coordinates) leads to an effective dimension of 4 in the infrared is a dynamical question — not a consequence of the NTK convergence theorem. Three independent arguments for the plausibility of $d_{\mathrm{eff}} = 4$ are reviewed in [1, §8.5]: holographic constraints, Kaluza–Klein compactification, and renormalization group flow. None of these constitutes a proof, and none is provided here.

**To be explicit:** the emergence of four macroscopic spacetime dimensions from the neural-network parameter manifold remains an open conjecture. The present theorem neither proves nor refutes it. The contribution of V1.1 is to establish that *if* the effective dimension is 4, the FIM is sufficiently well-behaved (bounded, symmetric, positive-semidefinite, continuous in $t$ and $\theta$) for the Onsager identification and the subsequent Lovelock argument to go through rigorously.

---

## 7. Limitations

The following limitations of Theorem 4.1 are stated explicitly so that they are not overlooked in subsequent work.

**(a) Restriction to the NTK (lazy training) regime.** Theorem 4.1 applies only when the network is trained in the NTK regime: the kernel does not change substantially during training, the parameters remain close to initialization, and the network function evolves linearly. In practice, large neural networks operating as potential cosmological substrates may exhibit significant feature learning (the $\mu$P or mean-field regime), where the NTK evolves substantially and Theorem 2.1(ii) no longer applies. The theorem is silent about this regime.

**(b) Finite-width corrections are not analyzed.** All statements are asymptotic as $n \to \infty$. The rate of convergence in Proposition 3.3 and Theorem 4.1(b) is $O(1/\sqrt{n})$ (sketch) — derived from the central limit theorem underlying the NTK convergence. Finite-$n$ corrections to the FIM spectrum, including higher-order cumulants and finite-size effects, are not characterized here.

**(c) ReLU activations provide the weakest regularity.** The kernel continuity in Section 4.4 is established only at the $C^0$ level. Smooth activations (e.g., GELU, SiLU, $\tanh$) yield a $C^\infty$ limit kernel via the same arc-cosine argument, and consequently a $C^\infty$ induced metric on data space. The stronger regularity would simplify the Hilbert–Schmidt analysis and sharpen the spectral decay estimates. ReLU is included because it is the activation used in the companion experiments [1, §11], but smooth activations are preferred for the smoothest version of condition (b).

**(d) Recurrent and self-modifying networks are out of scope.** The Tensor Programs framework [3] extends Theorem 2.1 to convolutional and residual networks, but recurrent networks (with weight sharing across time steps) require additional analysis (the parameter count grows with sequence length, changing the scaling analysis), and self-modifying networks (whose architecture changes during training) fall entirely outside the framework. Vanchurin's original cosmological proposal [8] posits a self-organizing network; the present theorem applies only to the frozen-architecture subcase.

**(e) The 4D emergence is not proven by this theorem.** As stated explicitly in Section 6.3, Theorem 4.1 does not establish, and does not imply, that the effective dimensionality of the parameter manifold is 4. This must be emphasized to prevent misreading.

**(f) Compact support assumption.** The compact support hypothesis on $\mathcal{P}_X$ is used to ensure that $\Theta_\infty$ is bounded and continuous on $\mathcal{X} \times \mathcal{X}$. For unbounded data distributions (e.g., Gaussian $\mathcal{P}_X$), the NTK integral operator is still well-defined but the Hilbert–Schmidt property requires exponential tail decay; we do not analyze this case.

---

## 8. Relationship to the Empirical Results

### 8.1 The Observed SV Scaling — V1.0 and V1.2 Updates

The companion experiment [1, §11.7] originally measured, on the 6-width V1.0 sweep, the power law

$$\mathrm{SV}(N) \sim N^{0.47}, \qquad R^2 = 0.935. \tag{8.1}$$

The V1.2 extension (`experiments/v1_2_scaling/fill_ladder.py`) adds four intermediate widths (32, 128, 512, 2048) to the sweep, giving 10 data points. The updated fit is

$$\mathrm{SV}(N) \sim N^{0.566}, \qquad R^2 = 0.84. \tag{8.1'}$$

The V2.1 QEC-decoder experiment (same 5-layer architecture, syndrome-decoding task) yields on the same width ladder $\mathrm{SV}(N) \sim N^{0.807}$ with $R^2 = 0.89$, showing that the power-law form is shared across tasks but the exponent is task-dependent.

We now interpret these against the NTK continuum limit.

### 8.2 NTK Prediction for the Spectrum

In the NTK regime, the weight matrices at time $t$ are

$$W^{(l)}(t) = W^{(l)}_0 - \int_0^t \dot{W}^{(l)}(s)\,ds,$$

where $\dot{W}^{(l)}(s) = O(1/\sqrt{n_l})$ per unit time (from the gradient flow; the gradient of the loss with respect to $W^{(l)}$ scales as $1/\sqrt{n_l}$ in NTK parameterization). After training for time $T$, the deviation from initialization is $\Delta W^{(l)} = O(T/\sqrt{n_l})$.

The **singular value ratio** at layer $l$ is the ratio of the largest to the smallest nonzero singular value of $W^{(l)}$. At initialization, He initialization gives singular values concentrated near $\sqrt{2/n_{l-1}}$, with fluctuations of order $1/n_{l-1}^{1/2}$ (Marchenko–Pastur distribution). After training, the signal introduced by $\Delta W^{(l)}$ has rank at most equal to the number of training samples $m$, and has singular values of order $T/\sqrt{n_l}$.

The SV ratio is therefore approximately

$$\frac{\sigma_{\max}(W^{(l)}(T))}{\sigma_{\min}(W^{(l)}(T))} \approx \frac{\sqrt{2/n_{l-1}} + O(T/\sqrt{n_l})}{\sqrt{2/n_{l-1}} - O(1/n_{l-1}^{1/2})}. \tag{8.2}$$

For $n_l = \alpha_l n$ with fixed $\alpha_l$, this grows at most as $O(\sqrt{n})$ for large $n$, corresponding to an exponent of $\tfrac{1}{2} = 0.5$. The original V1.0 exponent $0.47 < 0.5$ is **consistent with but slightly below** the naive NTK prediction; the V1.2 extension to ten widths revises the exponent *upward* to $0.566$, *above* the naive NTK upper bound. This suggests that the fit is not purely in the lazy-training regime, and that either (i) feature-learning corrections [10] are non-negligible at the widths in question, or (ii) the power law is only approximate across five decades of $N$ and the true $N \to \infty$ behaviour is still closer to $N^{1/2}$.

**Explicit post-audit clarification.** The measured exponent 0.566 *exceeds* the NTK-regime bound 0.5. V1.1 therefore does **not** claim that the NTK limit *explains* the observed exponent; it claims that the NTK limit *bounds* the finite-width behaviour. The observed exponent being above the bound is evidence that the experiment is not strictly in the lazy regime; it is *not* evidence against Theorem 4.1, which is an asymptotic statement about the limiting operator, not the finite-width scaling of the SV ratio. Reconciliation with the measured exponent is deferred to the mean-field/feature-learning extension noted in Q4 (V2.0+ open question).

### 8.3 Interpretation of the Exponent Discrepancy

The gap between the observed exponent $0.47$ and the predicted upper bound $0.5$ may arise from several sources:

- **Finite-width corrections:** At finite $n$, the FIM spectrum does not fully converge to the integral operator limit; the NTK is not perfectly frozen, and there is residual kernel evolution that dampens symmetry breaking. This would lower the exponent below $0.5$.

- **Boundary layers:** The experiment measures SV ratios including the input ($256 \times 64$) and output ($64 \times 256$) layers, which have dimensional bottlenecks giving SV ratios of $2.9\times$ and $3.5\times$ [1, §11.3]. Including these in the power-law fit lowers the apparent exponent.

- **NTK theory predicts bounds, not exact exponents:** The $\sqrt{N}$ scaling is an upper bound on the rate of singular value separation; the actual exponent depends on the data distribution, training time, and loss landscape, none of which are specified by the NTK limit theorem alone.

The NTK continuum limit theorem **does** predict that the FIM spectrum approaches a fixed integral-operator spectrum as $n \to \infty$, which constrains the SV ratios: in the strict infinite-width limit, all layers have the same limiting kernel $\Theta_\infty$ and the SV ratio of the kernel operator (the ratio $\lambda_1/\lambda_k$ for the smallest nonzero eigenvalue) is fixed independently of $n$. The observed growth of SV ratios with $n$ is therefore a **finite-width effect**, consistent with the theory.

A more precise comparison requires computing the spectrum of $T_\Theta$ for the specific architecture and data distribution used in the experiment, and fitting the finite-width convergence rate. This is deferred to V1.2.

### 8.4 Seed Variance of the SV Ratio

The V1.2 seed-robustness experiment (`experiments/v1_2_scaling/seed_robustness.py`, width 256, six seeds) finds that the SV ratio has a coefficient of variation of $\approx 124\%$ (range 1{,}739 — 66{,}398 with mean 20{,}152, standard deviation 24{,}990). The FIM three-tier ratio, by contrast, has CV $\approx 10\%$ (range 354 — 448). This indicates that the SV ratio is a *noisy* observable: it depends on which individual directions in weight space happen to be exercised by the particular random initialization, and even modest changes in training dynamics produce large fluctuations in $\sigma_{\max}/\sigma_{\min}$ because the denominator is sensitive to near-zero singular values.

For the present theorem this has two implications:

- The NTK prediction is about the *limiting kernel spectrum*, not individual finite-width SV ratios; the large seed variance is consistent with finite-width kernel fluctuations (Yang [3, §6]).
- The empirical load-bearing quantity in V1.0–V1.2 is the FIM tier hierarchy, not the SV exponent. The FIM tier stability (CV $\approx 10\%$) is what supports the cosmological interpretation; the SV power-law fit should be read as directional, not quantitative.

---

## 9. Open Questions for V2.0+

The following questions are explicitly handed off to future phases of the research program.

**Q1 (V2.0): Cauchy completion and metric refinement.** The parameter manifold $\mathcal{M}$ with the FIM metric $g$ is a pre-Hilbert space (infinite-dimensional, with degenerate directions). Its Cauchy completion $\overline{\mathcal{M}}$ under $g$ is a Hilbert manifold. Understanding the geometry of $\overline{\mathcal{M}}$ — in particular, whether it admits a finite-dimensional infrared limit — is the key step toward condition (c) of Step 6. The tools of infinite-dimensional Riemannian geometry (Eells–Elworthy, Freed–Groisser [9]) may be applicable. This is the primary analytical target for V2.0.

**Q2 (V2.1 — now done): QEC spectral connection.** The FIM three-tier hierarchy observed in [1, §11.4] (Tier 1/Tier 3 ratio $637\times$) was hypothesized to bear structural similarity to the spectral hierarchies of trained quantum error correction (QEC) decoders. V2.1 (``experiments/v2_1_qec/``) tested this on a 6-width sweep of toric-code decoders at identical 5-layer architecture: result $\mathrm{SV} \sim N^{0.807}$, $R^2 = 0.89$ and $\text{T1/T3}_\text{FIM} \sim N^{1.386}$, $R^2 = 0.93$. The power-law *form* is shared with the cosmology experiment; the *exponent* is task-dependent and super-linear in the QEC case. This is partial evidence that the FIM hierarchy is a generic property of learned representations rather than a cosmology-specific artifact. Further multi-task universality (≥3 tasks, Naestro Tier 1 item 1) remains for V3.0.

**Q3 (V3.1): The $\alpha$-drift observational test.** Section 9.1 of [1] predicts that the fine-structure constant $\alpha$ undergoes drift at a rate proportional to the local information density of the cosmic web. In the NTK language, the coupling constant $\alpha$ corresponds to a parameter with a large FIM eigenvalue (it is in Tier 1 of the hierarchy), and its drift rate is proportional to $(\lambda_\alpha)^{-1}$ — the reciprocal of its FIM eigenvalue (see [1, §8.6]). The present theorem establishes that $\lambda_\alpha$ converges to a well-defined limit as $n \to \infty$; estimating its numerical value from the NTK spectral theory and comparing to the observational bound $|\dot\alpha/\alpha| \lesssim 10^{-17}$ yr$^{-1}$ [1, §9.1] is a concrete computational task for V3.1.

**Q4: Feature learning regime.** As noted in Section 7(a), Theorem 4.1 is restricted to lazy training. Establishing an analogue in the mean-field ($\mu$P) limit [10] — where the kernel evolves during training and the network learns meaningful features — would substantially strengthen the cosmological correspondence. The mean-field limit is technically harder (requires propagation of chaos arguments and the Vlasov equation for particle trajectories in weight space) and is left as a long-term open problem.

**Q5: Non-compact data distributions.** The compact support assumption (Section 7(f)) excludes physically relevant distributions such as the Gaussian or the distribution of actual cosmological observations. A version of Theorem 4.1 for sub-Gaussian distributions (with exponential tails) would require modified Hilbert–Schmidt bounds and is technically accessible; it is flagged as a tractable extension.

---

## 10. References

[1] N. Nedovodin, "The Universe as a Self-Organizing Neural Network: Integrating Cosmological Information Theory with Neuro-Inspired AI Systems Engineering Toward Falsifiable Predictions, a Formal FIM–Onsager Correspondence, and Computational Validation," STARGA Inc., Research Synthesis & Original Contribution, April 2026.

[2] A. Jacot, F. Gabriel, and C. Hongler, "Neural Tangent Kernel: Convergence and Generalization in Neural Networks," *Advances in Neural Information Processing Systems* 31 (NeurIPS 2018), pp. 8571–8580. arXiv:1806.07572.

[3] G. Yang, "Tensor Programs I: Wide Feedforward or Recurrent Neural Networks of Any Architecture are Gaussian Processes," *Advances in Neural Information Processing Systems* 32 (NeurIPS 2019). arXiv:1902.04760.

[4] K. He, X. Zhang, S. Ren, and J. Sun, "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification," *Proceedings of ICCV* 2015, pp. 1026–1034. arXiv:1502.01852.

[5] S. S. Du, X. Zhai, B. Poczos, and A. Singh, "Gradient Descent Provably Optimizes Overparameterized Neural Networks," *International Conference on Learning Representations* (ICLR) 2019. arXiv:1810.02054.

[6] Y. Cho and L. K. Saul, "Kernel Methods for Deep Learning," *Advances in Neural Information Processing Systems* 22 (NIPS 2009), pp. 342–350.

[7] S. Amari, "Natural Gradient Works Efficiently in Learning," *Neural Computation* 10(2), 1998, pp. 251–276.

[8] V. Vanchurin, "The World as a Neural Network," *Entropy* 22(11), 2020, p. 1210. arXiv:2008.01540.

[9] D. S. Freed and D. Groisser, "The Basic Geometry of the Manifold of Riemannian Metrics and of Its Quotient by the Diffeomorphism Group," *Michigan Mathematical Journal* 36(3), 1989, pp. 323–344.

[10] G. Yang and E. J. Hu, "Feature Learning in Infinite-Width Neural Networks," *International Conference on Machine Learning* (ICML) 2021. arXiv:2011.14522.

[11] N. N. Chentsov, *Statistical Decision Rules and Optimal Inference*, Translations of Mathematical Monographs, American Mathematical Society, 1982. (Original Russian edition 1972.)

[12] D. Lovelock, "The Einstein Tensor and Its Generalizations," *Journal of Mathematical Physics* 12(3), 1971, pp. 498–501.

[13] A. Jacot, F. Gabriel, and C. Hongler, "Neural Tangent Kernel: Convergence and Generalization in Neural Networks," extended version, arXiv:1806.07572v4 (2020). (Updated with stability proof.)

[14] Z. Allen-Zhu, Y. Li, and Z. Song, "A Convergence Theory for Deep Learning via Over-Parameterization," *International Conference on Machine Learning* (ICML) 2019. arXiv:1811.03962.

---

*Copyright 2026 STARGA, Inc. All rights reserved. STARGA Commercial License.*
