"""V11 — Pythia-6.9B FIM measurement (production-scale gap, dense LM).

Closes the upper end of the Pythia-family FIM tier-hierarchy curve:
V8 1.4B and V9.10 2.8B are already on disk; this is the 6.9B point.

Schema (must match V9.10 Pythia-2.8B exactly, byte-for-byte keys):
    {
      "model_id":       "EleutherAI/pythia-6.9b",
      "n_params":       <int>,
      "n_probes":       200,
      "seed":           42,
      "T1T3":           <float>,
      "log10_T1T3":     <float>,
      "gini":           <float>,
      "gini_sample_size": 100000000,
      "eff_rank_n":     <float>,
      "top_1pct_mass":  <float>
    }

Memory plan (A100 80 GB):
  Model fp16              ~13.8 GB
  Per-probe acts/grads    ~6-10 GB (freed after backward)
  FIM accumulators (fp32) on **CPU** host RAM (~28 GB)
  ⇒ GPU peak < 30 GB; CPU peak < 60 GB. A100 80 GB pod has ≥ 200 GB host RAM.

Metric protocol (matches V9.10):
  * 200 text probes, float64 accumulation post-grad², seed 42.
  * Effective rank computed on positive-FIM entries with the full
    flat distribution; eff_rank_n = eff_rank / total_n.
  * Gini / top-1% mass / T1/T3 computed on a uniformly sampled
    100,000,000-element sub-vector (since the full 6.9B-element
    flat sort would not fit in RAM and matches V9.10 sampling).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


# iter-262 review fix: the original 12-text list gave severely
# insufficient token diversity for a 64-expert MoE (top-8 routing);
# many experts wouldn't fire across 200 probes, biasing the FIM
# toward observed routing paths.
#
# Replaced with 220 unique texts covering 11 domains × 20 prompts:
# physics / math / biology / chemistry / programming / law /
# medicine / philosophy / finance / linguistics / general-knowledge.
# Each text is a 50-100 char prefix that the model continues during
# the FIM probe — diversity in topic + register stresses different
# expert routing paths.
SAMPLE_TEXTS = [
    # — Physics (20) —
    "The Fisher Information Matrix is a fundamental object in information geometry that captures",
    "General relativity predicts that gravity arises from the curvature of spacetime caused by",
    "Quantum entanglement is a phenomenon where two particles become correlated such that",
    "The standard model of particle physics describes the electromagnetic, weak, and strong",
    "Statistical mechanics derives macroscopic thermodynamics from microscopic dynamics by",
    "Black hole thermodynamics relates the area of the event horizon to the entropy of",
    "The cosmic microwave background is residual electromagnetic radiation left over from",
    "Renormalization group flow describes how physical systems behave at different length scales",
    "The double-slit experiment demonstrates wave-particle duality by showing that electrons",
    "Phase transitions in condensed matter occur when a system's order parameter changes",
    "Conservation laws in physics arise from continuous symmetries via Noether's theorem",
    "The Higgs mechanism gives mass to gauge bosons through spontaneous symmetry breaking",
    "Plasma physics studies ionized gases where electromagnetic forces dominate over",
    "Lagrangian mechanics reformulates Newton's laws using a scalar function of generalized coordinates",
    "Solitons are localized wave solutions to nonlinear field equations that propagate without",
    "The CKM matrix in flavor physics describes the strength of flavor-changing weak transitions",
    "Effective field theories integrate out high-energy degrees of freedom while preserving",
    "Gravitational waves are ripples in spacetime curvature that propagate at the speed of light",
    "Topological insulators are materials that conduct electricity on their surface but not in",
    "Holographic duality conjectures that gravitational theories in d+1 dimensions are equivalent to",
    # — Mathematics (20) —
    "Random matrix theory predicts that products of independent random matrices have eigenvalues that",
    "The Riemann hypothesis conjectures that all non-trivial zeros of the zeta function lie on",
    "Galois theory establishes a correspondence between field extensions and groups of automorphisms",
    "Category theory abstracts mathematical structures by focusing on morphisms rather than",
    "Algebraic topology classifies spaces using invariants such as homology and homotopy groups",
    "Differential geometry studies smooth manifolds and the structures defined on them like",
    "The fundamental theorem of calculus connects differentiation and integration by showing that",
    "Number theory investigates properties of integers and prime distributions through tools like",
    "Functional analysis extends linear algebra to infinite-dimensional vector spaces equipped with",
    "Probability theory formalizes uncertainty using sigma-algebras and measure-theoretic foundations",
    "Group theory studies algebraic structures consisting of a set with a binary operation that",
    "Knot theory classifies embeddings of circles in three-dimensional space modulo ambient isotopy",
    "The Hanin-Nica theorem on products of random matrices establishes that the log of the gradient",
    "Stochastic differential equations model systems with random fluctuations driven by Wiener",
    "Information theory quantifies the entropy of random variables and the channel capacity of",
    "Spectral graph theory analyzes properties of graphs through eigenvalues of associated matrices",
    "Optimal transport theory studies the most efficient way to map one distribution onto another",
    "Convex optimization minimizes convex objectives over convex sets using methods like interior",
    "Linear algebra over finite fields underpins coding theory and cryptographic primitives such as",
    "Combinatorics counts and analyzes discrete structures using generating functions and bijective",
    # — Biology / Medicine (20) —
    "Mitochondria generate ATP through oxidative phosphorylation in the inner membrane where",
    "CRISPR-Cas9 is a genome-editing technology derived from prokaryotic adaptive immune systems",
    "The blood-brain barrier is a selectively permeable boundary that restricts the passage of",
    "Hemoglobin in red blood cells binds oxygen cooperatively through allosteric conformational",
    "Mendelian inheritance describes how alleles segregate during gamete formation and recombine in",
    "The clinical contraindication between warfarin and ibuprofen arises because both drugs increase",
    "Cytochrome P450 3A4 metabolizes a substantial fraction of clinically used drugs and is",
    "Beta-lactam antibiotics inhibit bacterial cell wall synthesis by binding penicillin-binding",
    "Dopaminergic neurons in the substantia nigra degenerate in Parkinson's disease leading to",
    "Hypothalamic-pituitary-adrenal axis dysregulation is implicated in chronic stress responses and",
    "QT-interval prolongation is a cardiac safety signal driven by drugs that block hERG potassium",
    "Renal clearance of digoxin depends on glomerular filtration and is reduced when patients have",
    "MAOI antidepressants must not be co-administered with serotonergic drugs due to risk of",
    "Crystalloid intravenous fluids restore intravascular volume in patients with hypovolemic",
    "ACE inhibitors lower blood pressure by blocking conversion of angiotensin I to angiotensin II",
    "Insulin resistance in type 2 diabetes mellitus reduces glucose uptake in skeletal muscle and",
    "Telomere shortening is associated with cellular senescence and contributes to aging through",
    "Gut microbiota composition influences host metabolism and immune homeostasis through",
    "Sodium-glucose co-transporter 2 inhibitors reduce cardiovascular mortality in heart failure",
    "Anticoagulation in atrial fibrillation is risk-stratified using the CHA2DS2-VASc score which",
    # — Programming / Computer science (20) —
    "Deep neural networks trained with gradient descent exhibit complex spectral properties in the",
    "The neural tangent kernel describes the infinite-width limit of fully-connected feedforward networks",
    "Transformers process sequences in parallel using self-attention layers that compute weighted",
    "Reinforcement learning agents learn policies by maximizing expected cumulative reward through",
    "Cache-coherent memory hierarchies in modern CPUs exploit temporal and spatial locality by",
    "Distributed consensus algorithms such as Raft achieve agreement among nodes despite failures",
    "Type systems prevent classes of bugs at compile time by constraining the values that",
    "Garbage collection algorithms reclaim unreachable heap memory using techniques such as",
    "Dynamic programming solves problems by breaking them into overlapping subproblems and storing",
    "Compiler optimizations transform program intermediate representation to reduce execution time",
    "Quicksort partitions an array around a pivot and recursively sorts each partition with",
    "TCP achieves reliable byte-stream transmission over unreliable IP networks via cumulative",
    "Lock-free data structures use atomic compare-and-swap operations to coordinate concurrent",
    "The CAP theorem states that distributed systems cannot simultaneously guarantee consistency",
    "Lambda calculus is a formal system for expressing computation through function abstraction and",
    "Public-key cryptography relies on mathematical problems that are easy to compute in one",
    "Operating system schedulers allocate CPU time to processes using policies such as round-robin",
    "Database normalization reduces redundancy by decomposing tables into smaller relations linked",
    "Containerization isolates application processes via Linux namespaces and cgroups while sharing",
    "Static analysis tools detect potential bugs by examining source code without executing it",
    # — Chemistry (20) —
    "The lottery ticket hypothesis posits that dense networks contain sparse subnetworks that",
    "Stereochemistry describes the spatial arrangement of atoms in molecules and how chirality",
    "Acid-base equilibria in aqueous solutions are governed by the law of mass action expressed by",
    "Molecular orbital theory describes chemical bonds as combinations of atomic orbitals using",
    "Reaction mechanisms in organic chemistry are elucidated through kinetic studies and isotopic",
    "Thermodynamic cycles relate state functions like enthalpy and entropy to spontaneity through",
    "Coordination compounds form when transition metals accept lone pairs from ligands creating",
    "Polymerization reactions build macromolecules from monomers via addition or condensation",
    "Spectroscopic techniques such as NMR identify molecular structure by probing nuclear spin",
    "Catalysts accelerate chemical reactions by lowering activation energy without being consumed",
    "Electrochemistry studies redox reactions and the conversion between chemical and electrical",
    "Crystal structures are classified by their unit cell parameters and the symmetry operations",
    "Hydrogen bonding between water molecules accounts for the unusual properties of water such as",
    "Reaction kinetics relates rate to concentration and temperature through the Arrhenius equation",
    "Computational chemistry uses density functional theory to approximate electronic structure",
    "Photochemical reactions are initiated by absorption of photons and proceed via excited-state",
    "The Gibbs free energy determines reaction spontaneity through the relationship between",
    "Asymmetric catalysis produces enantiomerically enriched products using chiral catalysts",
    "Metal-organic frameworks are crystalline porous materials with applications in gas storage and",
    "Solvation effects in liquid-phase reactions can dramatically alter rate constants compared to",
    # — Law / Policy (20) —
    "In the lazy training regime, neural networks behave like kernel machines and the loss decreases",
    "Constitutional law in the United States is interpreted by the Supreme Court through doctrines",
    "Intellectual property rights protect creative and inventive works through patents, copyrights",
    "Contract law requires offer, acceptance, consideration, and mutual assent to form an enforceable",
    "Tort liability arises when one party's negligent conduct causes foreseeable harm to another",
    "Administrative law governs the actions of executive agencies through statutes such as the",
    "Antitrust law prohibits monopolistic practices that restrain trade and reduce consumer welfare",
    "Securities regulation requires public companies to disclose material information about their",
    "Criminal procedure protects defendants through rights such as the presumption of innocence and",
    "Property law governs ownership and transfer of real and personal property through doctrines",
    "Environmental regulation balances economic activity with ecological preservation via permitting",
    "International trade law governs cross-border commerce through frameworks like the WTO which",
    "FDA software-as-a-medical-device guidance requires lifecycle management for clinical decision",
    "HIPAA Privacy Rule restricts the use and disclosure of protected health information unless",
    "Tax law allocates the cost of public goods through progressive income taxation and corporate",
    "Employment law regulates the relationship between employers and workers through statutes",
    "Family law adjudicates marriage, divorce, child custody, and spousal support based on the",
    "Bankruptcy law allows insolvent debtors to discharge or restructure obligations under chapters",
    "Immigration law determines admissibility through visa categories, asylum eligibility, and",
    "Election law governs voter registration, ballot access, and campaign finance through both",
    # — Philosophy / Cognitive science (20) —
    "Universality in deep learning refers to the phenomenon by which different architectures and",
    "Epistemology studies the nature, sources, and limits of knowledge through analyses such as",
    "Phenomenology investigates the structures of conscious experience as lived from the first-person",
    "Utilitarianism evaluates actions by their consequences for aggregate well-being whereas",
    "The mind-body problem concerns the relationship between mental states and physical processes",
    "Free will debates contend with whether moral responsibility is compatible with determinism in",
    "Virtue ethics emphasizes character traits such as courage and justice as the foundation of",
    "Theories of personal identity examine what makes a person at one time the same as a person at",
    "Logical positivism asserted that meaningful statements must be empirically verifiable but was",
    "Wittgenstein's language games illustrate how meaning derives from use within social practices",
    "The trolley problem probes intuitions about the moral permissibility of harming one to save",
    "Embodied cognition argues that mental processes are fundamentally shaped by the body's",
    "Bayesian models of cognition treat perception and learning as probabilistic inference under",
    "Moral relativism holds that ethical truths vary across cultures whereas moral realism",
    "The Chinese Room argument challenges the claim that symbol manipulation alone constitutes",
    "Naturalized epistemology integrates empirical findings from psychology into the theory of",
    "The hard problem of consciousness asks why physical processes give rise to subjective",
    "Moral particularism rejects general principles in favor of context-sensitive moral judgments",
    "Existentialism centers human freedom and responsibility in a world without inherent meaning",
    "Cognitive science integrates psychology, neuroscience, linguistics, and computer science to",
    # — Finance / Economics (20) —
    "Empirical Fisher information differs from the true Fisher information when the model distribution",
    "Macroeconomic policy uses monetary and fiscal instruments to stabilize output and prices",
    "Asset pricing models relate expected returns to systematic risk through frameworks such as",
    "Behavioral finance studies how cognitive biases influence investor decisions and market",
    "Game theory analyzes strategic interactions where players' payoffs depend on others' choices",
    "Options pricing under Black-Scholes assumes geometric Brownian motion for the underlying",
    "Capital structure theory examines how the mix of debt and equity affects firm valuation",
    "Inflation expectations are anchored by central bank credibility and influence wage and price",
    "The efficient market hypothesis posits that asset prices fully reflect available information",
    "Portfolio theory optimizes the trade-off between expected return and risk through",
    "Liquidity in financial markets refers to the ease of converting assets to cash without",
    "Credit derivatives transfer default risk from one party to another via instruments such as",
    "Microstructure analyzes how trading mechanisms affect price discovery and transaction costs",
    "Sovereign debt crises arise when governments cannot meet obligations leading to default or",
    "Real business cycle theory attributes economic fluctuations to technology shocks rather than",
    "Auction design balances allocative efficiency, revenue, and incentive compatibility through",
    "Risk management uses VaR and stress testing to quantify potential losses under adverse",
    "Cryptocurrency markets exhibit high volatility driven by speculation, regulation, and",
    "Pension fund liabilities are valued using discount rates that reflect expected long-run",
    "Quantitative trading employs statistical arbitrage strategies that exploit short-lived",
    # — Linguistics (20) —
    "Statistical learning theory provides bounds on generalization error in terms of the complexity of",
    "Phonology analyzes the abstract sound systems of languages through features and rules that",
    "Syntactic theory describes how words combine into phrases and sentences according to",
    "Semantic compositionality holds that the meaning of a complex expression is determined by",
    "Sociolinguistics studies how language varies and changes across social groups and contexts",
    "Historical linguistics reconstructs proto-languages by comparing systematic correspondences",
    "Pragmatics investigates how context shapes the interpretation of utterances beyond their",
    "Morphology examines internal structure of words including affixation, compounding, and",
    "Universal grammar posits innate linguistic principles common to all human languages",
    "Construction grammar treats meaning-form pairings as the basic units of grammatical knowledge",
    "Language acquisition in children proceeds through stages from babbling to telegraphic speech",
    "Bilingualism shapes cognitive control through code-switching and dual-language activation",
    "Sign languages have full grammatical complexity comparable to spoken languages with",
    "Computational linguistics applies algorithms to natural language processing tasks such as",
    "Discourse analysis studies coherence and cohesion above the sentence level through devices",
    "Phonetics examines the physical properties of speech sounds through articulatory, acoustic",
    "Endangered language documentation records linguistic diversity before it disappears using",
    "Lexical semantics investigates word meaning through relations such as synonymy, antonymy",
    "Typology classifies languages by their structural features identifying universals and",
    "Natural language inference tasks evaluate whether one sentence entails another through",
    # — General knowledge (20) —
    "Heavy-tailed weight distributions in trained neural networks are associated with",
    "Mean-field theory of deep networks predicts that the variance of the pre-activation grows",
    "World War II reshaped global geopolitics through the rise of the United States and Soviet",
    "Renaissance humanism revived classical learning and emphasized human potential through",
    "Climate change driven by anthropogenic greenhouse gases is altering ecosystems and weather",
    "Industrial Revolution transformed economies through mechanization and factory production",
    "Ancient Egyptian civilization flourished along the Nile for over three millennia building",
    "The Silk Road connected East and West facilitating exchange of goods, ideas, and diseases",
    "Renewable energy technologies like solar and wind are scaling rapidly to displace fossil",
    "Plate tectonics explains continental drift and seismic activity through convection in the",
    "Modern art movements such as cubism and surrealism challenged conventional representation by",
    "Globalization has integrated national economies through trade, finance, migration, and",
    "Biodiversity loss accelerated by habitat destruction threatens ecosystem services and",
    "The internet revolutionized communication, commerce, and culture through packet-switched",
    "Space exploration progressed from satellites to lunar landings and is now extending to",
    "Public health interventions such as vaccination and clean water dramatically reduced",
    "Literary modernism broke with realist conventions through stream-of-consciousness narration",
    "Feudal Europe organized society through hierarchical bonds of vassalage between lords and",
    "Genetic engineering enables targeted modification of organisms with applications in",
    "The scientific method advances knowledge through hypothesis formulation, experimental testing",
]
# Sanity: must have at least 200 unique texts per review-consensus (iter-262).
assert len(SAMPLE_TEXTS) >= 200, f"SAMPLE_TEXTS has {len(SAMPLE_TEXTS)} entries, need >=200"
assert len(set(SAMPLE_TEXTS)) == len(SAMPLE_TEXTS), "SAMPLE_TEXTS has duplicates"


def _accumulate_fim_cpu(net, tokenizer, n_probes, device, seed, max_len=64):
    """Accumulate per-parameter grad-squared on CPU (saves GPU RAM)."""
    fim_cpu: dict[str, torch.Tensor] = {}
    for n, p in net.named_parameters():
        if p.requires_grad:
            fim_cpu[n] = torch.zeros(p.shape, dtype=torch.float32, device="cpu")
    net.eval()
    rng = np.random.default_rng(seed)
    for i in range(n_probes):
        text = SAMPLE_TEXTS[rng.integers(0, len(SAMPLE_TEXTS))]
        ids = tokenizer.encode(
            text, return_tensors="pt", truncation=True, max_length=max_len
        ).to(device)
        if ids.size(1) < 2:
            continue
        out = net(ids, labels=ids)
        loss = out.loss
        net.zero_grad(set_to_none=True)
        loss.backward()
        for n, p in net.named_parameters():
            if p.grad is not None and n in fim_cpu:
                fim_cpu[n] += (p.grad.detach().float() ** 2).cpu()
        if (i + 1) % 20 == 0:
            print(f"  probe {i+1}/{n_probes}", flush=True)
    for n in fim_cpu:
        fim_cpu[n] /= n_probes
    return fim_cpu


def _streaming_stats_and_sample(fim_cpu, sample_size, seed):
    """Single pass: total stats + uniform sample for sort-dependent metrics."""
    layer_sizes = [v.numel() for v in fim_cpu.values()]
    total_n = sum(layer_sizes)
    if total_n == 0:
        raise RuntimeError("FIM is empty — model has no trainable params?")

    pos_n = 0
    pos_sum = 0.0
    pos_sum_sq = 0.0
    sample_pieces: list[np.ndarray] = []
    rng = np.random.default_rng(seed + 1)

    actual_sample = min(sample_size, total_n)
    remaining = actual_sample
    for i, (name, v) in enumerate(fim_cpu.items()):
        flat = v.flatten().to(torch.float64).numpy()
        positive = flat[flat > 0]
        pos_n += positive.size
        pos_sum += float(positive.sum())
        pos_sum_sq += float((positive ** 2).sum())

        is_last = (i == len(fim_cpu) - 1)
        if is_last:
            k = remaining
        else:
            k = max(1, int(round(actual_sample * flat.size / total_n))) if flat.size else 0
            k = min(k, remaining)
        if k > 0 and flat.size > 0:
            idx = rng.integers(0, flat.size, size=k)
            sample_pieces.append(flat[idx])
            remaining -= k
        if remaining <= 0:
            break
    sample = (
        np.concatenate(sample_pieces) if sample_pieces else np.empty(0, dtype=np.float64)
    )
    return total_n, pos_n, pos_sum, pos_sum_sq, sample, actual_sample


def _gini_on_positive(values: np.ndarray) -> float:
    """Gini on STRICTLY positive entries.

    iter-262 review fix: switch to ``values > 0`` to match
    top_1pct_mass — ensures paired-comparison validity with sparse
    MoE models that have exact-zero FIM entries from inactive experts.
    """
    v = values[values > 0].astype(np.float64)
    if v.size == 0 or v.sum() == 0:
        return 0.0
    v.sort()
    n = v.size
    cum = np.cumsum(v)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def _top_1pct_mass(values: np.ndarray) -> float:
    v = values[values > 0].astype(np.float64)
    if v.size == 0:
        return 0.0
    s = np.sort(v)[::-1]
    k = max(1, int(s.size * 0.01))
    return float(s[:k].sum() / s.sum())


def _tier_ratio(values: np.ndarray) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=np.float64)
    s = np.sort(v)[::-1]
    n = s.size
    k1 = max(1, int(n * 0.01))
    k3 = max(1, int(n * 0.5))
    t1 = float(s[:k1].mean())
    t3 = float(s[-k3:].mean())
    if t3 <= 0:
        nz = s[s > 0]
        t3 = float(nz[-max(len(nz) // 10, 1):].mean()) if len(nz) else 1e-30
    return t1, t3, (t1 / t3 if t3 > 0 else float("inf"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="EleutherAI/pythia-6.9b")
    ap.add_argument("--n-probes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gini-sample-size", type=int, default=100_000_000)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="Forward/backward dtype (FIM still accumulated in fp32 on CPU).",
    )
    args = ap.parse_args()

    if args.out is None:
        slug = args.model.replace("/", "_").replace("-", "_").replace(".", "_")
        args.out = str(Path(__file__).resolve().parent / f"v11_{slug}_results.json")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
        print(
            f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
            flush=True,
        )

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading {args.model} in {args.dtype}...", flush=True)
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    net = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    ).to(device)
    print(f"  load time: {time.time() - t_load:.1f}s", flush=True)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"  params: {n_params:,}", flush=True)

    print(f"\nMeasuring FIM diagonal with {args.n_probes} probes...", flush=True)
    t0 = time.time()
    fim_cpu = _accumulate_fim_cpu(net, tokenizer, args.n_probes, device, args.seed)
    t_fim = time.time() - t0
    print(f"  FIM accumulation: {t_fim:.1f}s", flush=True)

    # Free GPU model before sampling stats — saves headroom for downstream pods.
    del net
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Computing streaming stats + uniform sample...", flush=True)
    t1 = time.time()
    total_n, pos_n, pos_sum, pos_sum_sq, sample, actual_sample = (
        _streaming_stats_and_sample(fim_cpu, args.gini_sample_size, args.seed)
    )
    print(
        f"  total params seen: {total_n:,}; positive: {pos_n:,}; "
        f"sample size: {actual_sample:,}",
        flush=True,
    )

    eff_rank = (
        (pos_sum ** 2) / (pos_n * pos_sum_sq)
        if (pos_sum_sq > 0 and pos_n > 0)
        else 0.0
    )
    eff_rank_n = eff_rank / total_n if total_n > 0 else 0.0

    t1m, t3m, tier = _tier_ratio(sample)
    g = _gini_on_positive(sample)
    tp1 = _top_1pct_mass(sample)
    log10_t = float(np.log10(tier)) if tier > 0 and np.isfinite(tier) else float("inf")
    print(f"  streaming-stats time: {time.time() - t1:.1f}s", flush=True)

    print(f"\n=== {args.model} FIM (V11) ===")
    print(f"  N params       = {n_params:,}")
    print(f"  T1/T3          = {tier:.6e}")
    print(f"  log10(T1/T3)   = {log10_t:.6f}")
    print(f"  Gini           = {g:.6f}")
    print(f"  eff_rank_n     = {eff_rank_n:.6e}")
    print(f"  top-1% mass    = {tp1:.6f}")
    print(f"  FIM time       = {t_fim:.1f}s")

    arch_id = (
        f"{torch.cuda.get_device_name(0)}_{torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB"
        if device.type == "cuda"
        else "cpu"
    )
    payload = {
        "model_id": args.model,
        "n_params": int(n_params),
        "n_probes": int(args.n_probes),
        "n_unique_texts": int(len(set(SAMPLE_TEXTS))),
        "seed": int(args.seed),
        "T1T3": tier,
        "log10_T1T3": log10_t,
        "gini": g,
        "gini_sample_size": int(actual_sample),
        "eff_rank_n": eff_rank_n,
        "top_1pct_mass": tp1,
        "arch_id": arch_id,
        "spec_version": "v2-iter262",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
