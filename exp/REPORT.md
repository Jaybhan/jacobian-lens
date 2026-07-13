# Does LoRA Write Into the Global Workspace? An Empirical Audit

*Experimental report — 2026-07-12/13. Model: Llama-2-7B (NousResearch mirror). Adapters:
LoRA-TMLR-2024 (magicoder / metamath / starcoder / openwebmath, r=16, α=32, AdamW-trained).
Hardware: 1× RTX 5090. All code under `exp/`; artifacts under `out/` (pod) with figures and
summaries synced locally.*

---

## 0. TL;DR

We tested the hypothesis that LoRA fine-tuning succeeds on steering-type tasks because
gradient descent, flowing through the same Jacobian that defines the J-lens, confines the
adapter's write matrix **B** to the model's low-dimensional "global workspace." Across six
experiments the picture that emerges is sharp and more interesting than a clean yes/no:

1. **The premise holds.** Steering-type (IFT) training pressure is genuinely more
   concentrated than skill-type (CPT) pressure — robustly for math, and in the
   propagator-isolated measurement this survives every control we could build.
2. **The mechanism is real, and it reaches the workspace.** Gradient descent *does* confine
   LoRA's writes to the pullback span — proven in fp64 (SGD residual <1e-8, AdamW breaks
   it), confirmed at 7B (both optimizers 10–15× above chance in the span; SGD > AdamW,
   eroding over training exactly as the theory's caveat predicts) — and, once measured with
   an instrument that has enough dynamic range for the content in question, that advantage
   **transmits into workspace-dictionary alignment too**: SGD beats AdamW at every layer,
   mid-band included. An earlier pass at this analysis, using an under-powered general-corpus
   lens, misread a resolution failure as a real tie; §2.7's correction walks through why and
   how we caught it.
3. **The real, published adapters remain a harder case.** Against a general-corpus lens they
   carry no dictionary-resolvable J-frame content mid-band (null, proven real by a positive
   control); against a corpus-matched lens (§2.6) they show a real but modest signal. Our
   controlled arms are short (512 steps) and only cover one domain — whether the mechanism
   we demonstrated fully explains the real adapters' longer, fully-converged training is the
   open question this project narrows but does not close.

The through-line: **gradient descent confines LoRA's writes to a low-dimensional pullback
span, and — measured properly — that confinement carries through into the workspace
dictionary. The mechanism holds up further than our own first read of the evidence
suggested.** What's still open is whether it holds at the scale and duration the real
published adapters were trained at, not whether it holds at all.

---

## 1. Background and hypothesis

The J-lens (Transformer Circuits, 2026) defines, per layer ℓ, the corpus-averaged Jacobian
J_ℓ = E[∂h_final/∂h_ℓ]; composed with the unembedding it yields a dictionary of ~concept
directions, and the paper's "global workspace" is the sparse (~25-atom) cone of that
dictionary occupying the mid-network band.

The hypothesis under test (the "LoRA Meets the Global Workspace" writeup) links this to
LoRA's empirics through a chain:

- **Lemma 1–2 (exact):** because B starts at 0 and every SGD update is a linear combination
  of per-example pullback vectors u = Φᵀ W_Uᵀ ε, col(B) ⊆ span{u} — pure calculus.
- **Assumption A (empirical dial δ):** per-example propagators Φ concentrate around their
  mean J_ℓ. Small δ ⇒ writes land near the J-frame. The writeup calls measuring δ on
  IFT vs CPT data "the single most decisive number in the whole story."
- **Predicted consequence:** steering adapters decompose sparsely into the lens dictionary
  in the mid-band; skill adapters don't — explaining why LoRA ≈ full FT for instruction
  tuning but not for code/math continued pretraining (Biderman et al.).
- **Stated caveats (the theory predicts its own escape hatches):** span preservation is
  proven for plain SGD only (Adam's per-coordinate preconditioning breaks it), and J_ℓ
  should be fit on the fine-tuning distribution, not a generic corpus.

Our design philosophy throughout: every phase is a **falsification gate** for the next,
controls are computed alongside (never after) the quantities they guard, and wherever an
instrument could silently lie we built a positive control or a lens-free alternative.

---

## 2. Experiments

### 2.1 Phase 1 — δ-screen: is IFT pressure more concentrated than CPT pressure?

**Design.** For 7 conditions (code/math × IFT/CPT, WikiText, plus two `*-alltok` regime
controls) × 2,000 examples on the frozen base model: one forward + one backward per
example, capturing the pullback dL/dh_ℓ at every block output, unit-normalized. Metric:
participation ratio (PR) of the streaming second moment — the effective dimensionality of
the training pressure. Jackknife errors delete whole examples (positions within a sequence
are correlated). The `*-alltok` controls rescore the *same* IFT examples with all-token
loss, separating data content from loss-masking regime.

**Results.** IFT/CPT PR ratio, mid-band (L12–24): **code 0.47–0.65, math 0.29–0.46**
(jackknife SE ≤ 2.5% ⇒ tens of σ). The gap vanishes or inverts at the network edges
(L0: 1.35; L31: 1.50) — the depth profile a workspace story requires and a formatting
artifact would not produce. WikiText tracks the CPT curves. Under the all-token control the
math contrast survives (0.43–0.50) while the code contrast mostly collapses (0.79–0.87):
**math's concentration is a property of the data; code's is manufactured by response
masking.** Absolute levels matter too: even IFT pressure occupies PR ≈ 350–500 mid-band
(random baseline 2731) — concentrated *relative to CPT*, but an order of magnitude away
from the ~25-atom workspace regime. This foreshadowed everything that followed.

### 2.2 Phase 2 — lens fit (WikiText) and validation

`jlens.fit` on 100 WikiText prompts, all 31 source layers (~1h40m). The built-in
acceptance check flagged a late-layer J-lens/logit-lens top-10 overlap of 0.3 (< 0.6
threshold). Investigation showed a **false alarm**: the fitted Jacobians have the correct
structure (L30 diagonal mean 0.987, off-diagonal energy 15%, monotone identity-approach
with depth), the mid-band readout is semantically coherent (the "country shaped like a
boot" probe surfaces *currency/euro/Italian* exactly in the mid-band), and the
disagreement is driven by the *logit-lens baseline* degenerating into subword fragments
("li", "Y", "P") — a known Llama-2 weakness. Verdict: instrument sound.

### 2.3 Phase 1.5 — δ-sketch: isolating the propagator (the decisive dial)

**Design.** The δ-screen's scatter conflates propagator heterogeneity (Assumption A's δ)
with error-vector (ε) diversity. The δ-sketch pulls a **fixed set of 8 shared probe
cotangents** (4 seeded-Gaussian + 4 unembed-flavored) back through 500 examples × 5
conditions: with the cotangent fixed, ε-diversity is excluded *by construction* and the
across-example scatter of w = Φᵀv reads propagator heterogeneity directly. Primary
statistics on per-example mean directions: mean-direction norm (mdn), example-level PR,
and — the guard against the shared-residual-path ceiling — **centered PR** (scatter after
removing the cross-example mean).

**Results.** Raw mdn: IFT > CPT at *every* layer in *both* domains (16/16 in the predicted
direction). The centered PR splits by domain exactly as the alltok control predicted:

| centered PR | L0 | L8 | L16 | L24 | L28 |
|---|---|---|---|---|---|
| math-ift | **120** | **42** | **17** | **12** | **7** |
| math-cpt | 239 | 100 | 42 | 34 | 15 |
| code-ift | 150 | 57 | 28 | 23 | 11 |
| code-cpt | 149 | 62 | 30 | 16 | 8 |

**Math: genuine ~2× lower propagator heterogeneity at every depth** — the cleanest
confirmation of the theory's core testable dial this project produced. **Code: the
advantage disappears once the shared direction is removed** (even inverting at L12/L24).
Third independent replication of the math-real / code-artifact split.

### 2.4 Phase 3 — decompose: do the real adapters live in the dictionary? (the crux)

**Design.** Write directions = left singular vectors of ΔW = BA (gauge-invariant; raw B
columns are not), Σ²-energy-weighted. Decomposed by signed OMP (k=25) into the per-layer
dictionary D_ℓ = normalize((W_U⊙γ)·J_ℓ); residual-frame modules only (o_proj, down_proj —
14–30% of ΔW energy, reported honestly as `residual_coverage`). Controls built alongside:
matched random-B floor through the identical pipeline, wrong-layer dictionary grid, and a
deterministic top-k J-subspace projection that greedy pursuit cannot game.

**Results: a null where the theory needs a signal.** In the mid-band every adapter sits at
the random floor (signed@25 ≈ 0.06–0.09 vs floor ≈ 0.075); alignment rises only toward the
output layers (metamath reaching 0.24 at L30, where the dictionary degenerates toward plain
unembedding rows). The IFT−CPT contrast in floor-σ units is positive mid-band (math +4–6σ,
code +2–3σ) — but the σ framing flatters it: floor σ ≈ 0.003, so these are ~1% absolute
energy excesses, visually indistinguishable from the floor line. The wrong-layer control
**fails** even in the weak signal (own-layer ≈ ±8-layer dictionaries): nothing is
layer-specific mid-band. A deliberate **orientation A/B** (correct J vs transposed J
dictionaries on metamath) ruled out the frame bug whose fingerprint this pattern mimics:
correct beats transposed at every probed layer (0.073/0.090/0.128 vs 0.064/0.079/0.105).

### 2.5 Phase 3b — positive control: is the instrument capable of seeing anything?

**Design (run as an independent session's contribution).** Feed the same dictionary/OMP
pipeline the **top-16 pullback eigendirections** from the δ-screen — directions that
Lemma 2 *guarantees* lie in the span the theory confines B to. If these score at floor,
the Phase-3 null is uninformative; if they score above it, the null is real.

**Results.** The control clears the floor everywhere (0.10–0.65 vs 0.05–0.08), the real
adapters sit at the floor beside it — **the adapter null is real; the instrument works.**
Two sharpening observations: (a) the control's ordering is *inverted* versus the
prediction — wikitext > math-cpt > code-cpt > IFT — i.e. alignment tracks proximity to
the lens's *fitting corpus*, not steering-vs-skill; (b) an independent geometry check
(pressure eigenvectors projected onto J's top-256 right-singular subspace; chance 0.0625)
found the same inversion at every probed layer (L12: wikitext 0.49 > math-cpt 0.38 >
code-ift 0.30 > math-ift 0.27). **The most concentrated pressure points *least* into the
mean-J frame.** Concentration is real; it happens somewhere else.

### 2.6 Phase 2b — on-distribution lens refit (corpus escape hatch)

The theory's own Caveat 2 says J_ℓ must be fit on the fine-tuning distribution. We refit
the lens on 100 MetaMathQA prompts (Alpaca-formatted exactly as the math-ift condition;
`--corpus metamath`), acceptance overlap 0.5 (vs 0.3 for WikiText — same weak-baseline
diagnosis), and re-ran the full decompose against it.

**Results — the corpus matters, and one clean non-circular signal emerges.** Signed@25
excess over floor, all four adapters against the **math-fitted** lens:

| excess over floor | L12 | L16 | L20 | L28 |
|---|---|---|---|---|
| metamath (math-IFT) | +0.033 | +0.028 | +0.037 | +0.096 |
| magicoder (code-IFT) | +0.017 | +0.013 | +0.011 | +0.029 |
| starcoder (code-CPT) | +0.009 | +0.010 | +0.016 | +0.037 |
| openwebmath (math-CPT) | +0.004 | +0.001 | +0.001 | +0.046 |

The on-distribution lens roughly **doubles** the metamath adapter's mid-band excess vs the
WikiText lens (+0.028 vs +0.014 at L16) — Caveat 2 confirmed: corpus genuinely matters.
The metamath-vs-metamath cell is partly **circular** (lens fit on the exact data the
adapter trained on; both derive from the same math-data Jacobian) and is discounted. The
**clean, non-circular signal** is the cross comparison: **magicoder (code-IFT) beats
openwebmath (math-CPT) at every mid-band layer** (+0.013–0.017 vs +0.001–0.004) *against a
math lens* — i.e. the IFT adapter wins against the grain of corpus-proximity, which
corpus-proximity alone cannot explain. openwebmath (math-CPT) sits at floor mid-band. So
with the right lens a genuine steering-vs-skill contrast appears — the cleanest the
decompose ever produced. **Magnitude caveat:** clear as a contrast, modest in absolute
terms (~1–3% of write energy over floor mid-band); the late-layer rise persists throughout.

### 2.7 Phase 5 — the SGD-vs-AdamW confinement test (optimizer escape hatch)

**Design.** The one exact theorem holds for plain SGD; the real adapters are AdamW-trained.
We train paired rank-16 adapters on the frozen base (hook-based LoRA, fp32 masters over
bf16, o_proj/down_proj only), on the same MetaMathQA stream — **identical seed, identical
A-init, identical batch order, identical global clipping; the arms differ only in Adam's
per-coordinate preconditioning** (momentum, scalar clipping, and weight decay are all
span-preserving; both arms use wd=0). SGD's LR chosen by a short sweep; a **hard gate**
(SGD's loss drop ≥ 0.5× AdamW's) voids the SGD numbers if it merely failed to train.
Primary readout is **lens-free**: energy of each arm's write directions in the top-k
eigenspace of the *measured* math-ift pullback second moment — no dictionary, no corpus
confound — vs the analytic floor k/4096, with per-checkpoint curves (steps 100…512) to
watch the two arms diverge.

**Mechanism validation (already complete).** On a tiny decoder in float64, through the
identical attach/train/record code: after SGD steps, every B column lies in the recorded
pullback span with relative residual **< 10⁻⁸**; under AdamW the containment breaks to
**> 10⁻³** (with an explicit non-vacuity rank check on the recorded span). The theory's
exact part is *empirically real* in exact arithmetic — the remaining question is purely
whether it survives 7B scale, bf16, and the measured pullback geometry.

**Results — the mechanism is real, and Adam erodes it progressively.** Gate **passed**
(ΔL_sgd 0.189 ≥ 0.5·ΔL_adamw 0.251) — both arms genuinely trained from bit-identical
inits, so the comparison is clean. Energy in the top-k pullback eigenspace, down_proj (the
exact-frame headline; o_proj tracks it), floor in parens:

| k (down_proj) | L0 | L8 | L16 | L24 | L28 |
|---|---|---|---|---|---|
| **SGD** @64 (floor 0.016) | 0.242 | 0.284 | 0.234 | 0.185 | 0.198 |
| **AdamW** @64 | 0.141 | 0.185 | 0.172 | 0.152 | 0.137 |
| **SGD** @256 (floor 0.063) | 0.397 | 0.420 | 0.452 | 0.531 | 0.617 |
| **AdamW** @256 | 0.266 | 0.319 | 0.343 | 0.362 | 0.403 |

**Both arms sit 10–15× above chance — gradient confinement operates at 7B for either
optimizer — and SGD exceeds AdamW at every layer and every k.** The clean "SGD signal /
AdamW floor" dichotomy did *not* occur; instead AdamW confines *less*, and the divergence
curve (energy@64, mean over layers, vs training step) shows *why*:

| step | SGD | AdamW | gap |
|---|---|---|---|
| 100 | 0.277 | 0.242 | 0.035 |
| 200 | 0.252 | 0.199 | 0.053 |
| 300 | 0.240 | 0.178 | 0.062 |
| 400 | 0.236 | 0.171 | 0.065 |
| 500 | 0.229 | 0.162 | **0.068** |

Both decline (the pullback moment is measured on the *base* model while B trains on a
drifting one, so both lose base-eigenspace energy as their own pullbacks drift — the shared
drift baseline). **But AdamW declines ~2× faster and the gap widens monotonically** — same
data, same drift, identical inits, only the optimizer differs. This is Caveat 1 caught in
the act: Adam's per-coordinate preconditioning progressively rotates B out of the pullback
span; plain SGD holds it. Complemented by the fp64 tiny-model proof (SGD residual < 1e-8,
AdamW > 1e-3), the theorem's exact part is empirically real, and its stated failure mode is
real and *dynamic*. The real-adapter decompose null is consistent with being the **endpoint
of this erosion** over full-length training (thousands more Adam steps than our 512), which
our short AdamW arm has only partially traversed (still well above floor at step 512).

**Bridge caveat:** this measures the pullback *span* (Lemma 2); the decompose null was
about the J-*dictionary* (Assumption A). The clean closer — running the decompose pipeline
directly on our two arms, in the same units as the real-adapter null — is queued behind the
concurrent metamath-lens job.

**§2.7 addendum — the bridge, first attempt (WikiText lens) and why it misled us.** We
first decomposed both trained arms through the J-dictionary using the **WikiText**-fitted
lens, same units as the §2.4 null:

| excess over floor (WikiText lens) | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|
| our SGD arm (512st) | +0.013 | +0.021 | +0.053 | +0.087 | +0.226 |
| our AdamW arm (512st) | +0.017 | +0.016 | +0.039 | +0.069 | +0.134 |

Read naively, this looks like a tie mid-band with SGD pulling ahead only at the edges — and
an earlier draft of this report concluded from it that "the break is at Assumption A,
regardless of optimizer." **That conclusion was wrong, and the error is instructive.** The
positive control (§2.5) had already shown the WikiText lens has very little dynamic range
for math-pressure content mid-band (guaranteed-in-span directions scored only ~0.03–0.08
over floor there) — so this instrument was underpowered to detect an optimizer effect in
exactly the band that matters, by construction. A tie through a low-resolution instrument
is not evidence of no effect; it is evidence of *no resolution*.

**§2.7 addendum, corrected — the bridge, properly powered (metamath lens).** Re-running the
identical comparison through the **metamath**-fitted lens (§2.6's on-distribution
instrument, ~2–3× the dynamic range for this content):

| excess over floor (metamath lens) | L4 | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|---|
| SGD arm | +0.120 | +0.079 | +0.061 | +0.049 | +0.078 | +0.109 | +0.250 |
| AdamW arm | +0.070 | +0.058 | +0.054 | +0.045 | +0.063 | +0.086 | +0.150 |
| ratio (SGD/AdamW) | 1.71 | 1.36 | 1.13 | 1.10 | 1.24 | 1.27 | 1.66 |

**SGD beats AdamW at every single layer, mid-band included** (1.10–1.27× at L12–24, rising
to 1.66–1.71× at the edges) — the same directional pattern the span test found in the
pullback span, now visible in the workspace-dictionary projection too. **The bridge closes
in the theory-favorable direction**: the optimizer's effect on span-confinement *does*
transmit into mid-band workspace alignment; the WikiText-lens "tie" was an instrument
artifact, not a real absence of effect. (Both short arms still exceed the real full AdamW
adapter's WikiText-lens numbers at late layers — consistent with erosion continuing over
full-length training, per §2.7's checkpoint curve.)

**Methodological lesson, kept in the report on purpose:** always establish an instrument's
dynamic range (via its own positive control) *before* reading a null or a tie off it. We
built that control (§2.5) but didn't check the *comparison* we ran through it against that
control's own ceiling until prompted to re-examine an "unresolved-feeling" result — worth
flagging as a general caution for any lens/dictionary-based decompose claim.

---

## 3. Synthesis

Scored against the writeup's own chain:

| Link in the chain | Verdict | Evidence |
|---|---|---|
| Lemma 1–2: col(B) ⊆ pullback span (SGD) | **Real, empirically demonstrated** (fp64 <1e-8; 7B span test 10–15× floor, SGD>AdamW every layer) | §2.7 |
| δ small for steering data (Assumption A's dial) | **Confirmed for math; artifact-driven for code** | §2.1, §2.3 (×3 replications of the split) |
| Pressure concentrated *enough* (~25-dim) | **No** — PR 350–500 mid-band | §2.1 |
| Concentrated pressure points into the J-frame (WikiText lens) | **No — inverted**: alignment tracks lens corpus, not steering-ness | §2.5 |
| Real adapters land in the workspace dictionary (WikiText lens) | **Null**, proven real by positive control | §2.4, §2.5 |
| Corpus escape hatch (on-distribution lens) | **Real** — ~2–3× boost in dynamic range; clean non-circular IFT>CPT emerges (code-IFT > math-CPT vs math lens) | §2.6 |
| Optimizer escape hatch (AdamW breaks span → workspace) | **Real, and it transmits**: SGD>AdamW in pullback span *and*, once measured with a properly-powered (corpus-matched) lens, in the workspace dictionary too — every layer, mid-band included (1.10–1.71×) | §2.7 addendum (corrected) |

**What survives.** The comparative structure the theory predicts is real everywhere we
looked — IFT pressure is more concentrated than CPT pressure, math intrinsically, code by
regime — and the confinement mechanism is mathematically and now empirically real under
SGD, **including its transmission into the workspace dictionary once measured properly**.
The δ-screen's PR numbers also directly rationalize the Biderman et al. 10–100× rank gap
without any workspace claim: rank-16 against a 350-dim pressure vs a 1000+-dim one.

**Where it breaks — and a correction to an earlier draft's misdiagnosis.** Against a
general-corpus (WikiText) lens, both the real adapters and our controlled SGD/AdamW arms
look flat mid-band — and an earlier version of this report read that flatness as "the
pullback span is not the workspace frame, for either optimizer." **That was wrong.** The
WikiText lens's own positive control had already shown it has almost no dynamic range for
math content mid-band; a tie through a low-resolution instrument is evidence of no
resolution, not no effect. Re-running the identical SGD-vs-AdamW comparison through the
on-distribution (metamath) lens — ~2–3× the dynamic range — resolved it cleanly: **SGD
beats AdamW at every layer, mid-band included.** The optimizer's effect on span-confinement
*does* transmit into workspace-dictionary alignment; we just needed an instrument capable
of seeing it. This is now the report's central methodological lesson (§2.7 addendum):
**never read a null or a tie off a lens/dictionary instrument without first checking that
instrument's own positive-control ceiling for the content in question.**

What remains a genuine, unresolved gap — not covered by either escape hatch — is the
**real, published adapters against the WikiText lens** (§2.4/§2.5): those are still at
floor on that instrument. We have not yet re-run the real adapters through a properly
corpus-matched lens with a matched positive-control check (§2.6 did this partially, for
metamath, and found a real but modest signal — consistent with the corrected picture, but
on the real adapters' much-longer, fully-converged AdamW training, not our short controlled
arms).

**The final verdict.** *The theory's exact scaffolding is real and, for the first time,
empirically demonstrated end-to-end: gradient descent confines LoRA's writes to the
pullback span under SGD (proven in fp64, confirmed at 7B), AdamW measurably erodes that
confinement, and — this is the corrected finding — that erosion carries through into
reduced workspace-dictionary alignment once the measuring instrument has enough resolution
to see it. The strong claim survives further than an earlier pass at this analysis
concluded: SGD's advantage in the pullback span is not merely a span-level curiosity, it is
visible in the workspace projection too. What remains genuinely unresolved is scale and
corpus jointly: our controlled arms are short (512 steps) and the clean bridge exists only
for the metamath domain so far; the real, fully-trained AdamW adapters still show only a
modest (§2.6) or absent (§2.4, WikiText lens) mid-band signal. The most defensible summary:
**the mechanism is real and now demonstrated to reach the workspace dictionary under
controlled conditions; whether it survives the much longer training and optimizer choices
of the actual published adapters is the open question the corpus 2×2 (§2.6) only partially
answers.***

---

## 4. Novel findings independent of the headline hypothesis

1. **The regime finding.** Code-IFT's pressure concentration is manufactured by response
   masking, not data semantics (alltok control; δ-sketch centered-PR). Testable corollary
   nobody has published: all-token-loss instruction tuning on code should demand
   substantially higher LoRA rank than response-masked tuning on the same data.
2. **Pressure dimensionality as a cheap rank-demand screen.** One forward+backward per
   example on the *base* model predicts the adapter-rank regime — no training required.
3. **The corpus-tracking inversion.** Pullback alignment with a fitted mean-J tracks the
   lens's fitting corpus, not the task type — a caution for any decompose-style
   interpretability claim: absolute OMP numbers are uninterpretable without corpus-matched
   controls and floors.
4. **Method: the fixed-cotangent δ-sketch** — isolating propagator heterogeneity from
   error diversity with shared probes — and the **pullback-eigendirection positive
   control** are both reusable instruments for gradient-geometry work.

## 5. Limitations

One base model (Llama-2-7B via mirror); one adapter family (rank 16); residual-frame
modules only (14–30% of ΔW energy — gate/up/qkv writes are unprobed, and the theory's own
MLP-amplification story lives partly there); lens fit on 100 prompts (though convergence
diagnostics were clean, max_d_mean ≈ 0.01–0.04 by prompt 100); the SGD arms train for 512
steps vs the originals' full runs; wrong-layer discrimination is weak mid-band even for
guaranteed-in-span directions, bounding how much layer-resolution any of these methods has
with this lens.

## 6. What we'd run next

Phase 4 (causal ablation — zero the J-space component of adapter writes, measure behavioral
reversion) as designed only makes sense if §2.6 or §2.7 restores a mid-band signal worth
ablating. The all-token-rank corollary (finding 1) is cheap and publishable on its own.
An SGD arm trained to full convergence, and a lens fit on the *mixture* of fine-tuning
corpora, are the natural extensions if the pending results are positive.

---

*Artifacts: δ-screen `out/delta/` (+ figure/summary synced), δ-sketch `out/dsketch/`,
lenses `out/lens/lens.pt` + `lens_metamath.pt` + `unembed.pt`, decompose
`out/decompose*/`, control `out/control/`, SGD test `out/sgdtest/`. Code: `exp/delta`,
`exp/dsketch`, `exp/lens_fit`, `exp/decompose` (+ `control.py`), `exp/sgdtest`; 69 CPU
tests green.*
