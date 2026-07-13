# exp/ — LoRA-writes-into-the-workspace experiment

Experiment code layered on top of the unmodified `jlens` package. Hypothesis:
LoRA's B-matrix write directions are confined to sparse combinations of J-lens
concept directions, because B starts at 0 and accumulates gradients pulled back
through the same Jacobian that defines the lens. Predicts LoRA ≈ full FT for
steering (IFT) but not skill acquisition (CPT).

Phases (each gates the next; full plan in the session plan file):

1. **δ screen** (`exp/delta/` — built): directional scatter of per-example
   pullback vectors `dL/dh_l`, steering vs skill data. A null here falsifies
   the hypothesis before any expensive work.
2. **Lens fit** (`exp/lens_fit/` — built; run only if δ splits): `run_fit.py`
   fits the lens on WikiText (GPU, ~3-5h, resumable) with built-in acceptance
   checks; `export_unembed.py` saves `unembed.pt` so later phases never need
   the model. `scp lens.pt unembed.pt` off the pod (~1.3 GB total).
3. **Decompose** (`exp/decompose/` — built; runs anywhere on CPU once
   `lens.pt` + `unembed.pt` exist): SVD each adapter's ΔW per (layer, module)
   — gauge-invariant write directions — and score them against the lens
   dictionary (signed OMP primary; non-negative and top-k J-subspace
   projection as checks) with a random floor and a wrong-layer null.
   Residual-frame modules only (o_proj/down_proj — measured at ~25% of ΔW
   energy on the real magicoder adapter; reported as `residual_coverage`).
   `run_decompose.py` then `analyze.py` → `alignment_vs_depth.png` +
   decoded-atoms report.
3b. **Positive control** (`exp/decompose/control.py` + `run_control.py`):
   decompose the top eigendirections of the Phase-1 pullback second moments
   through the *identical* dictionary/OMP path. These directions are the
   Lemma-2 span the theory confines B to, so: control high + adapters at
   floor → the Phase-3 null is real (adapters left the J-frame, e.g. AdamW);
   control at floor too → the lens dictionary lacks resolving power and the
   Phase-3 null is uninformative. Needs `out/delta/*.pt` + `lens.pt` +
   `unembed.pt`; CPU.

4. **Causal ablation**: zero the J-space write component, check reversion
   (not yet built).

## Running the δ screen (on a GPU pod)

```bash
export HF_TOKEN=hf_...          # account with bigcode/starcoderdata accepted
bash exp/setup_runpod.sh        # installs, CPU tests, 1-prompt GPU smoke

# ~1-2 h on an A100:
python -m exp.delta.run_extract --n-examples 2000 --out out/delta \
    --conditions code-ift math-ift code-cpt math-cpt wikitext \
                 code-ift-alltok math-ift-alltok

python -m exp.delta.analyze --dir out/delta   # figure + summary.json + table
```

Decision gate: IFT participation-ratio curves clearly below CPT curves, gap
peaking mid-network, robust to the `*-alltok` regime controls → proceed to
Phase 2. Otherwise: stop, write up the null.

Conventions shared with `jlens` (and pinned by `exp/tests/`): pullbacks live at
**block outputs** (post-residual-add, pre-final-norm), source positions use
`valid_position_mask` (skip first 16, drop final), and
`test_pullback_frame_parity_with_jlens_estimator` proves the extraction path
reproduces `jacobian_for_prompt` rows exactly under the lens's own cotangent.

CPU tests: `python -m pytest exp/tests/ -q` (runs against `tests/tiny.py`,
no network, no GPU).
