# exp/ — LoRA-writes-into-the-workspace experiment

Experiment code layered on top of the unmodified `jlens` package. Hypothesis:
LoRA's B-matrix write directions are confined to sparse combinations of J-lens
concept directions, because B starts at 0 and accumulates gradients pulled back
through the same Jacobian that defines the lens. Predicts LoRA ≈ full FT for
steering (IFT) but not skill acquisition (CPT).

Phases (each gates the next; full plan in the session plan file):

1. **δ screen** (`exp/delta/`, this directory — built): directional scatter of
   per-example pullback vectors `dL/dh_l`, steering vs skill data. A null here
   falsifies the hypothesis before any expensive work.
2. **Lens fit** on Llama-2-7B via `jlens.fit` (only if δ splits).
3. **Decompose** LoRA-TMLR-2024 adapter B-columns into the lens dictionary;
   alignment-vs-depth (o_proj/down_proj only — the residual-frame modules).
4. **Causal ablation**: zero the J-space write component, check reversion.

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
