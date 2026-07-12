#!/usr/bin/env bash
# Phase 0 bootstrap for a RunPod- or Vast.ai-style PyTorch pod
# (CUDA 12.x/13.x, Python 3.10-3.13).
#
#   1. rsync/git-clone this repo onto the pod, cd into it
#   2. export HF_TOKEN=hf_...   (account with starcoderdata terms accepted)
#   3. bash exp/setup_runpod.sh
#
# Ends with a CPU test pass and a 1-prompt GPU smoke of grad-through-
# activations on Llama-2-7B — the two things that must work before spending
# on the delta run (Phase 1) or the lens fit (Phase 2).
set -euo pipefail

# Vast.ai PyTorch images ship torch preinstalled in /venv/main (see the image's
# /etc/vast-agents-guide.md). Activate it if present and not already active so we
# install into the env that already has a Blackwell-compatible torch, rather than
# a fresh system env. Harmless no-op on RunPod / other images.
if [[ -z "${VIRTUAL_ENV:-}" && -f /venv/main/bin/activate ]]; then
  echo "== activating /venv/main =="
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
fi

echo "== python / torch sanity =="
python - <<'PY'
import sys
version = sys.version_info
assert (3, 10) <= version[:2] <= (3, 13), f"need py3.10-3.13, got {sys.version}"
import torch
assert torch.cuda.is_available(), "no CUDA device visible"
print("python", sys.version.split()[0], "| torch", torch.__version__,
      "| gpu", torch.cuda.get_device_name(0))
PY

echo "== install =="
# Prefer uv (Vast images ship it and it's ~10x faster); fall back to pip.
if command -v uv >/dev/null 2>&1; then
  uv pip install --quiet -e ".[dev]" datasets matplotlib
else
  pip install --quiet -e ".[dev]" datasets matplotlib
fi

echo "== hf auth =="
python - <<'PY'
import os
assert os.environ.get("HF_TOKEN"), "export HF_TOKEN first"
from huggingface_hub import whoami
print("hf user:", whoami()["name"])
PY

echo "== CPU tests =="
python -m pytest tests/ exp/tests/ -q

echo "== GPU smoke: 1-prompt jacobian on Llama-2-7B (eager attention) =="
python - <<'PY'
import torch, transformers, jlens
from jlens.fitting import jacobian_for_prompt

name = "NousResearch/Llama-2-7b-hf"
hf = transformers.AutoModelForCausalLM.from_pretrained(
    name, torch_dtype=torch.bfloat16, attn_implementation="eager"
).cuda()
tok = transformers.AutoTokenizer.from_pretrained(name)
model = jlens.from_hf(hf, tok)
print(model)

prompt = ("The quick brown fox jumps over the lazy dog. " * 6).strip()
jacobians, seq_len, n_valid = jacobian_for_prompt(
    model, prompt, source_layers=[15], dim_batch=8, max_seq_len=64
)
J = jacobians[15]
assert torch.isfinite(J).all(), "non-finite Jacobian entries"
norm = J.norm().item() / (model.d_model ** 0.5)
print(f"seq_len={seq_len} n_valid={n_valid} ||J_15||/sqrt(d)={norm:.3f}")
assert norm > 1e-3, "Jacobian suspiciously near zero - grads not flowing?"
print("GPU smoke OK")
PY

cat <<'EOF'

Ready. Next (Phase 1, the delta screen):

  python -m exp.delta.run_extract --n-examples 2000 --out out/delta \
      --conditions code-ift math-ift code-cpt math-cpt wikitext \
                   code-ift-alltok math-ift-alltok
  python -m exp.delta.analyze --dir out/delta

Then copy out/delta/ off the pod before shutting it down.
EOF
