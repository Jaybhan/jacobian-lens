import json, torch
from jlens.lens import JacobianLens
from exp.decompose.adapters import load_adapter_writes
from exp.decompose.dictionary import load_unembed
from exp.decompose.pursuit import omp

lens = JacobianLens.load("out/lens/lens.pt")
W_U, gamma = load_unembed("out/lens/unembed.pt")
adapter = load_adapter_writes("LoRA-TMLR-2024/metamath-lora-rank-16-alpha-32")

def build(J):
    rows = (W_U * gamma) @ J.float()
    return rows / rows.norm(dim=1, keepdim=True).clamp(min=1e-12)

out = {"description": "orientation A/B: metamath adapter signed-OMP@25 (S2-weighted mean over o_proj+down_proj) vs correct-J vs transposed-J dictionary. Correct must beat transposed if the frame is right.", "layers": {}}
for layer in (8, 16, 24):
    J = lens.jacobians[layer]
    row = {}
    for tag, Jx in (("correct", J), ("transposed", J.T.contiguous())):
        D = build(Jx); num = den = 0.0
        for m in ("o_proj", "down_proj"):
            w = adapter.writes[(layer, m)]
            a = omp(w.U.T, D, k=25).alignment(); s2 = w.S**2
            num += float((a*s2).sum()); den += float(s2.sum())
        row[tag] = num/den
        del D
    out["layers"][str(layer)] = row
    print(f"L{layer}: correct={row['correct']:.4f} transposed={row['transposed']:.4f}")
json.dump(out, open("out/decompose/orientation_ab.json", "w"), indent=1)
print("wrote out/decompose/orientation_ab.json")
