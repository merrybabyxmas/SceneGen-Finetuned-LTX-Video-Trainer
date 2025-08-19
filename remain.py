import json, os
from pathlib import Path

dataset = Path("./data/videos/dataset.json")
precomp = dataset.parent / ".precomputed"
latents_root = precomp / "latents"

items = json.load(open(dataset, "r", encoding="utf-8"))
remaining = []
for e in items:
    rel = Path(e["media_path"]).with_suffix(".pt")   # media_path → .pt
    out = latents_root / rel
    if not out.exists():
        remaining.append(e)

out_file = dataset.parent / "dataset_remaining.json"
json.dump(remaining, open(out_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"Total: {len(items)}, missing latents: {len(remaining)} -> {out_file}")