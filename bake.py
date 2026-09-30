# Download one public HF file into an HF-cache tree (so hf_hub_download finds it offline-fast) and tar it as a layer.
import os, sys, tarfile
from huggingface_hub import hf_hub_download
repo, fn, out = sys.argv[1], sys.argv[2], sys.argv[3]
root = "/mnt/hfroot"
cache = f"{root}/root/.cache/huggingface/hub"
os.makedirs(cache, exist_ok=True)
p = hf_hub_download(repo, fn, cache_dir=cache)
print("downloaded", p, os.path.getsize(p) / 1e9, "GB", flush=True)
d = f"models--{repo.replace('/', '--')}"
with tarfile.open(out, "w") as t:  # symlinks kept (snapshot -> blob), exactly the HF cache layout
    t.add(f"{cache}/{d}", arcname=f"root/.cache/huggingface/hub/{d}")
os.system(f"rm -rf {root}")
print("layer", out, os.path.getsize(out) / 1e9, "GB", flush=True)
