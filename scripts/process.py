#!/usr/bin/env python3
"""process.py — batch background removal for the `batch` workflow.

Parent: spawns a WORKER PROCESS per model so one model failing (bad URL, OOM)
does not kill the rest. Worker: load model once, infer every image in IN_DIR,
apply the edge-cleanup pass, save raw_<stem>.png.

Runs anywhere (plain `python3`, no venv paths):
  python3 scripts/process.py IN_DIR OUT_ROOT
  python3 scripts/process.py --worker MODEL IN_DIR OUT_DIR
"""
import os
import subprocess
import sys
import time
from pathlib import Path

MODELS = [
    "isnet-anime",
    "birefnet-general-lite",
    "birefnet-general",
    "birefnet-massive",
]


def active_models() -> list:
    """BATCH_MODELS="birefnet-general,isnet-anime" env override (used by the
    workflow to pick the models per matrix leg)."""
    env = os.environ.get("BATCH_MODELS", "").strip()
    return [m.strip() for m in env.split(",") if m.strip()] or list(MODELS)


def defringe(rgba_bytes: bytes) -> bytes:
    import io

    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(rgba_bytes)).convert("RGBA")
    arr = np.array(img).astype(np.float32)
    a = arr[..., 3] / 255.0
    rgb = arr[..., :3]
    unmixed = (rgb - (1 - a[..., None]) * 255.0) / np.maximum(a[..., None], 1e-4)
    af = np.clip((a - 0.35) / 0.30, 0, 1)
    af = af * af * (3 - 2 * af)
    out = np.dstack([np.clip(unmixed, 0, 255), af * 255.0]).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(out, "RGBA").save(buf, "PNG")
    return buf.getvalue()


def images_of(in_dir: Path):
    return sorted(p for p in in_dir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def worker(model: str, in_dir: Path, out_dir: Path) -> None:
    from rembg import remove, new_session

    out_dir.mkdir(parents=True, exist_ok=True)
    t_load = time.time()
    session = new_session(model)
    print(f"[{model}] weights loaded in {time.time()-t_load:.1f}s", flush=True)
    for src in images_of(in_dir):
        dst = out_dir / f"raw_{src.stem}.png"
        if dst.exists():
            print(f"[{model}] {src.stem}: exists, skip", flush=True)
            continue
        t0 = time.time()
        out = remove(src.read_bytes(), session=session)
        dst.write_bytes(defringe(out))
        print(f"[{model}] {src.stem}: {time.time()-t0:.1f}s "
              f"{dst.stat().st_size // 1024}KB", flush=True)


def main() -> None:
    if len(sys.argv) >= 5 and sys.argv[1] == "--worker":
        worker(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))
        return
    in_dir, out_root = Path(sys.argv[1]), Path(sys.argv[2])
    out_root.mkdir(parents=True, exist_ok=True)
    models = active_models()
    print(f"[parent] models: {models}", flush=True)
    for model in models:
        mdir = out_root / model
        print(f"=== {model} ===", flush=True)
        r = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker",
             model, str(in_dir), str(mdir)],
        )
        if r.returncode != 0:
            (mdir.parent / f"FAILED_{model}.txt").write_text(f"rc={r.returncode}\n")
            print(f"[parent] {model} FAILED rc={r.returncode}", flush=True)
    print("[parent] batch done", flush=True)


if __name__ == "__main__":
    main()
