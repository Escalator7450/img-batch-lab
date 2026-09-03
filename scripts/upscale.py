#!/usr/bin/env python3
"""upscale.py — Real-ESRGAN ×4 batch upscaler for the `upscale` workflow.

Model: realesr-animevideov3 (SRVGGNetCompact) — anime/illustration-optimized,
compact (~2.5MB), fast on CPU runners. ×4 outscale by default.

Runs standalone:  python3 scripts/upscale.py IN_DIR OUT_DIR [--scale 4]
"""
import os
import sys
import time
import urllib.request
from pathlib import Path

MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
MODEL_FILE = "/tmp/realesr-animevideov3.pth"


def main() -> None:
    in_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    scale = 4
    if len(sys.argv) >= 5 and sys.argv[3] == "--scale":
        scale = int(sys.argv[4])
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(MODEL_FILE).exists():
        t0 = time.time()
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print(f"weights downloaded in {time.time()-t0:.1f}s", flush=True)

    import numpy as np
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact

    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_conv=32, upscale=4, act_type="prelu")
    upsampler = RealESRGANer(scale=4, model_path=MODEL_FILE, model=model,
                             tile=0, tile_pad=10, pre_pad=0, half=False,
                             device="cpu")

    files = sorted(p for p in in_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for src in files:
        dst = out_dir / f"up_{src.stem}.png"
        if dst.exists():
            print(f"{src.stem}: exists, skip", flush=True)
            continue
        t0 = time.time()
        from PIL import Image
        img = Image.open(src).convert("RGB")
        arr = np.array(img)
        out, _ = upsampler.enhance(arr, outscale=scale)
        Image.fromarray(out).save(dst)
        print(f"{src.stem}: {img.size} → {out.shape[1]}x{out.shape[0]} "
              f"({time.time()-t0:.1f}s)", flush=True)
    print(f"→ {len(files)} upscaled ×{scale} → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
