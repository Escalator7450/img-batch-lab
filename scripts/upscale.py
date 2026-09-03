#!/usr/bin/env python3
"""upscale.py — RealESRGAN x4 (anime) super-resolution for the batch workflow.

CPU-only inference: torch CPU wheels + spandrel (loads the ESRGAN arch straight
from the official RealESRGAN_x4plus_anime_6B.pth weights, no basicsr needed).

Usage:
  python3 scripts/upscale.py IN_DIR OUT_DIR FACTOR WEIGHTS_PATH

Saves <stem>.png (RGB) per input image. Model scale is fixed at 4x by the
weights; FACTOR is validated against it.
"""
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 5:
        print("usage: upscale.py IN_DIR OUT_DIR FACTOR WEIGHTS_PATH", file=sys.stderr)
        return 2
    in_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    factor = int(sys.argv[3])
    weights = Path(sys.argv[4])
    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from PIL import Image
    from spandrel import ModelLoader

    device = torch.device("cpu")
    model = ModelLoader().load_from_file(str(weights))
    model.eval()
    model.to(device)
    if getattr(model, "scale", factor) != factor:
        print(f"warning: model scale {model.scale} != requested factor {factor}; using {model.scale}")
        factor = model.scale
    print(f"model: {getattr(model, 'name', 'esrgan')} scale={factor} device=cpu", flush=True)

    files = sorted(p for p in in_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    ok = 0
    for p in files:
        try:
            img = Image.open(p).convert("RGB")
            t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            with torch.no_grad():
                out = model(t.to(device))
            arr = out.squeeze(0).clamp_(0, 1).permute(1, 2, 0).numpy()
            arr = (arr * 255.0).round().astype(np.uint8)
            Image.fromarray(arr, "RGB").save(out_dir / (p.stem + ".png"))
            ok += 1
            print(f"{p.name}: {img.size[0]}x{img.size[1]} -> {arr.shape[1]}x{arr.shape[0]}", flush=True)
        except Exception as e:  # one bad image must not kill the batch
            print(f"{p.name}: FAILED {e}", flush=True)
    print(f"done: {ok}/{len(files)} upscaled", flush=True)
    return 0 if ok == len(files) and ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
