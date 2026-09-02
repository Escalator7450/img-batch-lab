# img-batch-lab

Personal playground for running open-source image segmentation (background
removal) models as batch jobs on free GitHub Actions runners.

The runner installs [`rembg`](https://github.com/danielgatis/rembg) and
processes a folder of images with one model per matrix leg. Handy when the
bigger ONNX models don't fit in a laptop's RAM.

## Layout

- `scripts/process.py` — worker: loads one model, infers every image in the
  input folder, applies a soft-edge cleanup pass, writes PNGs with alpha.
- `.github/workflows/batch.yml` — orchestrates a run: one matrix job per
  model, artifacts as backup, results published to the `out` branch.

## Usage

1. Commit the inputs to `data/<run-id>/payload.zip` (a plain zip of images).
2. Fire a `repository_dispatch` (event type `batch`) with:

   ```json
   {"id": "run-001", "models": ["isnet-anime", "birefnet-general"]}
   ```

   `models` is optional (defaults to all bundled models). You can also run it
   manually from the Actions tab with `workflow_dispatch` and the same JSON.

3. Collect results from the `out` branch under `results/<run-id>/<model>/`
   (`raw_<stem>.png`, RGBA), or from the run's artifacts (expire after a week).

Notes: inputs travel with neutral names (`case01.png`, `case02.png`, ...);
keep your own mapping between those names and the originals. The `out` branch
mirrors them (`raw_case01.png`)
