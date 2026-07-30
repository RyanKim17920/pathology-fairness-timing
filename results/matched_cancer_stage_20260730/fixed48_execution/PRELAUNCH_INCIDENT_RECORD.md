# Fixed-48 prelaunch incident record

Date: 2026-07-30 UTC

This record covers three seed-32001 canary allocations that failed before any
calibration training, diagnostic fitting, prediction export, or inferential
analysis. Their logs and rejected source manifests are retained. No output
from these allocations is eligible for a seed-success receipt.

## Failed allocations

- Job `368174`, 15:37:33–15:37:54 UTC, failed after 21 seconds because the
  controller directly executed the packaged seed worker and Python could not
  resolve the repository-level `tools` package. It created no attempt
  directory. The worker is now invoked through the frozen venv interpreter
  with `-m tools.matched_cancer_fixed48_20260730.seed_worker`.
- Job `368175`, 15:44:14–15:44:42 UTC, failed after 28 seconds because the
  calibration driver required `SLURM_GPUS_PER_TASK`, which this cluster does
  not export. It created no attempt directory. The driver now enforces the
  observed one-GPU contract using `SLURM_GPUS_ON_NODE`, `SLURM_JOB_GPUS`,
  `CUDA_VISIBLE_DEVICES`, `SLURM_NTASKS`, and
  `torch.cuda.device_count()`.
- Job `368176`, 15:49:50–15:54:48 UTC, failed after 4 minutes 58 seconds
  because the pretrained checkpoint under the volatile user cache disappeared
  after attempt preparation and replay-manifest construction. It created
  `calibration/seed_32001/attempt_01` containing only the calibration contract,
  three preparation configs, and the replay manifest. It created no run
  directory, effective run config, learned checkpoint, metric stream,
  diagnostic attempt, prediction, or success receipt. This immutable partial
  attempt remains failure evidence; the next canary must use `attempt_02`.

## Durable checkpoint recovery

The checkpoint was recovered from the unchanged official DINOv2 URL and
accepted only after matching the originally frozen identity:

- bytes: `88,291,785`
- file SHA-256:
  `f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb`
- encoder-state SHA-256:
  `ba9418ed2138e42250085b04e0502d621b072c4bb60240f2845a27fbf3184bd6`

It is staged read-only at:

`/data/ryan.kim/nanopath/reruns/matched_cancer_fixed48_20260730/control/torch_home/hub/checkpoints/dinov2_vits14_reg4_pretrain.pth`

The containing checkpoint directory is non-writable. The calibration
contract, independent auditor, source manifest, and `TORCH_HOME` all bind this
same path. The driver rehashes the file at allocation startup and immediately
before each of the five serial training commands. A network-disabled strict
model-load smoke test reproduced the frozen encoder-state hash.

This is an operational path hardening only. Checkpoint bytes, model
architecture, seeds, data, objectives, exposures, diagnostics, estimands, and
statistical rules are unchanged.
