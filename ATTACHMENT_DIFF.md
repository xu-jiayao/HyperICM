# Attachment versus `single_image_encoder_patch.py`

This note records the differences requested for the A+2 choice. "Attachment"
means the user-provided `pasted-text.txt`; the comparison target is the
repository file `single_image_encoder_patch.py`.

| Area | Attachment | `single_image_encoder_patch.py` | Practical effect |
| --- | --- | --- | --- |
| Spectral group size | Fixed `GROUP_SIZE = 8` in training and testing | Configurable `--gop` (default 8) | A non-8 GOP changes the codec sequence; values above the 12-entry weight list can fail. |
| Distortion weights | Eight values: `[0.5, 1.2, 0.5, 0.9, 0.5, 0.9, 0.5, 0.9]` | Twelve values, alternating after the first four | With GOP 8 the first eight agree; other GOPs use a different objective. |
| Spatial padding | Pads H/W to a multiple of 16 and unpads before Hyperfree | Active train/test path uses original H/W; its padding helper is not used there | Non-multiple-of-16 inputs follow different tensor shapes and distortion accounting. |
| Exact wavelengths | Explicit `wavelengths1` (270) and `wavelengths2` (274) arrays | Same explicit arrays | No wavelength-table difference; the clean release preserves the exact arrays. |
| Empty remainder group | Training clears the list and then calls `torch.cat([])` when `C % 8 == 0` | Has an `if left_8 != 0` branch | The attachment can crash for channel counts divisible by 8; the repository file avoids that case. |
| GPU selection | Always `cuda:0`, Hyperfree `cuda:3`, DataParallel `[0,1,2]` | Branches on visible GPU count (`cuda:2`/three GPUs, otherwise `cuda:1`/two GPUs) | Neither original is portable; the release validates CLI/config devices. |
| Codec checkpoint | Hard-coded `/data2/.../Ep10...pth` | Hard-coded `/mnt/.../Ep10...pth` | Both originals ignore the declared `--checkpoint` in practice. |
| Hyperfree checkpoint | Hard-coded `/data/.../HyperFree-b.pth` | Hard-coded `/mnt/.../HyperFree-b.pth` | Both require workstation-specific edits. |
| Checkpoint loading | `strict=False`; optimizer restore disabled | Same | Missing/mismatched parameters can be silently accepted. |
| GOP 1 freezing | Always enables P-frame `encoder` after freezing all P parameters | Skips that call when `--gop 1` | GOP 1 leaves all P-frame parameters frozen instead of following the attachment rule. |
| Full-image cadence | `epoch % 10 == 0` (zero-based), or final epoch | `epoch % 20 == 0` (zero-based), or final epoch | Evaluation happens at different epochs. |
| Evaluator call | Calls `run_eval` without a GOP argument | Passes `gop=args.gop` | Decoding grouping can differ. |
| Per-epoch workflow | Train, test, scheduler, checkpoint; periodic codec/semantic evaluation | Same overall workflow | This part is shared. |
| Filename field | Literal `ave_bpp`, but writes QP | Same label; writes average QP | The field is mislabeled in both originals. |

## Deliberate clean-release changes

- `train.py` and `inference.py` use package-relative imports and explicit paths.
- Encoder-only freezing follows the attachment and empty remainder groups are guarded.
- Reconstructed full-group bands are restored in spectral order (the original
  `cat(dim=0)`/`view` sequence scrambled bands whenever more than one full
  group was processed together).
- Checkpoint loading is strict, with `module.` prefix handling.
- Hyperfree loads from `--hyperfree-checkpoint`; no model is created at import time.
- `inference.py` uses real CompressAI entropy coding and reports byte-derived `bpsp`.
- Exact attachment wavelengths are embedded in `hypericm/datasets.py`.
- Image rasters reject nodata/masked and non-finite pixels; label nodata becomes background 0.
- `--qp 0` draws a QP per batch with an isolated seeded random stream rather
  than mutating the function argument after the first draw.
- Hard-coded GPU, data, checkpoint, evaluator, and output paths are removed.

For strict comparison, training still uses the attachment-compatible
likelihood surrogate (latent-element normalization and original group
aggregation), while inference `bpsp` is compressed bytes per input pixel.
