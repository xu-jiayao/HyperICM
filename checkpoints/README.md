# Checkpoints

Model weights are intentionally not committed to this repository. GitHub's
regular file limit is 100 MB, while the supplied research checkpoints are
larger. Copy the required files from the external artifact directory into
this folder, or pass their external paths on the command line.

| File | Bytes | SHA-256 | Use |
| --- | ---: | --- | --- |
| `Ep10-dev0.076-0.000870-qprandom.pth` | 194,261,069 | `3131D51DD10EFDAA75DAF8A78FFE17F582B5D9B89CEA68E54210DD1F97387618` | Default base codec checkpoint |
| `Ep40-bpp0.0123-fl0.3819-ml0.0008-fb0.1-mb0.1-qp21.0-featureindex0_1_2_3_4.pth` | 194,559,642 | `36403A04DFBBA2C86500F898A431EB623B37D0B3BA37518C71C569EEE5774D53` | Optional online-adapted codec checkpoint |
| `HyperFree-b.pth` | 615,520,591 | `BF207BF0D6A64E5A5E939CFFBB850B0C72E29DDA894EF9EDA8950958688664C3` | Hyperfree feature/classification checkpoint |

PowerShell copy and verification example:

```powershell
$src = 'X:\path\to\weights'
Copy-Item -LiteralPath (Join-Path $src 'Ep10-dev0.076-0.000870-qprandom.pth') -Destination '.\checkpoints\'
Copy-Item -LiteralPath (Join-Path $src 'HyperFree-b.pth') -Destination '.\checkpoints\'
Get-FileHash -Algorithm SHA256 -LiteralPath '.\checkpoints\Ep10-dev0.076-0.000870-qprandom.pth'
```

The release `.gitignore` excludes `*.pth`; use Git LFS or an external
artifact store if weights need to be distributed with a fork.
