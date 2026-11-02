# XDVioDet Checkpoint Contract

## Approved Baseline

| Field | Value |
|---|---|
| Repository commit | `f846d8cfc454943ec165bd2489b71e7dc0a1064d` |
| Checkpoint | `ckpt/wsanodet_mix2.pkl` |
| Checkpoint SHA-256 | `27067896258294da0a711e3f0c071190ea5848961298cbde389265627191ec2c` |
| Checkpoint size | 3,382,342 bytes |
| Checkpoint load status | LOADABLE |
| Loader used | `torch.load(weights_only=False)` |

## Input Contract

| Input | Representation | Dimension |
|---|---|---:|
| Visual stream | I3D RGB | 1024 |
| Audio stream | VGGish | 128 |
| Combined MIX2 input | I3D RGB + VGGish | 1152 |
| Optical flow | Not required | - |

The checkpoint requires MIX2 input. RGB features use five-crop augmentation,
whereas one audio feature file is associated with every five RGB crop files.

## Checkpoint Structure

```json
{
  "type": "dict",
  "key_count": 31,
  "keys": [
    "conv1d1.weight",
    "conv1d1.bias",
    "conv1d2.weight",
    "conv1d2.bias",
    "conv1d3.weight",
    "conv1d3.bias",
    "conv1d4.weight",
    "conv1d4.bias",
    "gc1.weight",
    "gc1.residual.weight",
    "gc1.residual.bias",
    "gc2.weight",
    "gc3.weight",
    "gc3.residual.weight",
    "gc3.residual.bias",
    "gc4.weight",
    "gc5.weight",
    "gc5.residual.weight",
    "gc5.residual.bias",
    "gc6.weight",
    "simAdj.weight0",
    "simAdj.weight1",
    "disAdj.sigma",
    "classifier.weight",
    "classifier.bias",
    "approximator.0.weight",
    "approximator.0.bias",
    "approximator.2.weight",
    "approximator.2.bias",
    "conv1d_approximator.weight",
    "conv1d_approximator.bias"
  ]
}
```

## Decision

This checkpoint is the fixed baseline event-score generator. It is not trained,
modified, or presented as the proposed contribution. Its output scores will be
calibrated and used by the selective evidence-preservation experiments.
