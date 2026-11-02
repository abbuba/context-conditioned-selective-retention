# XDVioDet CPU Compatibility Patch

## Why the Patch Was Required

The official XDVioDet implementation contained a hard-coded CUDA placement in
`layers.py`:

```python
self.dist = torch.from_numpy(squareform(dist)).to('cuda')
```

The local environment uses PyTorch 2.14.1 with CPU execution and does not
provide CUDA. This caused inference to fail before score generation.

## Applied Change

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
self.dist = torch.from_numpy(squareform(dist)).to(device)
```

## Scope

| Item | Modified |
|---|---|
| Model checkpoint weights | No |
| Model architecture | No |
| Input features | No |
| Event-score calculation | No |
| Device-selection logic | Yes |

The original file was preserved at:

```text
C:\Users\abbub\OneDrive\Desktop\Experiment\cctv-selective-retention\external\XDVioDet\layers_original_cuda.py
```

Original source SHA-256:

```text
373f7b937cdda8fd74a98f07d90490f988c3a355d3b278875682baec93a36a4f
```

Patched source SHA-256:

```text
4e5333364fff7e657633bda67b2410be1badb1faaad6746daa2da6e3c3c799a9
```
