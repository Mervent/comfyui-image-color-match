# ComfyUI-Image-Color-Match

GPU-accelerated image color matching node for ComfyUI. Drop-in replacement for easy imageColorMatch with **~20-50x speedup** on 4K images.

## Features

- **Wavelet color transfer** — GPU-accelerated wavelet decomposition via `F.conv2d` with dilated convolution. Processes 4K images in <1s vs ~20s on CPU.
- **AdaIN color transfer** — GPU-accelerated adaptive instance normalization for fast per-channel style matching.
- **6 additional methods** via [color-matcher](https://pypi.org/project/color-matcher/) — `mkl`, `hm`, `reinhard`, `mvgd`, `hm-mvgd-hm`, `hm-mkl-hm`.
- Batch support — single reference image automatically broadcasts across the target batch.
- Zero PIL/numpy overhead — operates directly on ComfyUI's native BHWC tensors.

## Installation

### Manual

Clone into your `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Mervent/ComfyUI-Image-Color-Match.git
```

No additional dependencies required for `wavelet` and `adain` methods. The `color-matcher` package is automatically installed on first use of other methods.

## Usage

The node appears as **Image Color Match (Fast)** under `image/color`.

### Inputs

| Input | Type | Description |
|---|---|---|
| `image_ref` | IMAGE | Reference image to match colors from |
| `image_target` | IMAGE | Target image to recolor |
| `method` | Dropdown | Color transfer algorithm (default: `wavelet`) |
| `image_output` | Dropdown | `Preview`, `Save`, `Hide`, or `Hide/Save` |
| `save_prefix` | STRING | Filename prefix for saved images (default: `ComfyUI`) |

### Output

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | Color-matched result |

### Methods

| Method | Backend | Speed | Description |
|---|---|---|---|
| `wavelet` | GPU | ⚡ Fast | Wavelet decomposition — keeps target detail, applies source color |
| `adain` | GPU | ⚡ Fast | Adaptive instance normalization — matches per-channel statistics |
| `mkl` | CPU | Moderate | Monge-Kantorovitch linear transform |
| `hm` | CPU | Moderate | Histogram matching |
| `reinhard` | CPU | Moderate | Reinhard et al. color transfer |
| `mvgd` | CPU | Moderate | Mean-variance Gaussian distribution |
| `hm-mvgd-hm` | CPU | Moderate | Histogram matching + MVGD hybrid |
| `hm-mkl-hm` | CPU | Moderate | Histogram matching + MKL hybrid |

## Requirements

- Python ≥ 3.9
- PyTorch (included with ComfyUI)
- CUDA, MPS, or CPU (auto-detected via ComfyUI's model management)

## License

MIT
