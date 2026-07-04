[Read about it here](https://dwarez.dev/blog/fused-lion-optimizer-kernel)

# Lion fused kernel

Fused fp32 Lion optimizer step built with Hugging Face `kernels`.

This repo contains only the source files needed to build, test, and benchmark the kernel. Generated files such as `setup.py`, `CMakeLists.txt`, `cmake/`, `metadata-*.json`, `_ops.py`, `registration.h`, `_cmake_build/`, and `kernel/build/` are intentionally not committed. They are produced locally by `kernel-builder`.

## Requirements

You need Python, `uv`, Rust/Cargo for `kernel-builder`, and the backend toolchain you want to build against.

For CUDA, install a CUDA-enabled PyTorch plus the CUDA toolkit. For Metal/MPS, install full Xcode plus the Metal toolchain. For CPU, no GPU toolchain is needed.

## Setup

Clone this repo and clone Hugging Face `kernels` next to it:

```bash
git clone https://github.com/huggingface/kernels.git ../kernels
```

Create the Python environment from this repo root:

```bash
uv sync --group dev
source .venv/bin/activate
```

If you use fish:

```fish
source .venv/bin/activate.fish
```

Generate the local build project:

```bash
cargo run --manifest-path ../kernels/kernel-builder/Cargo.toml -- \
  create-pyproject -f --unique-id local kernel
```

This creates the local `setup.py`, CMake files, generated Python glue, and metadata under `kernel/`.

## Build and test

Run these from the repo root after setup.

CPU:

```bash
cd kernel
CMAKE_ARGS=-DGPU_LANG=CPU python setup.py build_kernel
LION_DEVICE=cpu python -m pytest tests -v
LION_DEVICE=cpu python example.py
```

CUDA:

```bash
cd kernel
CMAKE_ARGS=-DGPU_LANG=CUDA python setup.py build_kernel
LION_DEVICE=cuda python -m pytest tests -v
LION_DEVICE=cuda python example.py
```

Metal/MPS:

```bash
cd kernel
CMAKE_ARGS=-DGPU_LANG=METAL python setup.py build_kernel
LION_DEVICE=mps python -m pytest tests -v
LION_DEVICE=mps python example.py
```

If `kernel-builder` cannot find the Metal toolchain automatically, pass it explicitly:

```bash
CMAKE_ARGS="-DGPU_LANG=METAL -DMETAL_TOOLCHAIN=/path/to/Metal.xctoolchain" \
  python setup.py build_kernel
```

## Benchmarks

From the repo root, the standard `kernels` benchmark works for local builds:

```bash
kernels benchmark ./kernel --iterations 100 --warmup 10
```

The CUDA development benchmark compares the fused kernel with eager PyTorch and `torch._foreach_*`:

```bash
cd kernel
python benchmarks/dev_benchmark.py
```

## Notes

The kernel is fp32-only. Unsupported dtypes are rejected on purpose.

`lion_foreach` is a simple Python loop over fused per-tensor launches. That is intentional: it was already fast on the measured workloads, and avoids the extra complexity of a multi-tensor native launch.

If you change PyTorch versions or switch backend builds, regenerate and rebuild:

```bash
cargo run --manifest-path ../kernels/kernel-builder/Cargo.toml -- \
  create-pyproject -f --unique-id local kernel
cd kernel
rm -rf build _cmake_build
CMAKE_ARGS=-DGPU_LANG=CUDA python setup.py build_kernel
```
