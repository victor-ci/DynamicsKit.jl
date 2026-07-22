"""
Package extension registering the FP64-capable `:cuda` GPU vendor.

Loading `CUDA` alongside `DynamicsKit` makes `gpu_backend(:cuda)` usable when a functional CUDA device
is present. Unlike Apple/Metal, NVIDIA GPUs support double precision (Float64) in hardware, so `:cuda`
is a *real accelerator* for the library's Float64 period/Lyapunov sweeps — both the discrete
KernelAbstractions kernels and the continuous DiffEqGPU `EnsembleGPUKernel` path run on
`CUDA.CUDABackend()`.

Availability is gated on `CUDA.functional()` (driver + device actually usable), so on a machine with
`CUDA` installed but no working device, `gpu_backend_available(:cuda)` is `false` and an explicit
request raises a clear error rather than crashing mid-kernel.
"""
module DynamicsKitCUDAExt

using DynamicsKit
using CUDA

DynamicsKit._dynamicskit_gpu_available(::Val{:cuda}) = CUDA.functional()

DynamicsKit._dynamicskit_gpu_unavailable_reason(::Val{:cuda}) =
    "CUDA.jl is loaded but reports no functional CUDA device (CUDA.functional() == false): check the " *
    "NVIDIA driver and that a CUDA-capable GPU is present."

DynamicsKit._dynamicskit_gpu_backend(::Val{:cuda}) = CUDA.CUDABackend()

end # module
