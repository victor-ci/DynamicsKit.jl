"""
Package extension registering the FP64-capable `:amdgpu` GPU vendor.

Loading `AMDGPU` alongside `DynamicsKit` makes `gpu_backend(:amdgpu)` usable when a functional ROCm
device is present. AMD GPUs support double precision (Float64) in hardware, so `:amdgpu` is a *real
accelerator* for the library's Float64 period/Lyapunov sweeps — both the discrete KernelAbstractions
kernels and the continuous DiffEqGPU `EnsembleGPUKernel` path run on `AMDGPU.ROCBackend()`.

Availability is gated on `AMDGPU.functional()`, so on a machine with `AMDGPU` installed but no working
device, `gpu_backend_available(:amdgpu)` is `false` and an explicit request raises a clear error rather
than crashing mid-kernel.
"""
module DynamicsKitAMDGPUExt

using DynamicsKit
using AMDGPU

DynamicsKit._dynamicskit_gpu_available(::Val{:amdgpu}) = AMDGPU.functional()

DynamicsKit._dynamicskit_gpu_unavailable_reason(::Val{:amdgpu}) =
    "AMDGPU.jl is loaded but reports no functional ROCm device (AMDGPU.functional() == false): check " *
    "the ROCm stack and that a supported AMD GPU is present."

DynamicsKit._dynamicskit_gpu_backend(::Val{:amdgpu}) = AMDGPU.ROCBackend()

end # module
