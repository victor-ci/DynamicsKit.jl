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

const DEFAULT_CONTINUOUS_HEAP_MB = 256

function DynamicsKit._prepare_continuous_gpu_backend(::CUDA.CUDABackend, trajectories::Int)
    requested_mb = tryparse(Int, get(ENV, "DYNAMICSKIT_CUDA_HEAP_MB", string(DEFAULT_CONTINUOUS_HEAP_MB)))
    requested_mb === nothing && throw(ArgumentError(
        "DYNAMICSKIT_CUDA_HEAP_MB must be an integer number of MiB."))
    requested_mb > 0 || throw(ArgumentError(
        "DYNAMICSKIT_CUDA_HEAP_MB must be positive; got $(requested_mb)."))
    requested_mb <= 16 * 1024 || throw(ArgumentError(
        "DYNAMICSKIT_CUDA_HEAP_MB must not exceed 16384 MiB; got $(requested_mb)."))

    requested_bytes = requested_mb * 1024^2
    heap_limit = CUDA.LIMIT_MALLOC_HEAP_SIZE
    current_bytes = CUDA.limit(heap_limit)
    current_bytes < requested_bytes && CUDA.limit!(heap_limit, requested_bytes)
    return nothing
end

end # module
