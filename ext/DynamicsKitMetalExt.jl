"""
Package extension registering the `:metal` GPU vendor.

Deliberately *not* an accelerator: Apple GPUs (and Metal.jl / the Metal Shading Language) have no
double-precision (Float64) hardware support at all. DynamicsKit's period/Lyapunov closure tolerances
are specified and validated in Float64, and silently downgrading a sweep to Float32 would change
periodicity/regime classifications near boundaries — exactly the "scientifically weak GPU toggle" this
feature must not ship (see `src/analysis/compute_backend.jl` and `docs/julia-package.md`).

So loading `Metal` alongside `DynamicsKit` makes `:metal` a *recognized* vendor (it appears in
`available_gpu_backends()`'s probe set) but it always reports unavailable, with a reason that names the
actual hardware limitation rather than "not installed". An explicit `gpu_backend(:metal)` request
therefore still gets a clear, specific `ArgumentError` — never a silent CPU fallback and never a
silently wrong (reduced-precision) answer.
"""
module DynamicsKitMetalExt

using DynamicsKit
using Metal

DynamicsKit._dynamicskit_gpu_available(::Val{:metal}) = false

DynamicsKit._dynamicskit_gpu_unavailable_reason(::Val{:metal}) =
    "Apple/Metal GPUs have no double-precision (Float64) hardware support, which this library " *
    "requires for scientifically valid period/Lyapunov closure comparisons; DynamicsKit does not " *
    "offer a reduced-precision (Float32) fallback. Use :cuda or :amdgpu (both FP64-capable) instead."

function DynamicsKit._dynamicskit_gpu_backend(::Val{:metal})
    throw(ArgumentError(
        "GPU backend :metal is not offered by DynamicsKit: " *
        DynamicsKit._dynamicskit_gpu_unavailable_reason(Val(:metal))
    ))
end

end # module
